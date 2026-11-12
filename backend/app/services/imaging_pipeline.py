"""DICOM series segmentation and mesh-export pipeline.

This module deliberately never creates a clinical prediction from slice counts or
image intensities.  A study is only marked ``prediction_ready`` after a configured
MONAI or nnU-Net model has produced a label map.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class Volume:
    data: np.ndarray
    spacing: tuple[float, float, float]
    source: str = "dicom"
    label: str | None = None


def load_dicom_series(study_dir: Path) -> Volume:
    """Load one uncompressed CT/MR DICOM series and preserve physical spacing."""
    try:
        import pydicom
    except ImportError as exc:
        raise PipelineError("pydicom is not installed; install backend requirements") from exc

    datasets = []
    for path in study_dir.rglob("*"):
        if not path.is_file() or path.name == "study.json":
            continue
        try:
            ds = pydicom.dcmread(path, force=True)
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        except Exception:
            continue
    if len(datasets) < 2:
        raise PipelineError("No valid multi-slice DICOM series was found in the uploaded folder")

    # A DICOM-SEG object is already an acquired/annotated volumetric label.
    # Treat it as such instead of trying to run an intensity-based model on it.
    segs = [ds for ds in datasets if str(getattr(ds, "Modality", "")).upper() == "SEG"]
    if segs:
        # Whole-slide microscopy packages can contain hundreds of thousands of
        # tiled frames. Select the largest renderable annotation object; this
        # prevents treating a tiled pathology slide as a CT z-stack.
        renderable = [item for item in segs if int(getattr(item, "NumberOfFrames", 1)) <= 512]
        ds = max(renderable or segs, key=lambda item: int(getattr(item, "NumberOfFrames", 1)))
        frames = ds.pixel_array.astype(np.uint8)
        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]
        # A single-frame SEG is a real 2D annotation. Give it a minimal depth
        # so marching-cubes can create an extruded visual without inventing a
        # volumetric clinical measurement.
        if frames.shape[0] == 1:
            frames = np.repeat(frames, 2, axis=0)
        spacing_xy = getattr(ds, "PixelSpacing", [1.0, 1.0])
        z = float(getattr(ds, "SliceThickness", 1.0))
        if hasattr(ds, "SharedFunctionalGroupsSequence"):
            group = ds.SharedFunctionalGroupsSequence[0]
            if hasattr(group, "PixelMeasuresSequence"):
                measures = group.PixelMeasuresSequence[0]
                spacing_xy = getattr(measures, "PixelSpacing", spacing_xy)
                z = float(getattr(measures, "SliceThickness", z))
        label = None
        if hasattr(ds, "SegmentSequence") and ds.SegmentSequence:
            label = str(getattr(ds.SegmentSequence[0], "SegmentLabel", "DICOM segmentation"))
        return Volume(frames, (z, float(spacing_xy[0]), float(spacing_xy[1])), source="dicom_seg", label=label)

    series = {}
    for ds in datasets:
        series.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
    slices = max(series.values(), key=len)
    if len(slices) < 2:
        raise PipelineError("The uploaded folder does not contain a complete DICOM series")

    def position(ds: Any) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        return float(ipp[2]) if ipp is not None else float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=position)
    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    if any(int(ds.Rows) != rows or int(ds.Columns) != cols for ds in slices):
        raise PipelineError("DICOM series has inconsistent slice dimensions")

    pixels = []
    for ds in slices:
        image = ds.pixel_array.astype(np.float32)
        pixels.append(image * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0)))
    spacing_xy = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    z = abs(position(slices[1]) - position(slices[0])) or float(getattr(slices[0], "SliceThickness", 1.0))
    return Volume(np.stack(pixels), (float(z), float(spacing_xy[0]), float(spacing_xy[1])))


def infer_with_monai(volume: Volume, checkpoint: Path) -> np.ndarray:
    """Run a TorchScript MONAI segmentation model with MONAI preprocessing."""
    try:
        import torch
        from monai.inferers import sliding_window_inference
        from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensityRange
    except ImportError as exc:
        raise PipelineError("MONAI and PyTorch are required for MONAI inference") from exc
    if not checkpoint.is_file():
        raise PipelineError(f"MONAI checkpoint is not configured: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ])
    image = transform(volume.data).unsqueeze(0).to(device)
    try:
        model = torch.jit.load(str(checkpoint), map_location=device).eval()
    except Exception as exc:
        raise PipelineError("MONAI checkpoint must be a TorchScript segmentation model") from exc
    with torch.no_grad():
        logits = sliding_window_inference(image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def infer_with_nnunet(volume: Volume, model_dir: Path, work_dir: Path) -> np.ndarray:
    """Run nnU-Net v2. The configured model must be a trained nnU-Net results folder."""
    try:
        import SimpleITK as sitk
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise PipelineError("nnU-Net v2 and SimpleITK are required for nnU-Net inference") from exc
    if not model_dir.is_dir():
        raise PipelineError(f"nnU-Net model directory is not configured: {model_dir}")
    image = sitk.GetImageFromArray(volume.data)
    image.SetSpacing(tuple(reversed(volume.spacing)))
    input_file = work_dir / "nnunet_input_0000.nii.gz"
    output_dir = work_dir / "nnunet_output"
    sitk.WriteImage(image, str(input_file))
    predictor = nnUNetPredictor(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,))
    predictor.predict_from_files([[str(input_file)]], str(output_dir), save_probabilities=False, overwrite=True)
    result = sitk.ReadImage(str(output_dir / "nnunet_input.nii.gz"))
    return sitk.GetArrayFromImage(result).astype(np.uint8)


def export_mesh(mask: np.ndarray, spacing: tuple[float, float, float], output: Path) -> dict[str, Any]:
    """Convert non-background tumour labels to a renderer-neutral triangle mesh."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise PipelineError("scikit-image is required to export the 3D surface") from exc
    tumour = mask > 0
    if not tumour.any():
        raise PipelineError("The model returned no tumour label; no prediction mesh was generated")
    # Bound server memory for large slides/volumes while retaining physical
    # extent. A 3D preview does not need every source voxel.
    stride = max(1, int(np.ceil((tumour.size / 12_000_000) ** (1 / 3))))
    if stride > 1:
        tumour = tumour[::stride, ::stride, ::stride]
        spacing = tuple(value * stride for value in spacing)
    verts, faces, _, _ = marching_cubes(tumour.astype(np.uint8), level=0.5, spacing=spacing)
    # Keep JSON manageable for browser transfer while retaining the overall surface.
    if len(faces) > 120_000:
        faces = faces[::max(1, len(faces) // 120_000)]
    # Keep the mesh in the volume's physical frame (voxel * spacing) so the
    # tumour surface aligns with the CT volume in the SAME coordinate system.
    # Re-centring to the origin would break that alignment.
    mesh = {
        "positions": verts.astype(float).ravel().tolist(),
        "indices": faces.astype(int).ravel().tolist(),
        "units": "mm",
        "origin_mm": [0.0, 0.0, 0.0],
    }
    output.write_text(json.dumps(mesh), encoding="utf-8")
    voxel_volume = float(np.prod(spacing))
    return {"mesh_file": output.name, "vertices": int(len(verts)), "triangles": int(len(faces)), "tumour_volume_cm3": round(float(tumour.sum()) * voxel_volume / 1000, 3)}


def run_pipeline(study_dir: Path, engine: str) -> dict[str, Any]:
    volume = load_dicom_series(study_dir)
    if volume.source == "dicom_seg":
        mask = (volume.data > 0).astype(np.uint8)
        np.save(study_dir / "segmentation.npy", mask)
        mesh = export_mesh(mask, volume.spacing, study_dir / "tumour_mesh.json")
        return {
            "engine": "dicom-seg", "model": "Uploaded DICOM SEG (no AI inference)",
            "source_label": volume.label or "DICOM segmentation",
            "volume_shape": list(volume.data.shape), "spacing_mm": list(volume.spacing), **mesh,
        }
    models_root = Path(os.getenv("IMAGING_MODELS_DIR", "models/imaging"))
    if engine == "nnunet":
        mask = infer_with_nnunet(volume, Path(os.getenv("NNUNET_MODEL_DIR", models_root / "nnunet")), study_dir)
        model_name = "nnU-Net v2"
    else:
        mask = infer_with_monai(volume, Path(os.getenv("MONAI_CHECKPOINT", models_root / "monai_pancreas.ts")))
        model_name = "MONAI TorchScript"
    np.save(study_dir / "segmentation.npy", mask)
    mesh = export_mesh(mask, volume.spacing, study_dir / "tumour_mesh.json")
    return {"engine": engine, "model": model_name, "volume_shape": list(volume.data.shape), "spacing_mm": list(volume.spacing), **mesh}
