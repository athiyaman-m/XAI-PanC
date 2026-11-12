try:
    from .organ_models import Organ, Lesion
    from .mesh_utils import ellipsoid_mesh, tube_mesh, merge_meshes, compute_vertex_normals
    from .anatomy_loader import load_organ_mesh
except Exception:  # pragma: no cover - optional runtime dependencies
    Organ = Lesion = None
    ellipsoid_mesh = tube_mesh = merge_meshes = compute_vertex_normals = None
    load_organ_mesh = None

try:
    from .ct_mesh_pipeline import (
        discover_ct_candidates,
        load_real_ct_volume,
        create_pancreas_surface_from_volume,
        build_tumor_pancreas_mesh,
        load_pancreas_mesh_from_ct_or_fallback,
    )
except Exception:  # pragma: no cover - optional runtime dependencies
    discover_ct_candidates = load_real_ct_volume = create_pancreas_surface_from_volume = None
    build_tumor_pancreas_mesh = load_pancreas_mesh_from_ct_or_fallback = None

try:
    from .lesion_mapper import create_lesion_mesh, create_metastasis_chain
    from .tumor_engine import TumorEngine
    from .coordinate import to_mm, convert_units, voxel_to_world
except Exception:  # pragma: no cover - optional runtime dependencies
    create_lesion_mesh = create_metastasis_chain = TumorEngine = None
    to_mm = convert_units = voxel_to_world = None

try:
    from .models import PatientVolume, PancreasModel, TumorState, TumorModel
except Exception:  # pragma: no cover - optional runtime dependencies
    PatientVolume = PancreasModel = TumorState = TumorModel = None

try:
    from .ct_volume import load_patient_volume
except Exception:  # pragma: no cover - optional runtime dependencies
    load_patient_volume = None

try:
    from .segmentation import SegmentationBackend
except Exception:  # pragma: no cover - optional runtime dependencies
    SegmentationBackend = None

try:
    from .mesh_builder import mesh_from_mask, mesh_from_labelmap
except Exception:  # pragma: no cover - optional runtime dependencies
    mesh_from_mask = mesh_from_labelmap = None

try:
    from .region_mapper import PancreasRegionMapper, tumour_seed_point
except Exception:  # pragma: no cover - optional runtime dependencies
    PancreasRegionMapper = tumour_seed_point = None

try:
    from .organ_renderer import (
        RENDER_MODES,
        _organ_to_mesh3d,
        _torso_outline,
        render_normal,
        render_risk_heatmap,
        render_disease_progression,
        render_blood_supply,
        render_lymphatic_spread,
        render_tumour_localisation,
        render_treatment_simulation,
        render_compartmental_twin,
    )
except Exception:  # pragma: no cover - optional runtime dependencies
    RENDER_MODES = _organ_to_mesh3d = _torso_outline = None
    render_normal = render_risk_heatmap = render_disease_progression = None
    render_blood_supply = render_lymphatic_spread = render_tumour_localisation = None
    render_treatment_simulation = render_compartmental_twin = None
