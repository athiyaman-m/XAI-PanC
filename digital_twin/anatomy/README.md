Medical mesh strategy
- Primary: load real pancreatic CT volumes from DATASETS or backend uploads and reconstruct a surface using marching cubes.
- Fallback: generate a pancreas-like anatomical mesh with a tumor nodule when no CT volume is available.
- This replaces the earlier purely procedural pancreas mesh with a CT-aware path that is more realistic and data-driven.
