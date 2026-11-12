'use client';

import { Suspense, useMemo, useState, useEffect, useRef } from 'react';
import React from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface TwinMesh {
  name: string;
  color: string;
  opacity?: number;
  kind?: 'solid' | 'lines';
  legend?: boolean;
  layer?: 'ground-truth' | 'scenario';
  colors?: number[];
  positions: number[];
  indices?: number[];
}

export interface TwinPayload {
  mode: string;
  title?: string;
  organs?: TwinMesh[];
  meta?: Record<string, unknown>;
}

interface TwinMeshViewerProps {
  url?: string;
  directPayload?: TwinPayload | null;
  biomarkerTumorCm?: number;
  prob?: number;
}

// ── Fallback UI ───────────────────────────────────────────────────────────────
function ViewerFallback({ text }: { text: string }) {
  return (
    <div style={{
      width: '100%', height: '100%', minHeight: 420, display: 'flex',
      alignItems: 'center', justifyContent: 'center', background: '#09111a',
      borderRadius: 8, color: '#718e9a', fontSize: 13,
    }}>
      {text}
    </div>
  );
}

class WebGLBoundary extends React.Component<{ children: React.ReactNode; fallback: React.ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() { return this.state.hasError ? this.props.fallback : this.props.children; }
}

// ── Mesh helpers ──────────────────────────────────────────────────────────────
function tumorDiameterCm(positions: number[]): number {
  if (positions.length < 3) return 0;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const diamMm = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  return diamMm / 10;
}

function meshCentroid(positions: number[]): THREE.Vector3 {
  if (positions.length < 3) return new THREE.Vector3();
  let cx = 0, cy = 0, cz = 0, n = positions.length / 3;
  for (let i = 0; i < positions.length; i += 3) {
    cx += positions[i]; cy += positions[i + 1]; cz += positions[i + 2];
  }
  return new THREE.Vector3(cx / n, cy / n, cz / n);
}

// ══════════════════════════════════════════════════════════════════════════════
// Anatomy-Aware 3D Vertex-Displacement Tumor Growth Shell
// Based on: "Anatomy-Aware 3D Mesh Refinement" (arXiv:2607.19210)
//
// The paper's technique:
//   1. Construct a 3D vector field from vertex normals
//   2. Iteratively push vertices outward/inward along their surface normals
//   3. Apply Laplacian smoothing for geometric regularisation
//   4. The original mesh is NEVER modified — displacement is a separate layer
//
// Our adaptation for tumor growth visualization:
//   - biomarker severity → force magnitude (displacement distance)
//   - vertex normals → displacement direction (outward from tumor surface)
//   - Laplacian smoothing → prevents spiky/irregular artifacts
//   - The ground-truth red tumor mesh stays at 1:1 scale ALWAYS
// ══════════════════════════════════════════════════════════════════════════════

function buildAdjacency(indexArray: number[] | Uint16Array | Uint32Array, vertexCount: number): Map<number, Set<number>> {
  const adj = new Map<number, Set<number>>();
  for (let i = 0; i < vertexCount; i++) adj.set(i, new Set());
  for (let i = 0; i < indexArray.length; i += 3) {
    const a = indexArray[i], b = indexArray[i + 1], c = indexArray[i + 2];
    adj.get(a)!.add(b); adj.get(a)!.add(c);
    adj.get(b)!.add(a); adj.get(b)!.add(c);
    adj.get(c)!.add(a); adj.get(c)!.add(b);
  }
  return adj;
}

function laplacianSmooth(
  positions: Float32Array,
  adjacency: Map<number, Set<number>>,
  lambda: number = 0.5,
  iterations: number = 2,
) {
  const temp = new Float32Array(positions.length);
  for (let iter = 0; iter < iterations; iter++) {
    temp.set(positions);
    for (const [vi, neighbors] of adjacency) {
      if (neighbors.size === 0) continue;
      let cx = 0, cy = 0, cz = 0;
      for (const ni of neighbors) {
        cx += positions[ni * 3];
        cy += positions[ni * 3 + 1];
        cz += positions[ni * 3 + 2];
      }
      const n = neighbors.size;
      temp[vi * 3]     = positions[vi * 3]     + lambda * (cx / n - positions[vi * 3]);
      temp[vi * 3 + 1] = positions[vi * 3 + 1] + lambda * (cy / n - positions[vi * 3 + 1]);
      temp[vi * 3 + 2] = positions[vi * 3 + 2] + lambda * (cz / n - positions[vi * 3 + 2]);
    }
    positions.set(temp);
  }
}

// ── Growth shell material ─────────────────────────────────────────────────────
// NO animation / pulsing / glowing. Solid static red surface (#ff3b3b) that
// looks like the tumor region spreading or shrinking. Only changes shape
// when biomarker input values are adjusted.

interface AnatomyAwareTumorShellProps {
  sourceGeometry: THREE.BufferGeometry;
  scaleFactor: number;
  prob: number;
  centroid: THREE.Vector3;
}

function AnatomyAwareTumorShell({
  sourceGeometry, scaleFactor, prob, centroid,
}: AnatomyAwareTumorShellProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { shellGeo, basePositions, baseNormals, adjacency } = useMemo(() => {
    const geo = sourceGeometry.clone();
    geo.computeVertexNormals();
    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute;
    const normAttr = geo.getAttribute('normal') as THREE.BufferAttribute;
    const basePosArray = new Float32Array(posAttr.array);
    const baseNormArray = new Float32Array(normAttr.array);

    let adj: Map<number, Set<number>>;
    const idx = geo.index;
    if (idx) {
      adj = buildAdjacency(idx.array as Uint16Array | Uint32Array, posAttr.count);
    } else {
      const triCount = posAttr.count;
      adj = new Map();
      for (let i = 0; i < triCount; i++) adj.set(i, new Set());
      for (let i = 0; i < triCount; i += 3) {
        if (i + 2 < triCount) {
          adj.get(i)!.add(i + 1); adj.get(i)!.add(i + 2);
          adj.get(i + 1)!.add(i); adj.get(i + 1)!.add(i + 2);
          adj.get(i + 2)!.add(i); adj.get(i + 2)!.add(i + 1);
        }
      }
    }
    return { shellGeo: geo, basePositions: basePosArray, baseNormals: baseNormArray, adjacency: adj };
  }, [sourceGeometry]);

  const targetRef = useRef({ displacement: 0, prob: 0 });
  useEffect(() => {
    const maxDisplacementMm = 12.0;
    const displacementMm = (scaleFactor - 1.0) * maxDisplacementMm;
    targetRef.current = { displacement: displacementMm, prob };
  }, [scaleFactor, prob]);

  const currentDisplacement = useRef(0);

  // Only update geometry when displacement is transitioning toward target.
  // Once converged, stop updating (no continuous animation/pulsing).
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const target = targetRef.current.displacement;
    const diff = Math.abs(currentDisplacement.current - target);
    // Skip update if already converged (within 0.01mm)
    if (diff < 0.01) return;

    currentDisplacement.current = THREE.MathUtils.lerp(
      currentDisplacement.current, target, 0.08,
    );

    const d = currentDisplacement.current;
    const posAttr = shellGeo.getAttribute('position') as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;
    const vertexCount = posAttr.count;

    // Displace each vertex along its surface normal (Vector Field Force)
    for (let i = 0; i < vertexCount; i++) {
      const i3 = i * 3;
      const nx = baseNormals[i3], ny = baseNormals[i3 + 1], nz = baseNormals[i3 + 2];
      posArray[i3]     = basePositions[i3]     + d * nx;
      posArray[i3 + 1] = basePositions[i3 + 1] + d * ny;
      posArray[i3 + 2] = basePositions[i3 + 2] + d * nz;
    }

    // Laplacian smoothing (Geometric Regularisation Force)
    if (Math.abs(d) > 0.1) {
      const smoothingStrength = Math.min(0.4, Math.abs(d) / 20);
      laplacianSmooth(posArray, adjacency, smoothingStrength, 1);
    }

    posAttr.needsUpdate = true;
    shellGeo.computeVertexNormals();
    shellGeo.computeBoundingBox();
    shellGeo.computeBoundingSphere();
  });

  const showShell = Math.abs(scaleFactor - 1.0) > 0.01;

  if (!showShell) return null;

  // Solid static red surface — same color as the ground-truth tumor (#ff3b3b)
  // so the growth looks like the red tumor region itself is spreading/shrinking.
  // No animation, no pulsing, no glowing.
  return (
    <mesh ref={meshRef} geometry={shellGeo} frustumCulled={false} renderOrder={3}>
      <meshPhysicalMaterial
        color="#ff3b3b"
        emissive="#ff3b3b"
        emissiveIntensity={0.3}
        transparent
        opacity={0.7}
        roughness={0.3}
        metalness={0.1}
        side={THREE.DoubleSide}
        depthWrite={false}
        depthTest
      />
    </mesh>
  );
}

// ── Geometry builder ──────────────────────────────────────────────────────────
function buildGeometries(organs: TwinMesh[]): THREE.BufferGeometry[] {
  return organs.map((m) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3));
    if (m.colors && m.colors.length === m.positions.length) {
      geo.setAttribute('color', new THREE.Float32BufferAttribute(m.colors, 3));
    }
    if (m.indices && m.indices.length) {
      geo.setIndex(m.indices);
    }
    if (m.kind !== 'lines') {
      geo.computeVertexNormals();
      geo.computeBoundingBox();
      geo.computeBoundingSphere();
    }
    return geo;
  });
}

const isTumorMesh = (m: TwinMesh) => /Tumou?r|Cancer/i.test(m.name);
const isGroundTruthTumor = (m: TwinMesh) => isTumorMesh(m) && m.layer !== 'scenario';
const isScenarioMesh = (m: TwinMesh) => m.layer === 'scenario';

// ── OrgansGroup ───────────────────────────────────────────────────────────────
function OrgansGroup({
  organs, tumorCentroid, predictedScale, biomarkerTumorCm, prob = 0,
}: {
  organs: TwinMesh[];
  tumorCentroid: THREE.Vector3;
  predictedScale: number;
  biomarkerTumorCm?: number;
  prob?: number;
}) {
  const geometries = useMemo(() => buildGeometries(organs), [organs]);
  const tumorIndex = useMemo(() => organs.findIndex(isGroundTruthTumor), [organs]);

  return (
    <group>
      {organs.map((m, i) =>
        m.kind === 'lines' ? (
          <lineSegments key={m.name} geometry={geometries[i]} frustumCulled={false}>
            <lineBasicMaterial color={m.color} transparent opacity={m.opacity ?? 0.12} depthWrite={false} />
          </lineSegments>
        ) : isScenarioMesh(m) ? (
          <mesh key={m.name} geometry={geometries[i]} frustumCulled={false}>
            <meshPhysicalMaterial
              color={m.color} emissive={m.color} emissiveIntensity={0.35}
              transparent opacity={m.opacity ?? 0.45} roughness={0.3} metalness={0.1}
              side={THREE.DoubleSide} depthWrite={false} depthTest wireframe={false}
            />
          </mesh>
        ) : isGroundTruthTumor(m) ? (
          // Ground-truth tumor — IMMUTABLE at 1:1 scale. NEVER scaled or modified.
          // The Anatomy-Aware Growth Shell renders ON TOP of this layer.
          <mesh key={m.name} geometry={geometries[i]} frustumCulled={false} renderOrder={2}>
            <meshPhysicalMaterial
              color="#ff3b3b" emissive="#ff3b3b" emissiveIntensity={0.25}
              transparent opacity={0.55} roughness={0.3} metalness={0.1}
              side={THREE.DoubleSide} depthWrite={false} depthTest
            />
          </mesh>
        ) : (
          // Pancreas — solid pale sandal #e0a468. 100% permanent. NEVER modified.
          <mesh key={m.name} geometry={geometries[i]} frustumCulled={false} renderOrder={1}>
            <meshPhysicalMaterial
              color="#e0a468" emissive="#e0a468" emissiveIntensity={0.2}
              transparent={false} opacity={1.0} roughness={0.45} metalness={0.05}
              side={THREE.DoubleSide} depthWrite depthTest
            />
          </mesh>
        )
      )}


      {/* Anatomy-Aware Growth Shell (arXiv:2607.19210):
          A SEPARATE overlay that displaces vertices along surface normals.
          The ground-truth tumor is NEVER touched. */}
      {tumorIndex >= 0 && biomarkerTumorCm !== undefined && predictedScale !== 1 && (
        <AnatomyAwareTumorShell
          sourceGeometry={geometries[tumorIndex]}
          scaleFactor={predictedScale}
          prob={prob}
          centroid={tumorCentroid}
        />
      )}
    </group>
  );
}


function FitCamera({ organs }: { organs: TwinMesh[] }) {
  const { camera, controls } = useThree();
  const fittedRef = useRef<TwinMesh[] | null>(null);

  useEffect(() => {
    if (!organs.length) return;
    if (fittedRef.current === organs) return;
    fittedRef.current = organs;

    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (const m of organs) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3));
      geo.computeBoundingBox();
      const bb = geo.boundingBox;
      if (bb) { min.min(bb.min); max.max(bb.max); }
      geo.dispose();
    }
    const centre = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
    const size = max.distanceTo(min);
    const dist = Math.max(size * 1.4, 3);
    camera.position.set(centre.x + dist * 0.7, centre.y + dist * 0.45, centre.z + dist * 0.9);
    camera.lookAt(centre);
    camera.near = 0.01;
    camera.far = Math.max(dist * 4, 10000);
    camera.updateProjectionMatrix();

    if (controls && (controls as any).target) {
      (controls as any).target.copy(centre);
      (controls as any).update?.();
    }
  }, [organs, camera, controls]);

  return null;
}

// ── Main viewer ───────────────────────────────────────────────────────────────
export function TwinMeshViewer({ url, directPayload, biomarkerTumorCm, prob = 0 }: TwinMeshViewerProps) {
  const [payload, setPayload] = useState<TwinPayload | null>(directPayload || null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!directPayload);

  useEffect(() => {
    if (directPayload) { setPayload(directPayload); setLoading(false); setError(null); return; }
    if (!url) return;
    let cancelled = false;
    setLoading(true); setError(null); setPayload(null);
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: TwinPayload) => { if (!cancelled) setPayload(d); })
      .catch((e: Error) => { if (!cancelled) setError(e.message || String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [url, directPayload]);

  const tumorOrgans = (payload?.organs) || [];
  const tumorGeo = useMemo(() => tumorOrgans.find((o) => isGroundTruthTumor(o)), [tumorOrgans]);
  const gtDiameterCm = useMemo(() => (tumorGeo ? tumorDiameterCm(tumorGeo.positions) : 0), [tumorGeo]);
  const metaCentroid = useMemo(() => {
    const arr = payload?.meta?.tumor_centroid as number[] | undefined;
    return Array.isArray(arr) && arr.length >= 3
      ? new THREE.Vector3(arr[0], arr[1], arr[2]) : null;
  }, [payload]);
  const tumorCentroid = useMemo(() => (
    metaCentroid ?? (tumorGeo ? meshCentroid(tumorGeo.positions) : new THREE.Vector3())
  ), [metaCentroid, tumorGeo]);
  const predictedScale = useMemo(() => {
    if (!biomarkerTumorCm || !gtDiameterCm || gtDiameterCm <= 0) return 1;
    return Math.max(0.2, Math.min(4.5, biomarkerTumorCm / gtDiameterCm));
  }, [biomarkerTumorCm, gtDiameterCm]);

  if (loading) return <ViewerFallback text="Loading real 3D twin…" />;
  if (error || !payload) return <ViewerFallback text={`Twin mesh unavailable — ${error || 'no data'}`} />;

  const organs = payload.organs || [];
  const meta = payload.meta || {};
  const caseLabel = meta.case ? `case ${meta.case} · ` : '';
  const volumeLabel = meta.tumor_volume_cm3 !== undefined ? `${Number(meta.tumor_volume_cm3).toFixed(1)} cm³ tumour` : '';
  const slicesLabel = meta.slices ? ` · ${meta.slices} slices` : '';
  const caption = `${caseLabel}${volumeLabel}${slicesLabel}`;

  return (
    <WebGLBoundary fallback={<ViewerFallback text="3D rendering failed — showing fallback" />}>
      <div style={{
        width: '100%', height: '100%', minHeight: 420, borderRadius: 8,
        overflow: 'hidden', background: '#09111a', position: 'relative',
      }}>
        <Canvas camera={{ position: [3, 1.5, 4], fov: 40 }} gl={{ antialias: true, alpha: false }}>
          <color attach="background" args={['#09111a']} />
          <ambientLight intensity={0.65} />
          <directionalLight position={[5, 5, 5]} intensity={1.2} color="#d8ecff" />
          <pointLight position={[-3, 2, 3]} intensity={0.8} color="#62dec0" />
          <Suspense fallback={<Html center><span style={{ color: '#718e9a', fontSize: 13 }}>Rendering…</span></Html>}>
            {organs.length > 0 && (
              <>
                <OrgansGroup organs={organs} tumorCentroid={tumorCentroid} predictedScale={predictedScale} biomarkerTumorCm={biomarkerTumorCm} prob={prob} />
                <FitCamera organs={organs} />
              </>
            )}
          </Suspense>
          <OrbitControls makeDefault enablePan enableZoom dampingFactor={0.08} />
        </Canvas>

        {/* Title + metadata */}
        <div style={{ position: 'absolute', top: 8, left: 8, right: 8, pointerEvents: 'none' }}>
          <div style={{
            color: '#b7f6e4', fontSize: 13, fontWeight: 600,
            background: 'rgba(0,0,0,0.6)', padding: '4px 10px', borderRadius: 6,
            display: 'inline-block',
          }}>
            {payload.title || (payload.mode === 'real-ct' ? 'Real CT Twin' : 'Realistic Anatomy')}
          </div>
          {caption && (
            <div style={{
              marginTop: 4, color: '#718e9a', fontSize: 11,
              background: 'rgba(0,0,0,0.6)', padding: '2px 10px', borderRadius: 6,
              display: 'inline-block',
            }}>
              {caption}
            </div>
          )}
          {tumorGeo && (
            <div style={{
              marginTop: 4, color: '#ef565f', fontSize: 11, fontWeight: 600,
              background: 'rgba(0,0,0,0.6)', padding: '2px 10px', borderRadius: 6,
              display: 'inline-block', whiteSpace: 'nowrap',
            }}>
              {biomarkerTumorCm
                ? `Tumour ${biomarkerTumorCm.toFixed(1)} cm (biomarker) · GT ${gtDiameterCm.toFixed(1)} cm`
                : `Tumour ${gtDiameterCm.toFixed(1)} cm (Patient Segmentation)`}
            </div>
          )}
          {predictedScale !== 1 && biomarkerTumorCm !== undefined && (
            <div style={{
              marginTop: 4, color: '#ff9933', fontSize: 11, fontWeight: 600,
              background: 'rgba(0,0,0,0.6)', padding: '2px 10px', borderRadius: 6,
              display: 'inline-block', whiteSpace: 'nowrap',
            }}>
              🧬 Anatomy-Aware Growth Shell active · ×{predictedScale.toFixed(2)} displacement
            </div>
          )}
          {(payload?.organs || []).some(o => o.layer === 'scenario') && (
            <div style={{
              marginTop: 4, color: '#ff9933', fontSize: 11, fontWeight: 600,
              background: 'rgba(0,0,0,0.6)', padding: '2px 10px', borderRadius: 6,
              display: 'inline-block', whiteSpace: 'nowrap',
            }}>
              ⚠ Simulated/Projected tumor layer active
            </div>
          )}
        </div>

        {/* Organ legend */}
        {organs.filter((m) => m.legend !== false).length > 0 && (
          <div style={{
            position: 'absolute', bottom: 8, right: 8, maxWidth: 240,
            background: 'rgba(0,0,0,0.62)', borderRadius: 8, padding: '8px 10px',
            display: 'flex', flexDirection: 'column', gap: 3, pointerEvents: 'none',
          }}>
            {organs.filter((m) => m.legend !== false).map((m) => (
              <div key={m.name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: m.color }} />
                <span style={{ fontSize: 10, color: '#b0c8d8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {m.name}
                </span>
              </div>
            ))}
            {predictedScale !== 1 && biomarkerTumorCm !== undefined && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                  background: 'linear-gradient(135deg, #ff2e0a, #ff8c14)',
                  border: '1px solid #ff5500',
                }} />
                <span style={{ fontSize: 10, color: '#ff9933', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  Growth Shell (Anatomy-Aware)
                </span>
              </div>
            )}
          </div>
        )}

        <div style={{ position: 'absolute', bottom: 8, left: 8, color: '#4a6a76', fontSize: 11 }}>
          Scroll to zoom · Drag to rotate
        </div>
      </div>
    </WebGLBoundary>
  );
}
