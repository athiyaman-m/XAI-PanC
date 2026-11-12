'use client';

import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { TumorConfig } from '../../engine/imaging/syntheticModelEngine';
import type { BiomarkerForm } from './ClinicalTwinWorkspace';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface CtTumorInfo {
  source: 'segmentation' | 'msd' | 'default';
  label?: string | null;
  cm3?: number | null;
  /** Downsampled-grid voxel [z, y, x] anchoring the tumour. */
  voxel?: [number, number, number] | null;
}

export interface CtVolumePayload {
  mode: string;
  title?: string;
  width: number;
  height: number;
  depth: number;
  /** Voxel spacing in mm, same axis order as the payload array (z, y, x). */
  spacing_mm: [number, number, number];
  /** HU window used to build data_b64. */
  window: [number, number];
  /** zlib raw-deflate + base64 of uint8 [depth, height, width] HU-windowed. */
  data_b64: string;
  tumor?: CtTumorInfo | null;
  meta?: Record<string, unknown>;
}

interface CtVolumeViewerProps {
  payload: CtVolumePayload;
  config: TumorConfig;
  prob: number;
  biomarkers: BiomarkerForm;
  month?: number;
}

interface BoxInfo {
  w: number; h: number; d: number;
  dx: number; dy: number; dz: number;
  size: [number, number, number]; // scene units
  scale: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function detectWebGL2(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2'));
  } catch {
    return false;
  }
}

async function inflateDeflate(b64: string): Promise<Uint8Array> {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

function volumeCenterOfMass(raw: Uint8Array, w: number, h: number, d: number): [number, number, number] | null {
  let zs = 0, ys = 0, xs = 0, n = 0;
  for (let z = 0; z < d; z++) {
    for (let y = 0; y < h; y++) {
      const row = (z * h + y) * w;
      for (let x = 0; x < w; x++) {
        if (raw[row + x] > 110) { zs += z; ys += y; xs += x; n++; }
      }
    }
  }
  if (!n) return null;
  return [zs / n, ys / n, xs / n];
}

class WebGLBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() { return this.state.hasError ? this.props.fallback : this.props.children; }
}

// ── Raymarching volume ────────────────────────────────────────────────────────
function VolumeRaymarch({ tex, box }: { tex: THREE.Data3DTexture; box: BoxInfo }) {
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        uVolume: { value: tex },
        uBoxSize: { value: new THREE.Vector3(...box.size) },
        uCameraPos: { value: new THREE.Vector3() },
        uDensity: { value: 7.0 },
      },
      vertexShader: /* glsl */ `
        out vec3 vWorldPos;
        out vec3 vLocalPos;
        void main() {
          vec4 world = modelMatrix * vec4(position, 1.0);
          vWorldPos = world.xyz;
          vLocalPos = position + 0.5;
          gl_Position = projectionMatrix * viewMatrix * world;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        precision highp sampler3D;
        uniform sampler3D uVolume;
        uniform vec3 uBoxSize;
        uniform vec3 uCameraPos;
        uniform float uDensity;
        in vec3 vWorldPos;
        in vec3 vLocalPos;
        out vec4 pc_fragColor;
        void main() {
          vec3 dir = normalize(vWorldPos - uCameraPos);
          vec3 localDir = dir / uBoxSize;
          vec3 invDir = 1.0 / max(abs(localDir), vec3(1e-6)) * sign(localDir);
          vec3 t0 = -vLocalPos * invDir;
          vec3 t1 = (vec3(1.0) - vLocalPos) * invDir;
          vec3 tNearV = min(t0, t1);
          vec3 tFarV = max(t0, t1);
          float tNear = max(max(tNearV.x, tNearV.y), tNearV.z);
          float tFar = min(min(tFarV.x, tFarV.y), tFarV.z);
          if (tFar < tNear) { pc_fragColor = vec4(0.0); return; }
          float t = max(tNear, 0.0);
          float stepLen = max((tFar - tNear) / 200.0, 0.0004);
          vec4 acc = vec4(0.0);
          for (int i = 0; i < 220; i++) {
            if (acc.a > 0.96) break;
            if (t >= tFar) break;
            vec3 pos = vLocalPos + localDir * t;
            float v = texture(uVolume, pos).r;
            if (v > 0.004) {
              float aStep = 1.0 - exp(-v * uDensity * stepLen);
              vec3 col = vec3(0.38 + 0.62 * v, 0.38 + 0.62 * v, 0.36 + 0.64 * v);
              acc.rgb += (1.0 - acc.a) * col * aStep;
              acc.a += (1.0 - acc.a) * aStep;
            }
            t += stepLen;
          }
          pc_fragColor = acc;
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
    });
  }, [tex, box]);

  const { camera } = useThree();
  useFrame(() => {
    material.uniforms.uCameraPos.value.copy(camera.position);
  });

  useEffect(() => () => material.dispose(), [material]);

  return (
    <mesh material={material} position={[0, 0, 0]}>
      <boxGeometry args={box.size} />
    </mesh>
  );
}

// ── Tumour + live impact halo ─────────────────────────────────────────────────
function TumorAndHalo({
  position, radius, haloRadius, risk, config,
}: {
  position: [number, number, number];
  radius: number;
  haloRadius: number;
  risk: number;
  config: TumorConfig;
}) {
  const tumorRef = useRef<THREE.Mesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const pulse = 1 + Math.sin(t * 3.4) * 0.07;
    if (tumorRef.current) tumorRef.current.scale.setScalar(pulse);
    if (haloRef.current) {
      haloRef.current.scale.setScalar(1 + Math.sin(t * 2.0) * 0.05);
      haloRef.current.rotation.y += 0.0025;
    }
    if (wireRef.current) wireRef.current.scale.setScalar(1 + Math.sin(t * 1.5 + 1.2) * 0.12);
  });

  return (
    <group position={position}>
      {/* Tumour core (opaque — drawn before the transparent volume) */}
      <mesh ref={tumorRef}>
        <icosahedronGeometry args={[radius, 2]} />
        <meshPhysicalMaterial
          color="#d92525"
          emissive="#ff2222"
          emissiveIntensity={0.55 + risk * 0.45}
          roughness={0.3}
          metalness={0.1}
          clearcoat={0.4}
          depthWrite
        />
      </mesh>
      {/* Tumour highlight shell */}
      <mesh scale={1.12}>
        <icosahedronGeometry args={[radius, 1]} />
        <meshBasicMaterial color="#ff5c5c" wireframe transparent opacity={0.35} depthWrite={false} />
      </mesh>
      {/* Live biomarker-driven impact region */}
      <mesh ref={haloRef}>
        <sphereGeometry args={[haloRadius, 48, 36]} />
        <meshBasicMaterial
          color={risk > 0.6 ? '#ff3b2f' : '#ff7a3c'}
          transparent
          opacity={0.10 + risk * 0.10}
          depthWrite={false}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <lineSegments ref={wireRef}>
        <wireframeGeometry args={[new THREE.SphereGeometry(haloRadius, 24, 18)]} />
        <lineBasicMaterial
          color={risk > 0.6 ? '#ff5c5c' : '#ff9d5c'}
          transparent
          opacity={0.42 + risk * 0.3}
          depthWrite={false}
        />
      </lineSegments>
      <Html center position={[0, radius * 1.5, 0]} distanceFactor={10}>
        <div
          style={{
            padding: '5px 9px',
            borderRadius: 999,
            border: '1px solid rgba(255, 83, 83, 0.45)',
            background: 'rgba(39, 7, 7, 0.84)',
            color: '#ffc9c9',
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.04em',
            whiteSpace: 'nowrap',
          }}
        >
          Tumor {config.sizeCm} cm
        </div>
      </Html>
    </group>
  );
}

// ── Fit camera to the volume box ──────────────────────────────────────────────
function FitCamera({ box }: { box: BoxInfo }) {
  const { camera } = useThree();
  useEffect(() => {
    const size = new THREE.Vector3(...box.size);
    const dist = Math.max(size.length() * 1.35, 3);
    camera.position.set(dist * 0.55, dist * 0.4, dist * 0.78);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [box, camera]);
  return null;
}

function SceneLights() {
  return (
    <>
      <ambientLight intensity={1.3} color="#aac0d0" />
      <directionalLight position={[3, 4, 5]} intensity={1.4} color="#fff7ef" />
      <pointLight position={[-3, 1, 2]} intensity={22} color="#5fc6ff" distance={14} />
    </>
  );
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function CtScene({
  payload, raw, box, tumorPos, tumorRadius, haloRadius, risk, config,
}: {
  payload: CtVolumePayload;
  raw: Uint8Array;
  box: BoxInfo;
  tumorPos: [number, number, number];
  tumorRadius: number;
  haloRadius: number;
  risk: number;
  config: TumorConfig;
}) {
  const texture = useMemo(() => {
    const tex = new THREE.Data3DTexture(raw, payload.width, payload.height, payload.depth);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [raw, payload.width, payload.height, payload.depth]);

  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <Canvas
      dpr={[1, 1.4]}
      camera={{ position: [3.5, 2.5, 5], fov: 38 }}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      style={{ width: '100%', height: '100%' }}
    >
      <color attach="background" args={['#071018']} />
      <SceneLights />
      <VolumeRaymarch tex={texture} box={box} />
      <TumorAndHalo
        position={tumorPos}
        radius={tumorRadius}
        haloRadius={haloRadius}
        risk={risk}
        config={config}
      />
      <FitCamera box={box} />
      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.6}
        minDistance={1.6}
        maxDistance={box.size.reduce((a, b) => a + b, 0) * 3}
      />
    </Canvas>
  );
}

// ── Viewer ────────────────────────────────────────────────────────────────────
export function CtVolumeViewer({ payload, config, prob, biomarkers, month = 0 }: CtVolumeViewerProps) {
  const [raw, setRaw] = useState<Uint8Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [webgl2, setWebgl2] = useState<boolean | null>(null);

  useEffect(() => {
    setWebgl2(detectWebGL2());
  }, []);

  useEffect(() => {
    let cancelled = false;
    setRaw(null);
    setError(null);
    inflateDeflate(payload.data_b64)
      .then((data) => {
        if (!cancelled) setRaw(data);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => { cancelled = true; };
  }, [payload.data_b64]);

  const box = useMemo<BoxInfo>(() => {
    const [dz, dy, dx] = payload.spacing_mm;
    const w = payload.width, h = payload.height, d = payload.depth;
    const xmm = w * dx, ymm = h * dy, zmm = d * dz;
    const scale = 4.6 / Math.max(xmm, ymm, zmm);
    return {
      w, h, d, dx, dy, dz,
      size: [xmm * scale, zmm * scale, ymm * scale],
      scale,
    };
  }, [payload]);

  const bodyCentroid = useMemo(() => {
    if (!raw) return null;
    return volumeCenterOfMass(raw, payload.width, payload.height, payload.depth);
  }, [raw, payload.width, payload.height, payload.depth]);

  const { tumorPos, tumorRadius, haloRadius } = useMemo(() => {
    const toPos = (v: [number, number, number]): [number, number, number] => {
      const [vz, vy, vx] = v;
      return [
        (vx - (payload.width - 1) / 2) * payload.spacing_mm[2] * box.scale,
        (vz - (payload.depth - 1) / 2) * payload.spacing_mm[0] * box.scale,
        (vy - (payload.height - 1) / 2) * payload.spacing_mm[1] * box.scale,
      ];
    };
    const anchor: [number, number, number] = payload.tumor?.voxel
      ? toPos(payload.tumor.voxel)
      : bodyCentroid
        ? toPos(bodyCentroid)
        : [0, 0, 0];
    const radius = Math.max(0.06, Math.min(1.6, config.sizeCm * 10 * box.scale));
    const impact = Math.min(
      5.5,
      1.6 + prob * 2.6 + Math.min(biomarkers.ca199, 1200) / 1200 * 0.7 +
        (biomarkers.smad4 ? 0.5 : 0) + month * 0.12,
    );
    return { tumorPos: anchor, tumorRadius: radius, haloRadius: radius * impact };
  }, [payload, box, config.sizeCm, prob, biomarkers, bodyCentroid, month]);

  const fallback = (
    <div style={{
      width: '100%', height: '100%', minHeight: 420, display: 'flex',
      flexDirection: 'column', gap: 10, alignItems: 'center', justifyContent: 'center',
      background: '#071018', color: '#718e9a', fontSize: 13, borderRadius: 8, padding: 24, textAlign: 'center',
    }}>
      {error ? `CT volume failed to load — ${error}` : 'Loading patient CT volume…'}
    </div>
  );

  if (webgl2 === false) {
    return (
      <div style={{
        width: '100%', height: '100%', minHeight: 420, display: 'grid', placeItems: 'center',
        background: '#071018', color: '#8293a7', fontSize: 13, borderRadius: 8, padding: 24, textAlign: 'center',
      }}>
        CT volume rendering requires WebGL2. Tumour and impact tracking remain available from the biomarkers panel.
      </div>
    );
  }
  if (webgl2 === null || !raw) return fallback;

  const sourceLabel = (payload.meta?.source as string) || payload.title || 'CT';
  const cm3 = payload.tumor?.cm3;

  return (
    <WebGLBoundary fallback={fallback}>
      <div style={{
        width: '100%', height: '100%', minHeight: 420, borderRadius: 8,
        overflow: 'hidden', background: '#071018', position: 'relative',
      }}>
        <CtScene
          payload={payload}
          raw={raw}
          box={box}
          tumorPos={tumorPos}
          tumorRadius={tumorRadius}
          haloRadius={haloRadius}
          risk={prob}
          config={config}
        />

        {/* Caption */}
        <div style={{ position: 'absolute', top: 8, left: 8, pointerEvents: 'none' }}>
          <div style={{
            color: '#b7f6e4', fontSize: 12, fontWeight: 600,
            background: 'rgba(0,0,0,0.6)', padding: '4px 10px', borderRadius: 6,
            display: 'inline-block',
          }}>
            Patient CT volume — {sourceLabel}
          </div>
          <div style={{
            marginTop: 4, color: '#718e9a', fontSize: 11,
            background: 'rgba(0,0,0,0.6)', padding: '2px 10px', borderRadius: 6,
            display: 'inline-block',
          }}>
            {payload.width}×{payload.height}×{payload.depth} · {payload.spacing_mm.map((s) => s.toFixed(2)).join('/')} mm
            {payload.tumor?.source !== 'default' ? ` · ${payload.tumor?.label || 'lesion anchor'}` : ''}
            {cm3 != null ? ` · ${cm3.toFixed(1)} cm³` : ''}
          </div>
        </div>

        {/* Live impact readout */}
        <div style={{ position: 'absolute', bottom: 8, right: 8, pointerEvents: 'none' }}>
          <div style={{
            background: 'rgba(0,0,0,0.62)', borderRadius: 8, padding: '7px 10px', fontSize: 11,
            color: '#c8d9e5', display: 'flex', flexDirection: 'column', gap: 3,
          }}>
            <span style={{ color: prob > 0.6 ? '#ff7a6e' : '#ffb59c' }}>
              Impact region Ø {(haloRadius * 2 / (10 * box.scale)).toFixed(1)} cm
            </span>
            <span style={{ color: '#ff8f7e' }}>Tumour {config.sizeCm} cm (live)</span>
          </div>
        </div>

        <div style={{ position: 'absolute', bottom: 8, left: 8, color: '#4a6a76', fontSize: 11 }}>
          Scroll to zoom · Drag to rotate
        </div>
      </div>
    </WebGLBoundary>
  );
}
