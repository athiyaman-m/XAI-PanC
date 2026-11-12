import * as THREE from 'three';

export type OrganName =
  | 'Pancreas' | 'Liver' | 'Gallbladder' | 'Stomach' | 'Duodenum' | 'Spleen'
  | 'Portal Vein' | 'Superior Mesenteric Vein' | 'Superior Mesenteric Artery'
  | 'Common Bile Duct' | 'Aorta' | 'Inferior Vena Cava' | 'Regional Lymph Nodes'
  | 'Kidneys' | 'Lungs' | 'Heart';

export interface OrganTransform {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
}

export const ORGAN_TRANSFORMS: Record<string, OrganTransform> = {
  Pancreas:              { position: [0, -0.08, 0.05],  rotation: [0, 0, 0],       scale: [1, 0.35, 0.45] },
  Liver:                 { position: [0.55, 0.28, 0.02], rotation: [0.1, -0.15, 0], scale: [1.1, 0.55, 0.4] },
  Gallbladder:           { position: [0.52, 0.05, 0.22], rotation: [0, 0, 0.3],     scale: [0.18, 0.28, 0.18] },
  Stomach:               { position: [-0.15, -0.02, -0.25], rotation: [0, 0, 0],   scale: [0.65, 0.4, 0.35] },
  Duodenum:              { position: [-0.15, -0.1, 0.08], rotation: [0, 0, 0],     scale: [0.35, 0.45, 0.3] },
  Spleen:                { position: [-0.85, 0.1, 0.02], rotation: [0, 0, 0.2],    scale: [0.4, 0.5, 0.22] },
  'Portal Vein':         { position: [0.1, 0.05, 0.1],   rotation: [0, 0, 0],      scale: [0.6, 0.4, 0.3] },
  'Superior Mesenteric Vein': { position: [0.02, -0.25, 0.05], rotation: [0, 0, 0], scale: [0.15, 0.35, 0.15] },
  'Superior Mesenteric Artery': { position: [-0.03, -0.25, 0.03], rotation: [0, 0, 0], scale: [0.12, 0.3, 0.12] },
  'Common Bile Duct':    { position: [0.35, 0.0, 0.2],   rotation: [0.2, 0, 0],    scale: [0.08, 0.4, 0.08] },
  Aorta:                 { position: [-0.05, -0.05, -0.25], rotation: [0, 0, 0],   scale: [0.2, 0.9, 0.2] },
  'Inferior Vena Cava':  { position: [0.35, -0.08, -0.18], rotation: [0, 0, 0],    scale: [0.18, 0.85, 0.18] },
  'Regional Lymph Nodes': { position: [0.02, 0.1, 0.25], rotation: [0, 0, 0],     scale: [0.25, 0.2, 0.2] },
  Kidneys:               { position: [-0.6, -0.35, -0.12], rotation: [0, 0.2, 0],  scale: [0.3, 0.45, 0.22] },
  Lungs:                 { position: [0.15, 0.95, -0.08], rotation: [0, 0, 0],     scale: [1.1, 0.7, 0.5] },
  Heart:                 { position: [0.25, 0.55, -0.35], rotation: [0.2, 0, 0],   scale: [0.35, 0.4, 0.3] },
};

export interface OrganMaterial {
  color: string;
  roughness: number;
  metalness: number;
  emissive?: string;
  emissiveIntensity?: number;
  clearcoat?: number;
  subsurface?: number;
}

export const ORGAN_MATERIALS: Record<string, OrganMaterial> = {
  Pancreas:              { color: '#d4a373', roughness: 0.7, metalness: 0.05, subsurface: 0.3 },
  Liver:                 { color: '#7c3a3a', roughness: 0.65, metalness: 0.05, subsurface: 0.2 },
  Gallbladder:           { color: '#5b8c5a', roughness: 0.5, metalness: 0.1, clearcoat: 0.3 },
  Stomach:               { color: '#c48a7c', roughness: 0.7, metalness: 0.05, subsurface: 0.3 },
  Duodenum:              { color: '#b8957a', roughness: 0.65, metalness: 0.05, subsurface: 0.2 },
  Spleen:                { color: '#6b3fa0', roughness: 0.6, metalness: 0.05, subsurface: 0.2 },
  'Portal Vein':         { color: '#3b6ea5', roughness: 0.4, metalness: 0.15, clearcoat: 0.2 },
  'Superior Mesenteric Vein': { color: '#4a7fb5', roughness: 0.4, metalness: 0.1, clearcoat: 0.2 },
  'Superior Mesenteric Artery': { color: '#b53a3a', roughness: 0.4, metalness: 0.15, clearcoat: 0.2 },
  'Common Bile Duct':    { color: '#8ab87a', roughness: 0.5, metalness: 0.1, clearcoat: 0.2 },
  Aorta:                 { color: '#c23535', roughness: 0.4, metalness: 0.15, clearcoat: 0.2 },
  'Inferior Vena Cava':  { color: '#4a6fa5', roughness: 0.4, metalness: 0.1, clearcoat: 0.2 },
  'Regional Lymph Nodes': { color: '#c9a87c', roughness: 0.6, metalness: 0.05, subsurface: 0.2 },
  Kidneys:               { color: '#8b5e3c', roughness: 0.65, metalness: 0.05, subsurface: 0.15 },
  Lungs:                 { color: '#c4958a', roughness: 0.7, metalness: 0.05, subsurface: 0.3 },
  Heart:                 { color: '#8b3a3a', roughness: 0.55, metalness: 0.05, subsurface: 0.25 },
};

export interface TumorConfig {
  tumorLocation: 'head' | 'neck' | 'body' | 'tail';
  tumorLengthMm: number;
  tumorWidthMm: number;
  tumorHeightMm: number;
  tumorShape: 'spherical' | 'ellipsoid' | 'irregular';
  progression: number;
  invasionLevel: number;
  vascularInvolvement: boolean;

  // Legacy/computed fields
  sizeCm: number;
  position: [number, number, number];
  invasionDepth: number;
  necrosisPct: number;
  vascularInvasion: number;
  glowIntensity: number;
  spatialInfiltrationRadius: number;
}

export interface SyntheticOrganData {
  organ: string;
  geometry: THREE.BufferGeometry;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  material: OrganMaterial;
}

export interface GeneratedScene {
  organs: SyntheticOrganData[];
  tumor?: SyntheticOrganData & { config: TumorConfig };
}

function catmullRom(points: THREE.Vector3[], segments: number): THREE.Vector3[] {
  const result: THREE.Vector3[] = [];
  for (let i = 0; i < segments; i++) {
    const t = i / (segments - 1);
    const totalSegments = points.length - 1;
    const seg = Math.min(Math.floor(t * totalSegments), totalSegments - 1);
    const localT = (t * totalSegments - seg);
    const p0 = points[Math.max(0, seg - 1)];
    const p1 = points[seg];
    const p2 = points[Math.min(seg + 1, points.length - 1)];
    const p3 = points[Math.min(seg + 2, points.length - 1)];
    const t2 = localT * localT;
    const t3 = t2 * localT;
    const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * localT + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
    const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * localT + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
    const z = 0.5 * ((2 * p1.z) + (-p0.z + p2.z) * localT + (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t2 + (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t3);
    result.push(new THREE.Vector3(x, y, z));
  }
  return result;
}

function smoothNoise(x: number, y: number, z: number, freq: number = 3): number {
  const fx = x * freq, fy = y * freq, fz = z * freq;
  const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
  const dx = fx - ix, dy = fy - iy, dz = fz - iz;
  const sx = dx * dx * (3 - 2 * dx);
  const sy = dy * dy * (3 - 2 * dy);
  const sz = dz * dz * (3 - 2 * dz);
  const p = (n: number) => { const v = Math.sin(n * 12.9898 + n * 78.233) * 43758.5453; return v - Math.floor(v); };
  const v = (x: number, y: number, z: number) => p(x + p(y + p(z)));
  const n000 = v(ix, iy, iz), n100 = v(ix + 1, iy, iz);
  const n010 = v(ix, iy + 1, iz), n110 = v(ix + 1, iy + 1, iz);
  const n001 = v(ix, iy, iz + 1), n101 = v(ix + 1, iy, iz + 1);
  const n011 = v(ix, iy + 1, iz + 1), n111 = v(ix + 1, iy + 1, iz + 1);
  const nx0 = n000 + (n100 - n000) * sx, nx1 = n010 + (n110 - n010) * sx;
  const ny0 = nx0 + (nx1 - nx0) * sy;
  const nx2 = n001 + (n101 - n001) * sx, nx3 = n011 + (n111 - n011) * sx;
  const ny1 = nx2 + (nx3 - nx2) * sy;
  return ny0 + (ny1 - ny0) * sz;
}

function fbm(x: number, y: number, z: number, octaves: number = 3): number {
  let value = 0, amp = 1, freq = 1;
  for (let i = 0; i < octaves; i++) {
    value += amp * smoothNoise(x * freq, y * freq, z * freq);
    amp *= 0.5;
    freq *= 2;
  }
  return value;
}

function applyNoise(geo: THREE.BufferGeometry, intensity: number = 0.06, octaves: number = 3): void {
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const n = fbm(verts[i], verts[i + 1], verts[i + 2], octaves);
    const d = Math.sqrt(verts[i] ** 2 + verts[i + 1] ** 2 + verts[i + 2] ** 2);
    if (d > 0.01) {
      verts[i] += (verts[i] / d) * n * intensity;
      verts[i + 1] += (verts[i + 1] / d) * n * intensity;
      verts[i + 2] += (verts[i + 2] / d) * n * intensity;
    }
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

function createPancreas(): THREE.BufferGeometry {
  const segs = 40, rings = 20;
  const geo = new THREE.BufferGeometry();
  const verts: number[] = [], idxs: number[] = [], uvs: number[] = [];
  for (let j = 0; j <= rings; j++) {
    const v = j / rings;
    const angleY = v * Math.PI;
    for (let i = 0; i <= segs; i++) {
      const u = i / segs;
      const angleX = u * Math.PI * 2;
      const head = 1 + 0.35 * Math.exp(-((v - 0.15) ** 2) / 0.01);
      const body = 1 + 0.15 * Math.sin(v * Math.PI * 1.5);
      const tail = 0.65 + 0.3 * (1 - v);
      const r = tail + (head - tail) * (v < 0.3 ? 1 : Math.max(0, 1 - (v - 0.3) / 0.7));
      const xOff = v < 0.2 ? 0.3 * Math.sin(angleX * 0.5) : 0;
      const flat = 1 - 0.3 * Math.abs(Math.sin(angleX)) * (1 - v * 0.5);
      const rx = r * flat * Math.cos(angleX);
      const ry = r * flat * Math.sin(angleX) * 0.6;
      const zShift = 0.3 * (1 - v) ** 0.5;
      verts.push(
        -0.5 + v * 1.0 + rx * 0.3 + xOff * 0.2,
        ry * 0.25 - 0.05 + 0.08 * Math.sin(v * Math.PI * 2),
        zShift + rx * 0.2
      );
      uvs.push(u, v);
    }
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segs; i++) {
      const a = j * (segs + 1) + i;
      const b = a + segs + 1;
      idxs.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idxs);
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.computeVertexNormals();
  applyNoise(geo, 0.04, 2);
  return geo;
}

export function getPancreasAnatomicalPosition(location: 'head' | 'neck' | 'body' | 'tail'): [number, number, number] {
  let v = 0.5; // default body
  switch (location) {
    case 'head': v = 0.15; break;
    case 'neck': v = 0.3; break;
    case 'body': v = 0.5; break;
    case 'tail': v = 0.8; break;
  }
  const x = -0.5 + v * 1.0;
  const y = -0.05 + 0.08 * Math.sin(v * Math.PI * 2);
  const z = 0.3 * Math.pow(1 - v, 0.5);
  return [x, y, z];
}

function createLiver(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 48, 36);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const theta = Math.acos(y / d);
    const phi = Math.atan2(z, x);
    const rightLobe = 1 + 0.35 * Math.max(0, Math.cos(phi)) * (1 - theta / Math.PI * 0.5);
    const leftLobe = 1 + 0.2 * Math.max(0, -Math.cos(phi)) * (1 - theta / Math.PI * 0.3);
    const caudate = 1 + 0.15 * Math.max(0, Math.sin(phi)) * Math.sin(theta * 2);
    const r = rightLobe * leftLobe * caudate;
    const flat = 1 - 0.15 * Math.abs(Math.sin(theta)) * (1 - Math.abs(Math.cos(phi)));
    verts[i] = x * r * flat;
    verts[i + 1] = y * r * flat * 1.1;
    verts[i + 2] = z * r * flat * 0.85;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.05, 3);
  return geo;
}

function createGallbladder(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 24, 18);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const stretch = 1 + 0.4 * (y > 0 ? y : -y * 0.3);
    const taper = 1 - 0.2 * Math.max(0, -y);
    verts[i] = x * stretch * taper;
    verts[i + 1] = y * 1.3 * stretch;
    verts[i + 2] = z * stretch * taper;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.03, 2);
  return geo;
}

function createStomach(): THREE.BufferGeometry {
  const segs = 36, rings = 24;
  const geo = new THREE.BufferGeometry();
  const verts: number[] = [], idxs: number[] = [], uvs: number[] = [];
  for (let j = 0; j <= rings; j++) {
    const v = j / rings;
    for (let i = 0; i <= segs; i++) {
      const u = i / segs;
      const angle = u * Math.PI * 2;
      const r = 0.4 + 0.15 * Math.sin(v * Math.PI * 1.5);
      const bulge = 1 + 0.2 * Math.max(0, Math.sin(v * Math.PI * 2));
      const xOff = 0.3 * v;
      const yStretch = 1 + 0.25 * (1 - Math.abs(v - 0.5) * 2);
      verts.push(
        (r * bulge * Math.cos(angle) + xOff) * 0.8,
        (r * bulge * Math.sin(angle) * 0.6) * yStretch * 0.6 - 0.15,
        (r * bulge * Math.sin(angle) * 0.4 + 0.05 * Math.sin(angle * 2)) * 0.7
      );
      uvs.push(u, v);
    }
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segs; i++) {
      const a = j * (segs + 1) + i;
      const b = a + segs + 1;
      idxs.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idxs);
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.computeVertexNormals();
  applyNoise(geo, 0.04, 2);
  return geo;
}

function createDuodenum(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(-0.15, 0.25, 0.05),
    new THREE.Vector3(0.15, 0.2, 0.1),
    new THREE.Vector3(0.25, 0.05, 0.08),
    new THREE.Vector3(0.2, -0.1, 0.02),
    new THREE.Vector3(0.1, -0.2, -0.02),
    new THREE.Vector3(-0.05, -0.22, -0.04),
    new THREE.Vector3(-0.15, -0.15, -0.02),
  ];
  const curve = new THREE.CatmullRomCurve3(pts);
  const tube = new THREE.TubeGeometry(curve, 28, 0.15, 16, false);
  applyNoise(tube, 0.03, 2);
  return tube;
}

function createSpleen(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 28, 22);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const flatten = 1 - 0.35 * Math.abs(Math.sin(Math.acos(y / d)));
    const notch = 1 - 0.1 * Math.max(0, Math.sin(Math.atan2(z, x) * 3));
    verts[i] = x * 0.85 * flatten * notch;
    verts[i + 1] = y * 1.1 * flatten;
    verts[i + 2] = z * 0.55 * flatten * notch;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.04, 2);
  return geo;
}

function createVessel(points: THREE.Vector3[], radius: number, segments: number = 24): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(points);
  const tube = new THREE.TubeGeometry(curve, segments, radius, 8, false);
  return tube;
}

function createPortalVein(): THREE.BufferGeometry {
  const main = [
    new THREE.Vector3(0, 0.15, 0),
    new THREE.Vector3(0.02, 0.08, 0.02),
    new THREE.Vector3(0.05, -0.02, 0.03),
    new THREE.Vector3(0.06, -0.1, 0.02),
  ];
  const left = [
    new THREE.Vector3(0, 0.15, 0),
    new THREE.Vector3(-0.08, 0.18, -0.02),
    new THREE.Vector3(-0.15, 0.2, -0.04),
  ];
  const right = [
    new THREE.Vector3(0, 0.15, 0),
    new THREE.Vector3(0.12, 0.18, 0.04),
    new THREE.Vector3(0.2, 0.2, 0.06),
  ];
  const geo1 = createVessel(main, 0.08);
  const geo2 = createVessel(left, 0.05);
  const geo3 = createVessel(right, 0.05);
  const merged = mergeGeometries([geo1, geo2, geo3]);
  return merged;
}

function createSMV(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(0, 0.15, 0),
    new THREE.Vector3(0.01, 0.05, 0.01),
    new THREE.Vector3(0, -0.05, 0),
    new THREE.Vector3(-0.01, -0.15, -0.01),
    new THREE.Vector3(-0.02, -0.25, -0.02),
  ];
  return createVessel(pts, 0.07);
}

function createSMA(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(0, 0.12, -0.02),
    new THREE.Vector3(-0.01, 0.02, -0.01),
    new THREE.Vector3(-0.02, -0.08, 0),
    new THREE.Vector3(-0.03, -0.18, 0.01),
    new THREE.Vector3(-0.04, -0.28, 0.02),
  ];
  return createVessel(pts, 0.06);
}

function createBileDuct(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(0, 0.2, 0),
    new THREE.Vector3(0.02, 0.12, 0.02),
    new THREE.Vector3(0.03, 0.04, 0.03),
    new THREE.Vector3(0.02, -0.04, 0.02),
    new THREE.Vector3(0, -0.12, 0),
    new THREE.Vector3(-0.02, -0.18, -0.01),
  ];
  return createVessel(pts, 0.04);
}

function createAorta(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(0, 0.4, 0),
    new THREE.Vector3(0.01, 0.3, 0.01),
    new THREE.Vector3(0, 0.2, 0),
    new THREE.Vector3(-0.01, 0.1, -0.01),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.01, -0.1, 0.01),
    new THREE.Vector3(0, -0.2, 0),
    new THREE.Vector3(-0.01, -0.3, -0.01),
    new THREE.Vector3(0, -0.4, 0),
  ];
  const curve = new THREE.CatmullRomCurve3(pts);
  return new THREE.TubeGeometry(curve, 20, 0.1, 10, false);
}

function createIVC(): THREE.BufferGeometry {
  const pts = [
    new THREE.Vector3(0, 0.4, 0.02),
    new THREE.Vector3(0.01, 0.3, 0.03),
    new THREE.Vector3(0, 0.2, 0.02),
    new THREE.Vector3(-0.01, 0.1, 0.01),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.01, -0.1, -0.01),
    new THREE.Vector3(0, -0.2, -0.02),
    new THREE.Vector3(-0.01, -0.3, -0.03),
    new THREE.Vector3(0, -0.4, -0.02),
  ];
  const curve = new THREE.CatmullRomCurve3(pts);
  return new THREE.TubeGeometry(curve, 20, 0.09, 10, false);
}

function createLymphNodes(): THREE.BufferGeometry {
  const positions: [number, number, number][] = [
    [0, 0.15, 0.12], [-0.06, 0.12, 0.14], [0.05, 0.1, 0.15],
    [-0.03, 0.08, 0.18], [0.04, 0.06, 0.16], [-0.05, 0.04, 0.17],
    [0.02, 0.02, 0.2], [-0.02, 0.0, 0.19],
  ];
  const geos: THREE.BufferGeometry[] = [];
  for (const p of positions) {
    const sphere = new THREE.SphereGeometry(0.03 + Math.random() * 0.02, 8, 6);
    const pos = sphere.attributes.position;
    const verts = pos.array as Float32Array;
    for (let i = 0; i < verts.length; i += 3) {
      verts[i] += p[0];
      verts[i + 1] += p[1];
      verts[i + 2] += p[2];
    }
    pos.needsUpdate = true;
    geos.push(sphere);
  }
  return mergeGeometries(geos);
}

function createKidneys(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 28, 22);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const bean = 1 - 0.15 * Math.max(0, Math.cos(Math.atan2(z, x)));
    const hilum = 1 - 0.1 * Math.exp(-((x - 0.8) ** 2 + z ** 2) / 0.1);
    const flatten = 1 - 0.2 * Math.abs(Math.sin(Math.acos(y / d)));
    verts[i] = x * 0.7 * bean * hilum * flatten;
    verts[i + 1] = y * 1.2 * bean * flatten;
    verts[i + 2] = z * 0.5 * bean * hilum * flatten;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.03, 2);
  return geo;
}

function createLungs(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 36, 28);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const right = 1 + 0.15 * Math.max(0, Math.cos(Math.atan2(z, x)));
    const left = 1 + 0.1 * Math.max(0, -Math.cos(Math.atan2(z, x)));
    const apex = 1 + 0.15 * Math.max(0, y / d);
    const base = 1 - 0.1 * Math.max(0, -y / d);
    const cardiac = 1 - 0.08 * Math.max(0, Math.cos(Math.atan2(z, x))) * Math.max(0, -y / d);
    const r = right * left * apex * base * cardiac;
    verts[i] = x * r * 0.7;
    verts[i + 1] = y * r * 1.2;
    verts[i + 2] = z * r * 0.55;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.04, 3);
  return geo;
}

function createHeart(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 32, 24);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;
    const apex = 1 + 0.2 * Math.max(0, -y / d);
    const base = 1 - 0.1 * (y / d) * (y > 0 ? 1 : 0);
    const lv = 1 + 0.1 * Math.max(0, Math.cos(Math.atan2(z, x)));
    const rv = 1 + 0.1 * Math.max(0, -Math.cos(Math.atan2(z, x)));
    const r = apex * base * lv * rv;
    verts[i] = x * r * 0.75;
    verts[i + 1] = y * r * 0.9;
    verts[i + 2] = z * r * 0.55;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  applyNoise(geo, 0.03, 2);
  return geo;
}

export function createTumor(config: TumorConfig): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 28, 22);
  const pos = geo.attributes.position;
  const verts = pos.array as Float32Array;
  
  // 1 unit in Three.js roughly equals 150mm for our anatomy scaling
  const length = config.tumorLengthMm ? config.tumorLengthMm / 150 : config.sizeCm / 15;
  const width = config.tumorWidthMm ? config.tumorWidthMm / 150 : config.sizeCm / 15;
  const height = config.tumorHeightMm ? config.tumorHeightMm / 150 : config.sizeCm / 15;

  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i], y = verts[i + 1], z = verts[i + 2];
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d < 0.01) continue;

    let r = 1.0;
    if (config.tumorShape === 'irregular') {
      const spiculation = 1 + 0.15 * fbm(x, y, z, 3);
      r *= spiculation;
    }

    const necrosis = 1 - 0.15 * config.necrosisPct * Math.exp(-((x) ** 2 + y ** 2 + z ** 2) / 0.3);
    const invasion = 1 + 0.08 * config.invasionLevel * Math.max(0, Math.cos(Math.atan2(z, x) * 3));
    
    r *= necrosis * invasion;

    // Apply scale (radius is half the dimension)
    verts[i] = x * r * (length / 2);
    verts[i + 1] = y * r * (height / 2);
    verts[i + 2] = z * r * (width / 2);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

function mergeGeometries(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const merged = new THREE.BufferGeometry();
  const verts: number[] = [];
  const idxs: number[] = [];
  let offset = 0;
  for (const g of geos) {
    const pos = g.attributes.position;
    const arr = pos.array as Float32Array;
    for (let i = 0; i < arr.length; i += 3) {
      verts.push(arr[i], arr[i + 1], arr[i + 2]);
    }
    if (g.index) {
      const idx = g.index.array;
      for (let i = 0; i < idx.length; i++) {
        idxs.push(idx[i] + offset);
      }
    }
    offset += arr.length / 3;
  }
  merged.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  if (idxs.length) merged.setIndex(idxs);
  merged.computeVertexNormals();
  return merged;
}

const ORGAN_FACTORIES: Record<string, () => THREE.BufferGeometry> = {
  Pancreas: createPancreas,
  Liver: createLiver,
  Gallbladder: createGallbladder,
  Stomach: createStomach,
  Duodenum: createDuodenum,
  Spleen: createSpleen,
  'Portal Vein': createPortalVein,
  'Superior Mesenteric Vein': createSMV,
  'Superior Mesenteric Artery': createSMA,
  'Common Bile Duct': createBileDuct,
  Aorta: createAorta,
  'Inferior Vena Cava': createIVC,
  'Regional Lymph Nodes': createLymphNodes,
  Kidneys: createKidneys,
  Lungs: createLungs,
  Heart: createHeart,
};

export function generateScene(tumorConfig?: TumorConfig): GeneratedScene {
  const organs: SyntheticOrganData[] = [];
  for (const [name, factory] of Object.entries(ORGAN_FACTORIES)) {
    const transform = ORGAN_TRANSFORMS[name];
    const mat = ORGAN_MATERIALS[name];
    if (!transform || !mat) continue;
    const geometry = factory();
    organs.push({
      organ: name,
      geometry,
      position: transform.position,
      rotation: transform.rotation,
      scale: transform.scale,
      material: mat,
    });
  }

  let tumor: (SyntheticOrganData & { config: TumorConfig }) | undefined;
  if (tumorConfig) {
    const tumorGeo = createTumor(tumorConfig);
    tumor = {
      organ: 'Tumor',
      geometry: tumorGeo,
      position: tumorConfig.position,
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      material: {
        color: '#8b1a1a',
        roughness: 0.3,
        metalness: 0.1,
        emissive: '#ff2020',
        emissiveIntensity: tumorConfig.glowIntensity,
        clearcoat: 0.2,
      },
      config: tumorConfig,
    };
  }

  return { organs, tumor };
}

/**
 * Client-side biomarker → TumorConfig mapping (mirrors backend
 * ``biomarker_tumor_mapper.py``).  Runs in < 1ms so every slider
 * change produces an instant visual update without an API round-trip.
 */
export interface BiomarkerInputs {
  ca199?: number;
  cea?: number;
  bilirubin?: number;
  albumin?: number;
  weight_loss?: number;
  kras?: boolean;
  tp53?: boolean;
  smad4?: boolean;
  age?: number;
}

export interface BiomarkerTumorResult {
  tumorConfig: TumorConfig;
  organs: Record<string, number>;
  stageHint: string;
  riskScore: number;
}

function _clamp(v: number, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function _lerp(a: number, b: number, t: number) { return a + (b - a) * _clamp(t); }

export function createTumorFromBiomarkers(bio: BiomarkerInputs): BiomarkerTumorResult {
  const ca199 = bio.ca199 ?? 37;
  const cea = bio.cea ?? 3;
  const bilirubin = bio.bilirubin ?? 0.8;
  const weightLoss = bio.weight_loss ?? 0;
  const kras = bio.kras ?? false;
  const tp53 = bio.tp53 ?? false;
  const smad4 = bio.smad4 ?? false;
  const age = bio.age ?? 60;

  // Tumor size (cm) — primary driver is CA 19-9
  let sizeCm: number;
  if (ca199 <= 37) sizeCm = _lerp(0, 0.8, ca199 / 37);
  else if (ca199 <= 200) sizeCm = _lerp(0.8, 2.0, (ca199 - 37) / 163);
  else if (ca199 <= 1000) sizeCm = _lerp(2.0, 4.0, (ca199 - 200) / 800);
  else sizeCm = _lerp(4.0, 7.0, _clamp((ca199 - 1000) / 4000));

  sizeCm += _clamp(cea / 100) * 0.6;
  sizeCm += _clamp(weightLoss / 30) * 0.5;
  const mutCount = (kras ? 1 : 0) + (tp53 ? 1 : 0) + (smad4 ? 1 : 0);
  sizeCm += mutCount * 0.15;
  sizeCm = _clamp(sizeCm, 0, 8);

  // Invasion depth
  let invasion = 0;
  if (cea > 5) invasion += _clamp((cea - 5) / 50) * 0.35;
  if (kras) invasion += 0.15;
  if (tp53) invasion += 0.15;
  if (smad4) invasion += 0.2;
  invasion += _clamp(weightLoss / 30) * 0.15;
  invasion = _clamp(invasion);

  // Necrosis
  let necrosis = sizeCm > 3 ? _clamp((sizeCm - 3) / 5) * 0.6 : 0;
  if (tp53) necrosis += 0.1;
  necrosis = _clamp(necrosis);

  // Vascular invasion
  let vasc = 0;
  if (ca199 > 200) vasc += _clamp((ca199 - 200) / 2000) * 0.4;
  if (smad4) vasc += 0.25;
  if (invasion > 0.4) vasc += 0.15;
  vasc = _clamp(vasc);

  // Glow
  const glow = _clamp(_clamp(sizeCm / 6) * 0.8 + invasion * 0.2, 0.05, 1);

  // Position — bilirubin shifts toward pancreas head
  const biliT = _clamp(bilirubin / 8);
  const posX = _lerp(0, 0.15, biliT);
  const posY = _lerp(-0.08, -0.05, biliT);
  const posZ = _lerp(0.05, 0.15, biliT);

  // Organ burdens
  const pancBurden = _clamp(sizeCm / 6) * 70 + invasion * 20 + 10;
  const liverBurden = _clamp(bilirubin / 10) * 40 + _clamp(ca199 / 3000) * 30
    + (sizeCm > 3 && vasc > 0.3 ? 15 : 0);
  const bileBurden = _clamp(bilirubin / 6) * 65 + (bilirubin > 3 ? 20 : 0);
  const duoBurden = invasion * 50 + _clamp(sizeCm / 5) * 25;
  const spleenBurden = vasc * 30 + _clamp(sizeCm / 8) * 15;
  const lungBurden = _clamp(ca199 / 3000) * 25 + (smad4 ? 15 : 0) + (sizeCm > 4 ? 10 : 0);

  // Risk score
  const risk = Math.min(
    (_clamp(sizeCm / 6) * 35 + invasion * 20 + vasc * 15
    + _clamp(ca199 / 2000) * 15 + mutCount / 3 * 10
    + _clamp((age - 40) / 40) * 5) * 100,
    99,
  );

  // Stage hint
  let stageHint: string;
  if (sizeCm < 0.5) stageHint = 'Non-malignant';
  else if (sizeCm < 2 && invasion < 0.3) stageHint = 'Stage I';
  else if (sizeCm < 4 && vasc < 0.3) stageHint = 'Stage II';
  else if (vasc < 0.5 && lungBurden < 20) stageHint = 'Stage III';
  else stageHint = 'Stage IV';

  // Determine location based on position shift
  let loc: 'head' | 'neck' | 'body' | 'tail' = 'body';
  if (biliT > 0.6) loc = 'head';
  else if (biliT > 0.4) loc = 'neck';
  
  const anatomicalPos = getPancreasAnatomicalPosition(loc);

  return {
    tumorConfig: {
      tumorLocation: loc,
      tumorLengthMm: sizeCm * 10,
      tumorWidthMm: sizeCm * 10 * 0.8,
      tumorHeightMm: sizeCm * 10 * 0.9,
      tumorShape: 'irregular',
      progression: risk / 100,
      invasionLevel: Math.round(invasion * 1000) / 1000,
      vascularInvolvement: vasc > 0.3,
      sizeCm: Math.round(sizeCm * 100) / 100,
      position: anatomicalPos,
      invasionDepth: Math.round(invasion * 1000) / 1000,
      vascularInvasion: Math.round(vasc * 1000) / 1000,
      necrosisPct: Math.round(necrosis * 1000) / 1000,
      glowIntensity: Math.round(glow * 1000) / 1000,
      spatialInfiltrationRadius: Math.round(_clamp(sizeCm, 0.5, 8) * 100) / 100,
    },
    organs: {
      Pancreas: Math.round(Math.min(pancBurden, 98) * 10) / 10,
      Liver: Math.round(Math.min(liverBurden, 95) * 10) / 10,
      'Bile Duct': Math.round(Math.min(bileBurden, 92) * 10) / 10,
      Duodenum: Math.round(Math.min(duoBurden, 80) * 10) / 10,
      Spleen: Math.round(Math.min(spleenBurden, 70) * 10) / 10,
      Lungs: Math.round(Math.min(lungBurden, 60) * 10) / 10,
    },
    stageHint,
    riskScore: Math.round(risk * 10) / 10,
  };
}

export function disposeScene(scene: GeneratedScene): void {
  for (const o of scene.organs) {
    o.geometry.dispose();
  }
  if (scene.tumor) {
    scene.tumor.geometry.dispose();
  }
}
