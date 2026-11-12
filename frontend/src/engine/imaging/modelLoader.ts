import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { OrganHealth, OrganName } from '../../types';
import { generateScene, disposeScene } from './syntheticModelEngine';
import type { SyntheticOrganData } from './syntheticModelEngine';

const NIH_3D_BASE = 'https://code4fukui.github.io/human_organs/glb';

export const ORGAN_MODEL_URLS: Record<OrganName, string> = {
  Pancreas: `${NIH_3D_BASE}/3d-vh-m-pancreas.glb`,
  Liver: `${NIH_3D_BASE}/VH_M_Liver.glb`,
  'Bile Duct': `${NIH_3D_BASE}/VH_M_Liver.glb`,
  Duodenum: `${NIH_3D_BASE}/3d-vh-m-pancreas.glb`,
  Stomach: `${NIH_3D_BASE}/realistic_stomach.glb`,
  Spleen: `${NIH_3D_BASE}/VH_F_Spleen.glb`,
  Lungs: `${NIH_3D_BASE}/3d-vh-f-lung.glb`,
  'Lymph Nodes': `${NIH_3D_BASE}/VH_M_Liver.glb`,
  Heart: `${NIH_3D_BASE}/VH_M_Heart.glb`,
  Kidneys: `${NIH_3D_BASE}/VH_M_Kidney_L.glb`,
};

const modelCache = new Map<OrganName, SyntheticOrganData>();
const loader = new GLTFLoader();

export function loadOrganModel(organ: OrganName, health: OrganHealth): Promise<SyntheticOrganData> {
  if (modelCache.has(organ)) {
    return Promise.resolve(modelCache.get(organ)!);
  }

  const url = ORGAN_MODEL_URLS[organ];
  return new Promise((resolve) => {
    loader.load(
      url,
      (gltf) => {
        const group = gltf.scene;
        group.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        const geo = getOrganGeometryFromGroup(group);
        if (geo) {
          const mat = getOrganMaterial(organ, health);
          const data: SyntheticOrganData = {
            organ,
            geometry: geo,
            material: { color: mat.color.getHexString(), roughness: mat.roughness, metalness: mat.metalness },
            position: [0, 0, 0],
            rotation: [0, 0, 0],
            scale: [1, 1, 1],
          };
          modelCache.set(organ, data);
          resolve(data);
        } else {
          resolve(getProceduralFallback(organ, health));
        }
      },
      undefined,
      () => resolve(getProceduralFallback(organ, health))
    );
  });
}

function getProceduralFallback(organ: OrganName, health: OrganHealth): SyntheticOrganData {
  const scene = generateScene();
  const found = scene.organs.find((o) => o.organ === organ);
  const idx = scene.organs.indexOf(found ?? scene.organs[0]);
  const data = found ? { ...found, material: { ...found.material } } : {
    organ, geometry: new THREE.IcosahedronGeometry(1, 2),
    material: { color: health.color, roughness: 0.35, metalness: 0.05 },
    position: [0, 0, 0] as [number, number, number],
    rotation: [0, 0, 0] as [number, number, number],
    scale: [1, 1, 1] as [number, number, number],
  };
  for (let i = 0; i < scene.organs.length; i++) {
    if (i !== idx) scene.organs[i].geometry.dispose();
  }
  return data;
}

function getOrganGeometryFromGroup(group: THREE.Group): THREE.BufferGeometry | null {
  const geometries: THREE.BufferGeometry[] = [];
  group.traverse((child) => {
    if (child instanceof THREE.Mesh && child.geometry) geometries.push(child.geometry);
  });
  if (geometries.length === 0) return null;
  if (geometries.length === 1) return geometries[0];

  const merged = new THREE.BufferGeometry();
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  let vertexOffset = 0;
  for (const geo of geometries) {
    const p = geo.getAttribute('position'), n = geo.getAttribute('normal'), idx = geo.getIndex();
    if (p && n) {
      for (let i = 0; i < p.count; i++) positions.push(p.getX(i), p.getY(i), p.getZ(i));
      for (let i = 0; i < n.count; i++) normals.push(n.getX(i), n.getY(i), n.getZ(i));
      if (idx) { for (let i = 0; i < idx.count; i++) indices.push(idx.getX(i) + vertexOffset); }
      vertexOffset += p.count;
    }
  }
  merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  merged.setIndex(indices);
  merged.computeVertexNormals();
  return merged;
}

export function preloadAllModels(organs?: Record<OrganName, OrganHealth>): Promise<void[]> {
  return Promise.all(
    Object.keys(ORGAN_MODEL_URLS).map((name) => {
      const organName = name as OrganName;
      const health = organs?.[organName] ?? {
        organ: organName, healthScore: 100, riskScore: 0, tumorBurden: 0,
        inflammation: 0, perfusion: 100, fibrosis: 0, necrosis: 0,
        metastasisRisk: 0, obstruction: 0, functionScore: 100,
        color: '#ffffff', status: 'healthy' as const, affected: false,
        lesions: [], biomarkers: {},
      };
      return loadOrganModel(organName, health).then(() => undefined).catch(() => undefined);
    })
  );
}

export function getModelLoadingProgress(): number {
  return modelCache.size / Object.keys(ORGAN_MODEL_URLS).length;
}

export function clearModelCache() {
  for (const [_, data] of modelCache) {
    data.geometry.dispose();
  }
  modelCache.clear();
}

export function getOrganMaterial(organ: OrganName, health: OrganHealth): THREE.MeshPhysicalMaterial {
  const baseColor = health.color;
  const riskFactor = health.riskScore / 100;
  const color = new THREE.Color(baseColor).lerp(new THREE.Color('#ef565f'), riskFactor * 0.4);

  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.45, metalness: 0.02,
    clearcoat: 0.15, clearcoatRoughness: 0.5,
    transparent: true, opacity: 0.88,
    side: THREE.DoubleSide,
  });
}

export function createTumorMarker(organHealth: OrganHealth): {
  geometry: THREE.BufferGeometry; material: THREE.Material; position: [number, number, number];
} | null {
  if (organHealth.tumorBurden < 10) return null;
  const tumorSize = 0.12 + (organHealth.tumorBurden / 100) * 0.25;
  const tumorGeo = new THREE.SphereGeometry(tumorSize, 16, 16);
  const p = tumorGeo.getAttribute('position');
  for (let i = 0; i < p.count; i++) {
    const vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
    const noise = Math.sin(vx * 8) * Math.cos(vy * 6) * Math.sin(vz * 7) * 0.08;
    p.setXYZ(i, vx + noise, vy + noise * 0.8, vz + noise * 0.6);
  }
  p.needsUpdate = true;
  tumorGeo.computeVertexNormals();

  return {
    geometry: tumorGeo,
    material: new THREE.MeshPhysicalMaterial({
      color: '#cc2233', emissive: '#ff1122',
      emissiveIntensity: 0.4 + organHealth.tumorBurden / 200,
      roughness: 0.3, metalness: 0.05, clearcoat: 0.1,
      transparent: true, opacity: 0.85,
    }),
    position: [0.2, 0, 0],
  };
}

export function isModelLoaded(organ: OrganName): boolean {
  return modelCache.has(organ);
}
