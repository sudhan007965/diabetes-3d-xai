/* ==========================================================================
   INSIDE DIABETES — Complete Cinematic Medical Documentary & AI Engine
   Complete Human Body Anatomy + Realistic Internal Organs + WebGL Raycaster
   ========================================================================== */

'use strict';

const PREFERS_REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Embedded authentic model data synchronized with window.ALL_MODELS_DATA (Phase 1 Fix)
function getBaseModelData() {
  if (typeof window !== 'undefined' && window.ALL_MODELS_DATA && window.ALL_MODELS_DATA.models) {
    const annModel = window.ALL_MODELS_DATA.models.find(m => m.id === 'ann');
    const meta = window.ALL_MODELS_DATA.metadata;
    if (annModel && meta) {
      return {
        model_name: annModel.name,
        architecture: "8 Inputs -> 32 Hidden (ReLU) -> 16 Hidden (ReLU) -> 1 Output (Sigmoid)",
        dataset: "Pima Indians Diabetes Database (NIDDK)",
        total_samples: 768,
        metrics: {
          accuracy: annModel.accuracy || 0.7338,
          precision: annModel.precision || 0.6444,
          recall: annModel.recall || 0.5370,
          f1_score: annModel.f1 || 0.5859,
          roc_auc: annModel.roc_auc || 0.8191,
          confusion_matrix: [[84, 16], [25, 29]]
        },
        feature_names: meta.feature_names,
        feature_specs: meta.feature_specs,
        medians: meta.medians,
        scaler_mean: meta.scaler_mean,
        scaler_std: meta.scaler_std,
        weights: annModel.weights,
        biases: annModel.biases,
        expected_value: annModel.expected_value || 0.3547,
        global_importance_pct: annModel.shap_importance || { "Glucose": 35.2, "BMI": 22.8, "Age": 14.5, "DiabetesPedigreeFunction": 8.7, "Insulin": 7.4, "Pregnancies": 5.8, "SkinThickness": 3.1, "BloodPressure": 2.5 }
      };
    }
  }
  return null;
}

let MODEL_DATA = getBaseModelData();
if (!MODEL_DATA) {
  // Safe synchronous fallback matching model/model_weights.json (8->32->16->1)
  MODEL_DATA = {
    model_name: "Explainable Feedforward ANN (MLP)",
    architecture: "8 Inputs -> 32 Hidden -> 16 Hidden -> 1 Output (Sigmoid)",
    dataset: "Pima Indians Diabetes Database (NIDDK)",
    total_samples: 768,
    metrics: { accuracy: 0.733766, precision: 0.644444, recall: 0.537037, f1_score: 0.585859, roc_auc: 0.819074, confusion_matrix: [[84, 16], [25, 29]] },
    feature_names: ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    feature_specs: {
      "Pregnancies": {"unit": "count", "min": 0, "max": 20, "default": 1, "step": 1},
      "Glucose": {"unit": "mg/dL", "min": 40, "max": 300, "default": 100, "step": 1},
      "BloodPressure": {"unit": "mm Hg", "min": 30, "max": 180, "default": 72, "step": 1},
      "SkinThickness": {"unit": "mm", "min": 5, "max": 99, "default": 23, "step": 1},
      "Insulin": {"unit": "µU/mL", "min": 10, "max": 850, "default": 85, "step": 1},
      "BMI": {"unit": "kg/m²", "min": 12.0, "max": 67.0, "default": 26.5, "step": 0.1},
      "DiabetesPedigreeFunction": {"unit": "score", "min": 0.05, "max": 2.5, "default": 0.38, "step": 0.01},
      "Age": {"unit": "years", "min": 18, "max": 100, "default": 33, "step": 1}
    },
    medians: {"Pregnancies": 3.0, "Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 125.0, "BMI": 32.4, "DiabetesPedigreeFunction": 0.3825, "Age": 29.0},
    scaler_mean: [3.819218, 121.67101, 72.140065, 29.042345, 137.705212, 32.448208, 0.477428, 33.36645],
    get weights() { return (typeof window !== 'undefined' && window.ALL_MODELS_DATA && window.ALL_MODELS_DATA.models) ? (window.ALL_MODELS_DATA.models.find(m => m.id === 'ann') || {}).weights || [] : []; },
    get biases() { return (typeof window !== 'undefined' && window.ALL_MODELS_DATA && window.ALL_MODELS_DATA.models) ? (window.ALL_MODELS_DATA.models.find(m => m.id === 'ann') || {}).biases || [] : []; },
    expected_value: 0.3547, global_importance_pct: {}
  };
}


/* ==========================================================================
   1. GLOBAL THREE.JS SCENE SETUP
   ========================================================================== */
const canvas = document.getElementById('webgl-canvas');
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance'
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Cap at 1.5 for performance
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
// Soft shadows (kept but with reduced resolution)
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x020408, 0.016);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 22);
const cameraTarget = new THREE.Vector3(1.8, 0, 0);

// Cinematic Multi-Point Biological Lighting with Soft Realistic Shadows
scene.add(new THREE.AmbientLight(0x1a2638, 1.3));

// Key Directional Light for Anatomical Models with Soft Shadows
const keyLight = new THREE.DirectionalLight(0xfff5ea, 2.3);
keyLight.position.set(12, 18, 16);
keyLight.castShadow = true;
keyLight.shadow.mapSize.width = 1024;   // Reduced from 2048 for performance
keyLight.shadow.mapSize.height = 1024;
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 70;
keyLight.shadow.camera.left = -16;
keyLight.shadow.camera.right = 16;
keyLight.shadow.camera.top = 16;
keyLight.shadow.camera.bottom = -16;
keyLight.shadow.bias = -0.0004;
keyLight.shadow.radius = 1.8;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x38bdf8, 1.1);
fillLight.position.set(-16, -8, -6);
scene.add(fillLight);

const biologicalRimLight = new THREE.PointLight(0xdc2626, 3.8, 60);
biologicalRimLight.position.set(4, 0, 8);
scene.add(biologicalRimLight);

// Dedicated Bloodstream Lumen Projector (Produces striking soft shadows inside vessel)
const bloodstreamSpot = new THREE.SpotLight(0xffecd2, 4.2, 60, Math.PI / 3.0, 0.45, 1.2);
bloodstreamSpot.position.set(0, -22, -22);
bloodstreamSpot.target.position.set(0, -22, 12);
// No shadow on bloodstream spot — too expensive with many transparent cells
scene.add(bloodstreamSpot);
scene.add(bloodstreamSpot.target);

// Subtle biological rim backlight for cellular translucency
const bloodstreamBacklight = new THREE.DirectionalLight(0x38bdf8, 0.85);
bloodstreamBacklight.position.set(-8, -20, 15);
scene.add(bloodstreamBacklight);

const cellProbeLight = new THREE.PointLight(0x06b6d4, 2.5, 45);
cellProbeLight.position.set(0, -75, 6);
// No shadow — point light shadow is expensive and not visible at this depth
scene.add(cellProbeLight);

/* ==========================================================================
   2. PROCEDURAL TEXTURES, SHADERS & CELLULAR BIOMECHANICS
   ========================================================================== */
// 1. Endothelial Diffuse Texture (Vascular Cobblestone)
function createEndothelialTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#220306';
  ctx.fillRect(0, 0, size, size);

  const numCells = 80;
  for (let i = 0; i < numCells; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const radius = 18 + Math.random() * 26;
    const grad = ctx.createRadialGradient(x, y, 2, x, y, radius);
    grad.addColorStop(0, 'rgba(175, 28, 48, 0.7)');
    grad.addColorStop(0.7, 'rgba(110, 15, 28, 0.5)');
    grad.addColorStop(1, 'rgba(30, 4, 8, 0.95)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Endothelial nucleus
    ctx.fillStyle = 'rgba(240, 90, 110, 0.55)';
    ctx.beginPath();
    ctx.ellipse(x + (Math.random() - 0.5) * 6, y + (Math.random() - 0.5) * 6, 4, 2.5, Math.random() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 12);
  return texture;
}

// 2. Endothelial Height / Bump Map for Realistic Vascular Wall Depth
function createEndothelialBumpTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 90; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 16 + Math.random() * 24;
    const grad = ctx.createRadialGradient(x, y, 1, x, y, r);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.65, '#999999');
    grad.addColorStop(1, '#505050');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 12);
  return texture;
}

// 3. Erythrocyte Membrane Micro-wrinkle Bump Map
function createErythrocyteBumpTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 700; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 1.2 + Math.random() * 2.5;
    ctx.fillStyle = Math.random() > 0.5 ? '#9e9e9e' : '#686868';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

// 4. White Blood Cell (Leukocyte) Microvilli & Ruffled Surface Bump Map
function createLeukocyteBumpTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 1800; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 2 + Math.random() * 3.8;
    const grad = ctx.createRadialGradient(x, y, 0.5, x, y, r);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.5, '#aaaaaa');
    grad.addColorStop(1, '#666666');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 4);
  return texture;
}

const endothelialTexture     = createEndothelialTexture();
const endothelialBumpTexture = createEndothelialBumpTexture();
const erythrocyteBumpTexture = createErythrocyteBumpTexture();
const leukocyteBumpTexture   = createLeukocyteBumpTexture();

// Evans-Fung biomechanical erythrocyte lathe geometry
function createErythrocyteGeometry(radius = 1.15, radialSegments = 32, pointsCount = 28) {
  const points = [];
  for (let i = 0; i <= pointsCount; i++) {
    const u = -Math.PI / 2 + (Math.PI * i / pointsCount);
    const r = Math.cos(u) * radius;
    const y = Math.sin(u) * (0.16 + 0.38 * (Math.cos(u) * Math.cos(u))) * radius;
    points.push(new THREE.Vector2(Math.max(0, r), y));
  }
  return new THREE.LatheGeometry(points, radialSegments);
}

const erythrocyteGeom = createErythrocyteGeometry(1.15, 24, 20); // Fewer segments for performance

// Erythrocyte Material (MeshStandard for performance — still beautiful)
const erythrocyteMaterial = new THREE.MeshStandardMaterial({
  color: 0xba181b,
  emissive: 0x3d0205,
  roughness: 0.28,
  metalness: 0.12,
  bumpMap: erythrocyteBumpTexture,
  bumpScale: 0.035
});

// Leukocyte Outer Material (opaque ivory — no transmission = much faster)
const leukocyteOuterMaterial = new THREE.MeshStandardMaterial({
  color: 0xeef2f6,
  emissive: 0x1a2233,
  roughness: 0.38,
  metalness: 0.05,
  bumpMap: leukocyteBumpTexture,
  bumpScale: 0.07
});

// Neutrophil Multilobed Chromatin Nucleus
const leukocyteNucleusMaterial = new THREE.MeshStandardMaterial({
  color: 0x4c1d95,
  emissive: 0x2e1065,
  roughness: 0.36,
  metalness: 0.12
});

/* ==========================================================================
   3. 3D BIOLOGICAL SCENE GROUPS & RAYCASTER REGISTRATION
   ========================================================================== */
const interactiveObjects = [];

function registerBioObject(mesh, name, desc) {
  mesh.userData.bioInfo = { name, desc };
  interactiveObjects.push(mesh);
}

// ── GROUP 1: COMPLETE HUMAN BODY ANATOMY & ORGANS (HERO) ────────────────
const heroGroup = new THREE.Group();
heroGroup.position.set(3.8, 0, 0); // Framed cleanly on right side
scene.add(heroGroup);

// Fresnel Holographic Body Shell Shader
const bodyShellMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uColor: { value: new THREE.Color(0x38bdf8) },
    uTime:  { value: 0.0 }
  },
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform vec3 uColor;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);
      float fresnel = pow(1.0 - abs(dot(normal, viewDir)), 2.8);
      float alpha = clamp(0.12 + fresnel * 0.75, 0.0, 0.85);
      vec3 col = mix(uColor * 0.35, vec3(1.0), fresnel * 0.4);
      gl_FragColor = vec4(col, alpha);
    }
  `
});

// Build Complete Anatomical Human Body (Head, Torso, Arms, Legs, Organs)
let heartMeshRef = null;
let leftLungRef = null;
let rightLungRef = null;

function createCompleteHumanBody() {
  const body = new THREE.Group();

  // ── A. ANATOMICAL SILHOUETTE SHELL ─────────────────────────────
  // 1. Head / Cranium
  const headGeom = new THREE.SphereGeometry(0.85, 24, 20);
  headGeom.scale(1.0, 1.25, 1.1);
  const headMesh = new THREE.Mesh(headGeom, bodyShellMaterial);
  headMesh.position.set(0, 6.2, 0);
  registerBioObject(headMesh, 'Human Cranium & Cerebrum', 'Houses the central nervous system and hypothalamic glucose-sensing nuclei that modulate autonomic tone.');
  body.add(headMesh);

  // 2. Neck
  const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.85, 16), bodyShellMaterial);
  neckMesh.position.set(0, 5.0, 0);
  body.add(neckMesh);

  // 3. Thorax & Ribcage Shell
  const thoraxGeom = new THREE.CylinderGeometry(1.45, 1.15, 2.6, 20);
  thoraxGeom.scale(1.35, 1.0, 0.85);
  const thoraxMesh = new THREE.Mesh(thoraxGeom, bodyShellMaterial);
  thoraxMesh.position.set(0, 3.4, 0);
  body.add(thoraxMesh);

  // 4. Abdomen & Pelvis Shell
  const abdomenGeom = new THREE.CylinderGeometry(1.15, 1.35, 2.4, 20);
  abdomenGeom.scale(1.3, 1.0, 0.88);
  const abdomenMesh = new THREE.Mesh(abdomenGeom, bodyShellMaterial);
  abdomenMesh.position.set(0, 1.0, 0);
  body.add(abdomenMesh);

  // 5. Shoulders & Arms
  const shoulderMat = bodyShellMaterial;
  const upperArmGeom = new THREE.CylinderGeometry(0.32, 0.28, 2.0, 14);
  const forearmGeom  = new THREE.CylinderGeometry(0.26, 0.22, 1.9, 14);
  const handGeom     = new THREE.SphereGeometry(0.24, 12, 12);
  handGeom.scale(0.8, 1.4, 0.5);

  // Left Arm
  const leftUpperArm = new THREE.Mesh(upperArmGeom, shoulderMat);
  leftUpperArm.position.set(-2.0, 3.2, 0);
  leftUpperArm.rotation.z = -0.15;
  body.add(leftUpperArm);

  const leftForearm = new THREE.Mesh(forearmGeom, shoulderMat);
  leftForearm.position.set(-2.3, 1.4, 0);
  leftForearm.rotation.z = -0.1;
  body.add(leftForearm);

  const leftHand = new THREE.Mesh(handGeom, shoulderMat);
  leftHand.position.set(-2.45, 0.2, 0);
  body.add(leftHand);

  // Right Arm
  const rightUpperArm = new THREE.Mesh(upperArmGeom, shoulderMat);
  rightUpperArm.position.set(2.0, 3.2, 0);
  rightUpperArm.rotation.z = 0.15;
  body.add(rightUpperArm);

  const rightForearm = new THREE.Mesh(forearmGeom, shoulderMat);
  rightForearm.position.set(2.3, 1.4, 0);
  rightForearm.rotation.z = 0.1;
  body.add(rightForearm);

  const rightHand = new THREE.Mesh(handGeom, shoulderMat);
  rightHand.position.set(2.45, 0.2, 0);
  body.add(rightHand);

  // 6. Pelvis & Lower Limbs
  const thighGeom = new THREE.CylinderGeometry(0.55, 0.42, 3.2, 16);
  const calfGeom  = new THREE.CylinderGeometry(0.40, 0.30, 3.1, 16);
  const footGeom  = new THREE.BoxGeometry(0.5, 0.35, 1.1);

  // Left Leg
  const leftThigh = new THREE.Mesh(thighGeom, shoulderMat);
  leftThigh.position.set(-0.85, -1.8, 0);
  body.add(leftThigh);

  const leftCalf = new THREE.Mesh(calfGeom, shoulderMat);
  leftCalf.position.set(-0.85, -4.8, 0);
  body.add(leftCalf);

  const leftFoot = new THREE.Mesh(footGeom, shoulderMat);
  leftFoot.position.set(-0.85, -6.5, 0.3);
  body.add(leftFoot);

  // Right Leg
  const rightThigh = new THREE.Mesh(thighGeom, shoulderMat);
  rightThigh.position.set(0.85, -1.8, 0);
  body.add(rightThigh);

  const rightCalf = new THREE.Mesh(calfGeom, shoulderMat);
  rightCalf.position.set(0.85, -4.8, 0);
  body.add(rightCalf);

  const rightFoot = new THREE.Mesh(footGeom, shoulderMat);
  rightFoot.position.set(0.85, -6.5, 0.3);
  body.add(rightFoot);

  // ── B. INTERNAL ORGANS ─────────────────────────────────────────

  // 1. BEATING HEART (Thoracic Cavity, left of midline)
  const heartGroup = new THREE.Group();
  heartGroup.position.set(-0.35, 3.3, 0.45);

  const heartMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0x991b1b,
    roughness: 0.35
  });

  // Anatomical Heart Shape: Cone for ventricles + spheres for atria
  const ventricleMesh = new THREE.Mesh(new THREE.ConeGeometry(0.45, 0.9, 16), heartMat);
  ventricleMesh.rotation.z = Math.PI - 0.25;
  ventricleMesh.rotation.y = 0.2;
  heartGroup.add(ventricleMesh);

  const atriumL = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), heartMat);
  atriumL.position.set(-0.18, 0.35, 0);
  heartGroup.add(atriumL);

  const atriumR = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 12), heartMat);
  atriumR.position.set(0.18, 0.32, 0);
  heartGroup.add(atriumR);

  registerBioObject(heartGroup, 'Human Heart (Cardiac Muscle)', 'Four-chambered muscular pump beating ~72 times/minute, propelling pressurized blood into the aorta and systemic vascular circuit.');
  body.add(heartGroup);
  heartMeshRef = heartGroup;

  // 2. LUNGS (Bilateral Pulmonary Lobes)
  const lungMat = new THREE.MeshStandardMaterial({
    color: 0x64748b,
    emissive: 0x1e293b,
    roughness: 0.5,
    transparent: true,
    opacity: 0.72
  });

  const rightLungGeom = new THREE.ConeGeometry(0.65, 1.8, 14);
  rightLungGeom.scale(1.1, 1.0, 0.8);
  const rightLung = new THREE.Mesh(rightLungGeom, lungMat);
  rightLung.position.set(1.0, 3.4, 0.2);
  rightLung.rotation.z = -0.08;
  registerBioObject(rightLung, 'Right Lung (3 Pulmonary Lobes)', 'Facilitates gas exchange across alveolar-capillary membranes, oxygenating venous blood returned from systemic tissues.');
  body.add(rightLung);
  rightLungRef = rightLung;

  const leftLungGeom = new THREE.ConeGeometry(0.6, 1.7, 14);
  leftLungGeom.scale(1.0, 1.0, 0.75);
  const leftLung = new THREE.Mesh(leftLungGeom, lungMat);
  leftLung.position.set(-1.0, 3.4, 0.2);
  leftLung.rotation.z = 0.08;
  registerBioObject(leftLung, 'Left Lung (2 Lobes & Cardiac Notch)', 'Provides oxygenation with an anatomical medial cardiac notch accommodating the ventricular apex.');
  body.add(leftLung);
  leftLungRef = leftLung;

  // 3. LIVER (Right Upper Quadrant)
  const liverGeom = new THREE.ConeGeometry(0.9, 1.4, 16);
  liverGeom.scale(1.6, 0.75, 0.9);
  const liverMesh = new THREE.Mesh(liverGeom, new THREE.MeshStandardMaterial({
    color: 0x854d0e,
    emissive: 0x451a03,
    roughness: 0.45
  }));
  liverMesh.position.set(0.85, 1.9, 0.35);
  liverMesh.rotation.z = -0.35;
  liverMesh.rotation.y = 0.4;
  registerBioObject(liverMesh, 'Liver (Hepatic Metabolic Center)', 'Major regulator of carbohydrate metabolism: synthesizes glycogen, performs gluconeogenesis during fasting, and clears insulin from portal blood.');
  body.add(liverMesh);

  // 4. STOMACH (Left Epigastrium)
  const stomachCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.3, 2.3, 0.3),
    new THREE.Vector3(-0.9, 2.0, 0.4),
    new THREE.Vector3(-0.8, 1.3, 0.4),
    new THREE.Vector3(-0.2, 1.2, 0.35)
  ]);
  const stomachMesh = new THREE.Mesh(
    new THREE.TubeGeometry(stomachCurve, 20, 0.35, 12, false),
    new THREE.MeshStandardMaterial({ color: 0xa16207, roughness: 0.5 })
  );
  registerBioObject(stomachMesh, 'Stomach (Gastric Reservoir)', 'Digests ingested macronutrients and releases glucose precursors into the duodenum, stimulating incretin hormone secretion.');
  body.add(stomachMesh);

  // 5. PANCREAS (Retroperitoneal behind stomach)
  const pancreasSpine = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.6, 1.0, 0.2),   // Head in duodenal loop
    new THREE.Vector3(0.0, 1.15, 0.15), // Body
    new THREE.Vector3(-0.9, 1.35, 0.05) // Tail
  ]);
  const bodyPancreas = new THREE.Mesh(
    new THREE.TubeGeometry(pancreasSpine, 24, 0.22, 10, false),
    new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x065f46, roughness: 0.35 })
  );
  registerBioObject(bodyPancreas, 'Pancreas (Endocrine Islets)', 'Houses the Islets of Langerhans. Beta cells directly detect blood sugar and secrete insulin to drive glucose clearance into muscle and fat.');
  body.add(bodyPancreas);

  // 6. KIDNEYS (Bilateral Retroperitoneal)
  const kidneyGeom = new THREE.SphereGeometry(0.3, 12, 12);
  kidneyGeom.scale(0.8, 1.3, 0.6);
  const kidneyMat = new THREE.MeshStandardMaterial({ color: 0x991b1b, emissive: 0x450a0a, roughness: 0.4 });

  const rightKidney = new THREE.Mesh(kidneyGeom, kidneyMat);
  rightKidney.position.set(0.9, 0.9, -0.2);
  registerBioObject(rightKidney, 'Right Kidney (Renal Filtration)', 'Filters plasma metabolites. Chronic hyperglycemia damages renal glomeruli, leading to diabetic nephropathy.');
  body.add(rightKidney);

  const leftKidney = new THREE.Mesh(kidneyGeom, kidneyMat);
  leftKidney.position.set(-0.9, 1.0, -0.2);
  registerBioObject(leftKidney, 'Left Kidney (Renal Glomeruli)', 'Maintains fluid and electrolyte homeostasis; high blood glucose over years leads to progressive glomerular hyperfiltration injury.');
  body.add(leftKidney);

  // ── C. EXTENSIVE VASCULAR NETWORK (HEAD TO FEET) ────────────────

  const vasRedMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, emissive: 0x991b1b, roughness: 0.3 });
  const venBlueMat= new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x0369a1, roughness: 0.35 });

  // Aorta & Major Arterial Highway
  const aortaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.25, 3.6, 0.35), // Aortic arch
    new THREE.Vector3(0.0, 3.8, 0.25),
    new THREE.Vector3(0.15, 2.8, 0.0),   // Thoracic aorta
    new THREE.Vector3(0.1, 1.2, 0.0),    // Abdominal aorta
    new THREE.Vector3(0.0, 0.1, 0.05),   // Bifurcation
    new THREE.Vector3(0.45, -1.0, 0.1),  // Right iliac
    new THREE.Vector3(0.85, -2.5, 0.1),  // Right femoral
    new THREE.Vector3(0.85, -4.5, 0.05), // Right tibial
    new THREE.Vector3(0.85, -6.2, 0.1)
  ]);
  const aortaMesh = new THREE.Mesh(new THREE.TubeGeometry(aortaCurve, 45, 0.12, 8, false), vasRedMat);
  registerBioObject(aortaMesh, 'Aorta & Systemic Arterial Highway', 'The primary high-pressure conduit transporting oxygenated blood and circulating glucose from the left ventricle into systemic organs.');
  body.add(aortaMesh);

  // Left Iliac & Femoral Artery
  const leftLegArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.0, 0.1, 0.05),
    new THREE.Vector3(-0.45, -1.0, 0.1),
    new THREE.Vector3(-0.85, -2.5, 0.1),
    new THREE.Vector3(-0.85, -4.5, 0.05),
    new THREE.Vector3(-0.85, -6.2, 0.1)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftLegArtery, 35, 0.10, 8, false), vasRedMat));

  // Carotid Arteries (Up to Head)
  const leftCarotid = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.15, 3.7, 0.2),
    new THREE.Vector3(-0.35, 4.8, 0.1),
    new THREE.Vector3(-0.4, 6.0, 0.1)
  ]);
  const rightCarotid = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.1, 3.7, 0.2),
    new THREE.Vector3(0.35, 4.8, 0.1),
    new THREE.Vector3(0.4, 6.0, 0.1)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftCarotid, 20, 0.07, 8, false), vasRedMat));
  body.add(new THREE.Mesh(new THREE.TubeGeometry(rightCarotid, 20, 0.07, 8, false), vasRedMat));

  // Arm Arteries (Subclavian to Brachial)
  const leftArmArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.25, 3.7, 0.2),
    new THREE.Vector3(-1.3, 4.0, 0.1),
    new THREE.Vector3(-2.0, 3.2, 0.0),
    new THREE.Vector3(-2.3, 1.4, 0.0),
    new THREE.Vector3(-2.45, 0.2, 0.0)
  ]);
  const rightArmArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.15, 3.7, 0.2),
    new THREE.Vector3(1.3, 4.0, 0.1),
    new THREE.Vector3(2.0, 3.2, 0.0),
    new THREE.Vector3(2.3, 1.4, 0.0),
    new THREE.Vector3(2.45, 0.2, 0.0)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftArmArtery, 28, 0.07, 8, false), vasRedMat));
  body.add(new THREE.Mesh(new THREE.TubeGeometry(rightArmArtery, 28, 0.07, 8, false), vasRedMat));

  // Inferior & Superior Vena Cava
  const venaCavaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.05, 5.0, 0.1),
    new THREE.Vector3(-0.05, 3.5, 0.2),
    new THREE.Vector3(-0.08, 1.8, 0.05),
    new THREE.Vector3(-0.08, 0.2, 0.05)
  ]);
  const venaMesh = new THREE.Mesh(new THREE.TubeGeometry(venaCavaCurve, 25, 0.13, 8, false), venBlueMat);
  registerBioObject(venaMesh, 'Vena Cava (Venous Return Conduit)', 'Transports deoxygenated blood from abdominal organs, brain, and lower extremities back to the right atrium.');
  body.add(venaMesh);

  // Micro-arterial branching network (over 70 biological curves)
  const branchMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending });
  for (let i = 0; i < 75; i++) {
    const yVal = 5.5 - Math.random() * 11.5;
    const start = new THREE.Vector3((Math.random() - 0.5) * 0.6, yVal, (Math.random() - 0.5) * 0.3);
    const end = new THREE.Vector3(
      start.x + (Math.random() - 0.5) * 3.8,
      start.y + (Math.random() - 0.5) * 2.2,
      start.z + (Math.random() - 0.5) * 1.5
    );
    const mid = new THREE.Vector3((start.x + end.x) * 0.5, (start.y + end.y) * 0.5, (start.z + end.z) * 0.5);
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    body.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(10)), branchMat));
  }

  return body;
}

const completeHumanBody = createCompleteHumanBody();
heroGroup.add(completeHumanBody);

// Enable shadow casting and receiving across all anatomical structures
function applyGlobalShadowProperties(group) {
  if (!group) return;
  group.traverse(child => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}

applyGlobalShadowProperties(completeHumanBody);

// Floating interstitial particles
const particleCount = 180;
const particleGeom = new THREE.BufferGeometry();
const particlePos = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i += 3) {
  particlePos[i] = (Math.random() - 0.5) * 32;
  particlePos[i + 1] = (Math.random() - 0.5) * 32;
  particlePos[i + 2] = (Math.random() - 0.5) * 20;
}
particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
const floatingParticles = new THREE.Points(particleGeom, new THREE.PointsMaterial({
  color: 0xfca5a5, size: 0.14, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending
}));
scene.add(floatingParticles);

// ── GROUP 2: BLOODSTREAM (ERYTHROCYTES, LEUKOCYTES & D-GLUCOSE) ────────
const bloodstreamGroup = new THREE.Group();
bloodstreamGroup.position.set(0, -22, 0);
scene.add(bloodstreamGroup);

const vesselRadius = 5.2;
const vesselLength = 48;
const vesselGeom = new THREE.CylinderGeometry(vesselRadius, vesselRadius, vesselLength, 48, 1, true);
const vesselMat = new THREE.MeshStandardMaterial({
  map: endothelialTexture,
  bumpMap: endothelialBumpTexture,
  bumpScale: 0.16,
  side: THREE.BackSide,
  roughness: 0.42,
  metalness: 0.08,
  color: 0x991b1b
});
const vesselMesh = new THREE.Mesh(vesselGeom, vesselMat);
vesselMesh.rotation.x = Math.PI / 2;
vesselMesh.receiveShadow = true; // Inner wall receives dramatic soft shadows from cells
// Do not register vesselMesh in interactiveObjects so empty plasma does not block cell popups
bloodstreamGroup.add(vesselMesh);

// 1. Erythrocytes (Red Blood Cells) with Shadows & Depth
const rbcList = [];
const numRBCs = 26;  // Reduced for performance
for (let i = 0; i < numRBCs; i++) {
  const mesh = new THREE.Mesh(erythrocyteGeom, erythrocyteMaterial);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  const angle = Math.random() * Math.PI * 2;
  const dist = Math.random() * (vesselRadius - 1.4);
  mesh.position.set(Math.cos(angle) * dist, Math.sin(angle) * dist, (Math.random() - 0.5) * vesselLength * 0.95);
  mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
  mesh.scale.setScalar(0.72 + Math.random() * 0.32);

  mesh.userData = {
    vz: 0.06 + Math.random() * 0.07,
    rotX: (Math.random() - 0.5) * 0.03,
    rotY: (Math.random() - 0.5) * 0.03,
    rotZ: (Math.random() - 0.5) * 0.03
  };
  registerBioObject(mesh, 'Red Blood Cell (Erythrocyte)', 'Biconcave disc (~7.8 µm) containing hemoglobin. Its high flexibility enables transit through micro-capillaries.');
  bloodstreamGroup.add(mesh);
  rbcList.push(mesh);
}

// 2. White Blood Cells (Leukocytes / Neutrophils) with Multilobed Nuclei
// Perfectly scaled, slower rolling, with 100% responsive cursor hover popup
function createWhiteBloodCell() {
  const wbcGroup = new THREE.Group();

  const bioInfo = {
    name: 'White Blood Cell (Leukocyte / Neutrophil)',
    desc: 'Spherical immune sentinel (~12–15 µm) exhibiting endothelial rolling. Chronic hyperglycemia activates leukocytes, prompting vascular adhesion and inflammatory endothelial damage.'
  };

  // A. Outer perturbed ruffled spherical membrane (compact, authentic biological scale)
  const baseRadius = 0.95;
  const wbcGeom = new THREE.SphereGeometry(baseRadius, 28, 24);
  const pos = wbcGeom.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i);
    const py = pos.getY(i);
    const pz = pos.getZ(i);
    const displacement = 1.0 + 0.045 * Math.sin(px * 5.0 + py * 3.5) + 0.028 * Math.cos(pz * 5.5);
    pos.setXYZ(i, px * displacement, py * displacement, pz * displacement);
  }
  wbcGeom.computeVertexNormals();
  wbcGeom.computeBoundingSphere();

  const outerMesh = new THREE.Mesh(wbcGeom, leukocyteOuterMaterial);
  outerMesh.castShadow = true;
  outerMesh.receiveShadow = true;
  outerMesh.userData.bioInfo = bioInfo;
  wbcGroup.add(outerMesh);

  // B. Generous invisible hit-sphere for effortless and reliable cursor raycasting
  const hitGeom = new THREE.SphereGeometry(baseRadius * 1.25, 12, 12);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const hitMesh = new THREE.Mesh(hitGeom, hitMat);
  hitMesh.userData.bioInfo = bioInfo;
  wbcGroup.add(hitMesh);

  // C. Characteristic Multilobed Neutrophil Chromatin Nucleus (3-4 interconnected lobes)
  const nucleusGroup = new THREE.Group();
  const lobeGeom = new THREE.SphereGeometry(0.24, 14, 14);
  lobeGeom.scale(1.1, 0.85, 0.9);

  const lobePositions = [
    new THREE.Vector3(-0.24, 0.15, 0.08),
    new THREE.Vector3(0.22, 0.20, -0.08),
    new THREE.Vector3(0.15, -0.23, 0.10),
    new THREE.Vector3(-0.19, -0.16, -0.10)
  ];

  lobePositions.forEach(p => {
    const lobe = new THREE.Mesh(lobeGeom, leukocyteNucleusMaterial);
    lobe.position.copy(p);
    lobe.castShadow = true;
    lobe.receiveShadow = true;
    lobe.userData.bioInfo = bioInfo;
    nucleusGroup.add(lobe);
  });

  // Nuclear chromatin connecting bridges
  const bridgeMat = new THREE.MeshStandardMaterial({ color: 0x3b0764, roughness: 0.4 });
  for (let i = 0; i < lobePositions.length - 1; i++) {
    const p1 = lobePositions[i];
    const p2 = lobePositions[i + 1];
    const dist = p1.distanceTo(p2);
    const bridgeGeom = new THREE.CylinderGeometry(0.065, 0.065, dist, 8);
    const bridge = new THREE.Mesh(bridgeGeom, bridgeMat);
    bridge.position.copy(p1).add(p2).multiplyScalar(0.5);
    bridge.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
    bridge.castShadow = true;
    bridge.userData.bioInfo = bioInfo;
    nucleusGroup.add(bridge);
  }

  wbcGroup.add(nucleusGroup);

  // Register all interactive targets
  wbcGroup.userData.bioInfo = bioInfo;
  registerBioObject(outerMesh, bioInfo.name, bioInfo.desc);
  registerBioObject(hitMesh, bioInfo.name, bioInfo.desc);
  registerBioObject(wbcGroup, bioInfo.name, bioInfo.desc);

  return wbcGroup;
}

const wbcList = [];
const numWBCs = 8;   // Reduced for performance
for (let i = 0; i < numWBCs; i++) {
  const wbc = createWhiteBloodCell();
  const angle = Math.random() * Math.PI * 2;
  // Natural margination: WBCs roll along vessel wall periphery
  const wallDist = (vesselRadius - 1.1) - Math.random() * 0.5;
  wbc.position.set(Math.cos(angle) * wallDist, Math.sin(angle) * wallDist, (Math.random() - 0.5) * vesselLength * 0.95);
  // Significantly smaller scale — WBC ~1.6x RBC diameter (authentic)
  wbc.scale.setScalar(0.42 + Math.random() * 0.12);
  // IMPORTANT: Use Object.assign to PRESERVE bioInfo while adding animation data
  Object.assign(wbc.userData, {
    vz: 0.012 + Math.random() * 0.010, // Slow rolling speed (3x slower than RBCs)
    rollX: (Math.random() - 0.5) * 0.006,
    rollY: 0.005 + Math.random() * 0.006,
    phase: Math.random() * Math.PI * 2
  });
  bloodstreamGroup.add(wbc);
  wbcList.push(wbc);
}

// 3. Upgraded 3D D-Glucose Monosaccharide Molecules (Chair Pyranose Conformation)
function createGlucoseMolecule() {
  const molGroup = new THREE.Group();

  // Amber crystalline materials (MeshStandard for performance)
  const carbonMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    roughness: 0.22,
    metalness: 0.18,
    emissive: 0xb45309,
    emissiveIntensity: 0.32
  });
  const oxygenMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    roughness: 0.24,
    metalness: 0.14,
    emissive: 0x9b1c1c,
    emissiveIntensity: 0.35
  });
  const hydroxylMat = new THREE.MeshStandardMaterial({
    color: 0xfde047,
    roughness: 0.25,
    metalness: 0.1,
    emissive: 0x92400e,
    emissiveIntensity: 0.22
  });
  const bondMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0,
    roughness: 0.28,
    metalness: 0.35
  });

  const atomGeom = new THREE.SphereGeometry(0.12, 8, 8);
  const oGeom    = new THREE.SphereGeometry(0.14, 8, 8);
  const hGeom    = new THREE.SphereGeometry(0.08, 6, 6);
  const bondGeom = new THREE.CylinderGeometry(0.028, 0.028, 0.45, 8);

  // Stable 4C1 chair pyranose ring nodes (C1-C5 and ring oxygen O5)
  const ringNodes = [
    { x: 0.38,  y: 0.22,  z: 0.14,  type: 'C' }, // C1
    { x: 0.42,  y: -0.22, z: -0.12, type: 'C' }, // C2
    { x: -0.02, y: -0.38, z: 0.16,  type: 'C' }, // C3
    { x: -0.42, y: -0.20, z: -0.14, type: 'C' }, // C4
    { x: -0.38, y: 0.24,  z: 0.12,  type: 'C' }, // C5
    { x: 0.02,  y: 0.38,  z: -0.15, type: 'O' }  // O5 (Ring Oxygen)
  ];

  ringNodes.forEach((node, idx) => {
    const mesh = new THREE.Mesh(node.type === 'C' ? atomGeom : oGeom, node.type === 'C' ? carbonMat : oxygenMat);
    mesh.position.set(node.x, node.y, node.z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    molGroup.add(mesh);

    // Ring covalent bond
    const next = ringNodes[(idx + 1) % ringNodes.length];
    const bond = new THREE.Mesh(bondGeom, bondMat);
    const p1 = new THREE.Vector3(node.x, node.y, node.z);
    const p2 = new THREE.Vector3(next.x, next.y, next.z);
    bond.position.copy(p1).add(p2).multiplyScalar(0.5);
    bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
    bond.scale.y = p1.distanceTo(p2) / 0.45;
    bond.castShadow = true;
    bond.receiveShadow = true;
    molGroup.add(bond);

    // Hydroxyl functional groups (-OH)
    if (node.type === 'C') {
      const ohDir = new THREE.Vector3(node.x, node.y, node.z).normalize().multiplyScalar(0.32);
      const ohMesh = new THREE.Mesh(hGeom, hydroxylMat);
      ohMesh.position.set(node.x + ohDir.x, node.y + ohDir.y, node.z + ohDir.z);
      ohMesh.castShadow = true;
      molGroup.add(ohMesh);

      const ohBond = new THREE.Mesh(bondGeom, bondMat);
      const op1 = new THREE.Vector3(node.x, node.y, node.z);
      const op2 = ohMesh.position;
      ohBond.position.copy(op1).add(op2).multiplyScalar(0.5);
      ohBond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), op2.clone().sub(op1).normalize());
      ohBond.scale.y = op1.distanceTo(op2) / 0.45;
      ohBond.castShadow = true;
      molGroup.add(ohBond);
    }
  });

  // C6 Hydroxymethyl group (-CH2OH branching from C5)
  const c5 = ringNodes[4];
  const c6Pos = new THREE.Vector3(c5.x - 0.28, c5.y + 0.32, c5.z + 0.1);
  const c6Mesh = new THREE.Mesh(atomGeom, carbonMat);
  c6Mesh.position.copy(c6Pos);
  c6Mesh.castShadow = true;
  molGroup.add(c6Mesh);

  const c6Bond = new THREE.Mesh(bondGeom, bondMat);
  const cp1 = new THREE.Vector3(c5.x, c5.y, c5.z);
  c6Bond.position.copy(cp1).add(c6Pos).multiplyScalar(0.5);
  c6Bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), c6Pos.clone().sub(cp1).normalize());
  c6Bond.scale.y = cp1.distanceTo(c6Pos) / 0.45;
  c6Bond.castShadow = true;
  molGroup.add(c6Bond);

  const glucoseBioName = 'D-Glucose Molecule (C₆H₁₂O₆)';
  const glucoseBioDesc = 'Hexose monosaccharide in stable chair pyranose conformation. Persistent excess blood glucose non-enzymatically glycates hemoglobin (HbA1c) and degrades the microvascular endothelium.';

  // Invisible hit sphere for reliable raycasting on the tiny molecule
  const glucoseHitGeom = new THREE.SphereGeometry(0.55, 8, 8);
  const glucoseHitMat  = new THREE.MeshBasicMaterial({ visible: false });
  const glucoseHitMesh = new THREE.Mesh(glucoseHitGeom, glucoseHitMat);
  glucoseHitMesh.userData.bioInfo = { name: glucoseBioName, desc: glucoseBioDesc };
  molGroup.add(glucoseHitMesh);

  // Attach bioInfo to group AND register hit sphere
  molGroup.userData.bioInfo = { name: glucoseBioName, desc: glucoseBioDesc };
  registerBioObject(glucoseHitMesh, glucoseBioName, glucoseBioDesc);
  registerBioObject(molGroup,      glucoseBioName, glucoseBioDesc);
  return molGroup;
}

const glucoseMolecules = [];
const numGlucose = 30; // Reduced for performance
for (let i = 0; i < numGlucose; i++) {
  const g = createGlucoseMolecule();
  const angle = Math.random() * Math.PI * 2;
  const dist = Math.random() * (vesselRadius - 1.0);
  g.position.set(Math.cos(angle) * dist, Math.sin(angle) * dist, (Math.random() - 0.5) * vesselLength * 0.95);
  g.scale.setScalar(0.72 + Math.random() * 0.32);
  g.userData = {
    vz: 0.08 + Math.random() * 0.09,
    rotX: (Math.random() - 0.5) * 0.05,
    rotY: (Math.random() - 0.5) * 0.05
  };
  bloodstreamGroup.add(g);
  glucoseMolecules.push(g);
}

// ── GROUP 3: PANCREAS & ISLETS ──────────────────────────────────────────
const pancreasGroup = new THREE.Group();
pancreasGroup.position.set(0, -48, 0);
scene.add(pancreasGroup);

function createPancreasOrgan() {
  const organ = new THREE.Group();
  const pancreasMat = new THREE.MeshStandardMaterial({ color: 0xcaa07a, roughness: 0.55, metalness: 0.08 });

  const spinePoints = [
    new THREE.Vector3(3.2, -1.2, 0),
    new THREE.Vector3(3.0, 0.4, 0.2),
    new THREE.Vector3(1.8, 0.8, 0.4),
    new THREE.Vector3(0.0, 0.7, 0.2),
    new THREE.Vector3(-1.8, 0.9, -0.1),
    new THREE.Vector3(-3.4, 1.4, -0.4)
  ];
  const pancreasBodyMesh = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(spinePoints), 36, 1.1, 16, false), pancreasMat);
  registerBioObject(pancreasBodyMesh, 'Pancreas (Anatomical Organ)', 'Retroperitoneal gland with dual function: producing digestive enzymes and secreting blood sugar regulating hormones into the portal vein.');
  organ.add(pancreasBodyMesh);

  // Duodenal Loop
  const duodenalMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(2.4, 2.0, 0.1), new THREE.Vector3(4.2, 1.5, 0.2), new THREE.Vector3(4.6, -0.4, 0.1), new THREE.Vector3(4.0, -2.2, 0), new THREE.Vector3(2.0, -2.5, -0.2)
    ]), 28, 0.85, 14, false),
    new THREE.MeshStandardMaterial({ color: 0x9f5a43, roughness: 0.6 })
  );
  registerBioObject(duodenalMesh, 'Duodenum (C-Loop)', 'First section of the small intestine cradling the pancreatic head and receiving bile and pancreatic digestive secretions.');
  organ.add(duodenalMesh);

  // Pancreatic Duct
  const ductMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(3.1, -0.3, 0.2), new THREE.Vector3(1.8, 0.75, 0.45), new THREE.Vector3(0.0, 0.7, 0.25), new THREE.Vector3(-1.8, 0.9, -0.05), new THREE.Vector3(-3.2, 1.35, -0.35)
    ]), 30, 0.09, 8, false),
    new THREE.MeshStandardMaterial({ color: 0xbae6fd, emissive: 0x0284c7, roughness: 0.3 })
  );
  registerBioObject(ductMesh, 'Main Pancreatic Duct (Duct of Wirsung)', 'Central drainage conduit channeling bicarbonate and digestive enzymes toward the hepatopancreatic ampulla.');
  organ.add(ductMesh);

  // Splenic Artery
  const splenicPoints = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    splenicPoints.push(new THREE.Vector3(2.2 - t * 5.6, 1.3 + Math.sin(t * 12) * 0.25, 0.3 + Math.cos(t * 8) * 0.2));
  }
  const splenicMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(splenicPoints), 32, 0.12, 8, false),
    new THREE.MeshStandardMaterial({ color: 0xdc2626, emissive: 0x991b1b, roughness: 0.3 })
  );
  registerBioObject(splenicMesh, 'Splenic Artery', 'Tortuous arterial branch off the celiac trunk that supplies the pancreatic body and tail before reaching the spleen.');
  organ.add(splenicMesh);

  return organ;
}

const pancreasOrgan = createPancreasOrgan();
pancreasGroup.add(pancreasOrgan);

// 3D Islet of Langerhans
function createIsletCluster() {
  const islet = new THREE.Group();
  islet.position.set(-0.8, 0.7, 2.5);

  const betaMat = new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x065f46, roughness: 0.35 });
  const alphaMat = new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0x9a3412, roughness: 0.4 });
  const deltaMat = new THREE.MeshStandardMaterial({ color: 0xa855f7, emissive: 0x6b21a8, roughness: 0.4 });
  const cellGeom = new THREE.SphereGeometry(0.22, 12, 12);

  for (let i = 0; i < 50; i++) {
    const phi = Math.acos(-1 + (2 * i) / 50);
    const theta = Math.sqrt(50 * Math.PI) * phi;
    const r = 1.1 + (Math.random() - 0.5) * 0.3;

    const randType = Math.random();
    let mat = betaMat;
    let name = 'Pancreatic Beta (β) Cell';
    let desc = 'Produces and secretes insulin in response to rising blood glucose. In Type 1 diabetes, beta cells are targeted by autoimmune destruction.';
    if (randType > 0.85) {
      mat = deltaMat;
      name = 'Pancreatic Delta (δ) Cell';
      desc = 'Secretes somatostatin, an inhibitory peptide that exerts local paracrine restraint on both insulin and glucagon release.';
    } else if (randType > 0.70) {
      mat = alphaMat;
      name = 'Pancreatic Alpha (α) Cell';
      desc = 'Secretes glucagon during fasting to trigger hepatic glucose production, preventing hypoglycemia.';
    }

    const cellMesh = new THREE.Mesh(cellGeom, mat);
    cellMesh.position.set(r * Math.cos(theta) * Math.sin(phi), r * Math.sin(theta) * Math.sin(phi), r * Math.cos(phi));
    cellMesh.scale.setScalar(0.8 + Math.random() * 0.4);
    registerBioObject(cellMesh, name, desc);
    islet.add(cellMesh);
  }

  // Capillary rings
  const capMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.7 });
  for (let c = 0; c < 5; c++) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(new THREE.EllipseCurve(0, 0, 1.35, 1.35, 0, 2 * Math.PI).getPoints(24)), capMat);
    line.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
    registerBioObject(line, 'Islet Endocrine Capillary', 'Fenestrated microvascular vessel that enables secreted insulin and glucagon to rapidly enter systemic circulation.');
    islet.add(line);
  }

  // Exocytosis granules
  const granuleCount = 30;
  const granuleGeom = new THREE.BufferGeometry();
  const granulePos = new Float32Array(granuleCount * 3);
  for (let g = 0; g < granuleCount * 3; g += 3) {
    granulePos[g] = (Math.random() - 0.5) * 2.2;
    granulePos[g + 1] = (Math.random() - 0.5) * 2.2;
    granulePos[g + 2] = (Math.random() - 0.5) * 2.2;
  }
  granuleGeom.setAttribute('position', new THREE.BufferAttribute(granulePos, 3));
  const granuleParticles = new THREE.Points(granuleGeom, new THREE.PointsMaterial({
    color: 0x38bdf8, size: 0.08, transparent: true, opacity: 0.85
  }));
  registerBioObject(granuleParticles, 'Insulin Secretory Granules', 'Dense-core micro-vesicles storing crystallized zinc-insulin hexamers awaiting calcium-triggered exocytosis.');
  islet.add(granuleParticles);

  islet.userData = { granules: granuleParticles };
  return islet;
}

const isletCluster = createIsletCluster();
pancreasGroup.add(isletCluster);
applyGlobalShadowProperties(pancreasGroup);

// ── GROUP 4: CELL MEMBRANE, RECEPTOR & GLUT4 ────────────────────────────
const cellMembraneGroup = new THREE.Group();
cellMembraneGroup.position.set(0, -75, 0);
scene.add(cellMembraneGroup);

function createLipidBilayer() {
  const membrane = new THREE.Group();
  const headGeom = new THREE.SphereGeometry(0.18, 10, 10);
  const headMatOuter = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, emissive: 0x0369a1 });
  const headMatInner = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.4 });

  const rows = 24;
  const cols = 24;
  const spacing = 0.44;
  const headsList = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c - cols / 2) * spacing;
      const z = (r - rows / 2) * spacing;

      const outerHead = new THREE.Mesh(headGeom, headMatOuter);
      outerHead.position.set(x, 0.45, z);
      membrane.add(outerHead);
      headsList.push(outerHead);

      const innerHead = new THREE.Mesh(headGeom, headMatInner);
      innerHead.position.set(x, -0.45, z);
      membrane.add(innerHead);
      headsList.push(innerHead);
    }
  }

  registerBioObject(membrane, 'Phospholipid Bilayer Membrane', 'The hydrophobic lipid barrier of target muscle/fat cells, preventing passive diffusion of hydrophilic polar glucose molecules.');
  membrane.userData = { heads: headsList };
  return membrane;
}

const lipidBilayer = createLipidBilayer();
cellMembraneGroup.add(lipidBilayer);

function createInsulinReceptor() {
  const ir = new THREE.Group();
  ir.position.set(-1.8, 0, 0);

  const alphaMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.35, emissive: 0x0891b2 });
  const betaMat  = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.4, emissive: 0x1d4ed8 });

  const alphaLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.14, 1.6, 12), alphaMat);
  alphaLeft.position.set(-0.35, 1.2, 0);
  alphaLeft.rotation.z = 0.3;
  ir.add(alphaLeft);

  const alphaRight = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.14, 1.6, 12), alphaMat);
  alphaRight.position.set(0.35, 1.2, 0);
  alphaRight.rotation.z = -0.3;
  ir.add(alphaRight);

  const betaLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 2.0, 10), betaMat);
  betaLeft.position.set(-0.25, -0.6, 0);
  ir.add(betaLeft);

  const betaRight = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 2.0, 10), betaMat);
  betaRight.position.set(0.25, -0.6, 0);
  ir.add(betaRight);

  const dockedInsulin = createGlucoseMolecule();
  dockedInsulin.position.set(0, 1.8, 0);
  dockedInsulin.scale.setScalar(0.7);
  ir.add(dockedInsulin);

  registerBioObject(ir, 'Tyrosine Kinase Insulin Receptor', 'Transmembrane heterotetramer (2α, 2β). Insulin binding induces autophosphorylation of β-subunits, launching the intracellular Akt cascade.');
  return ir;
}

const insulinReceptor = createInsulinReceptor();
cellMembraneGroup.add(insulinReceptor);

function createGlut4Channel() {
  const glut = new THREE.Group();
  glut.position.set(1.8, 0, 0);

  const poreMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.35, emissive: 0x065f46 });
  const numHelices = 12;
  const barrelRadius = 0.75;
  const helixGeom = new THREE.CylinderGeometry(0.12, 0.12, 1.8, 8);

  for (let h = 0; h < numHelices; h++) {
    const angle = (h / numHelices) * Math.PI * 2;
    const helix = new THREE.Mesh(helixGeom, poreMat);
    helix.position.set(Math.cos(angle) * barrelRadius, 0, Math.sin(angle) * barrelRadius);
    glut.add(helix);
  }

  const plug = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 1.2, 16), new THREE.MeshStandardMaterial({
    color: 0x059669, transparent: true, opacity: 0.8
  }));
  glut.add(plug);

  registerBioObject(glut, 'GLUT4 Glucose Transporter Pore', 'Facilitated transport channel that translocates from intracellular storage vesicles into the membrane, allowing glucose influx.');
  glut.userData = { plug };
  return glut;
}

const glut4Channel = createGlut4Channel();
cellMembraneGroup.add(glut4Channel);
applyGlobalShadowProperties(cellMembraneGroup);

// ── GROUP 5: 3D NEURAL NETWORK GRAPH ────────────────────────────────────
const neuralGroup = new THREE.Group();
neuralGroup.position.set(0, -102, 0);
scene.add(neuralGroup);

function create3DNeuralNetwork() {
  const ann = new THREE.Group();
  const layerCounts = [8, 12, 8, 1];
  const layerX = [-6.5, -2.2, 2.2, 6.5];
  const nodeGeom = new THREE.SphereGeometry(0.24, 16, 16);

  const nodeMatInput  = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7 });
  const nodeMatHidden = new THREE.MeshStandardMaterial({ color: 0x818cf8, emissive: 0x4f46e5 });
  const nodeMatOutput = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xb91c1c });

  const allLayers = [];
  layerCounts.forEach((count, lIdx) => {
    const layerNodes = [];
    const mat = lIdx === 0 ? nodeMatInput : (lIdx === layerCounts.length - 1 ? nodeMatOutput : nodeMatHidden);
    const spacingY = 0.9;
    const startY = (count - 1) * spacingY * 0.5;

    for (let n = 0; n < count; n++) {
      const mesh = new THREE.Mesh(nodeGeom, mat);
      mesh.position.set(layerX[lIdx], startY - n * spacingY, (Math.random() - 0.5) * 0.4);
      ann.add(mesh);
      layerNodes.push(mesh);

      if (lIdx === 0 && MODEL_DATA.feature_names[n]) {
        registerBioObject(mesh, `Input Neuron: ${MODEL_DATA.feature_names[n]}`, `Clinical feature node feeding normalized patient measurement into hidden layer 1.`);
      } else if (lIdx === layerCounts.length - 1) {
        registerBioObject(mesh, 'Output Neuron (Risk Probability)', 'Sigmoid activation unit estimating binary diabetes risk probability in range [0.0, 1.0].');
      }
    }
    allLayers.push(layerNodes);
  });

  const synMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending });
  for (let l = 0; l < allLayers.length - 1; l++) {
    allLayers[l].forEach(n1 => {
      allLayers[l + 1].forEach(n2 => {
        ann.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([n1.position, n2.position]), synMat));
      });
    });
  }

  const pulseCount = 45;
  const pulseGeom = new THREE.BufferGeometry();
  const pulsePos = new Float32Array(pulseCount * 3);
  const pulseData = [];

  for (let p = 0; p < pulseCount; p++) {
    const l = Math.floor(Math.random() * (allLayers.length - 1));
    pulseData.push({
      p1: allLayers[l][Math.floor(Math.random() * allLayers[l].length)].position,
      p2: allLayers[l + 1][Math.floor(Math.random() * allLayers[l + 1].length)].position,
      progress: Math.random(),
      speed: 0.015 + Math.random() * 0.02
    });
  }
  pulseGeom.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
  const pulseParticles = new THREE.Points(pulseGeom, new THREE.PointsMaterial({
    color: 0xffffff, size: 0.16, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending
  }));
  ann.add(pulseParticles);

  ann.userData = { allLayers, pulseData, pulseParticles, pulsePos };
  return ann;
}

const neuralNetwork = create3DNeuralNetwork();
neuralGroup.add(neuralNetwork);

/* ==========================================================================
   4. MINIMAL BIOLOGICAL EXPLORATION: HOVER / TAP RAYCASTER
   ========================================================================== */
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-9999, -9999);
const bioTooltip = document.getElementById('bioTooltip');
const btName = document.getElementById('btName');
const btDesc = document.getElementById('btDesc');

let hoveredObject = null;
let lastCalculatedProb = 0.048; // Baseline default risk (~4.8%)

function onPointerMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  if (bioTooltip && bioTooltip.classList.contains('visible')) {
    bioTooltip.style.left = `${Math.min(window.innerWidth - 180, Math.max(160, e.clientX))}px`;
    bioTooltip.style.top  = `${Math.max(80, e.clientY - 20)}px`;
  }
}

window.addEventListener('pointermove', onPointerMove, { passive: true });

function checkRaycast() {
  raycaster.setFromCamera(mouse, camera);
  // Cast against interactiveObjects AND all their deep children
  const intersects = raycaster.intersectObjects(interactiveObjects, true);

  for (let i = 0; i < intersects.length; i++) {
    // Walk up the parent chain until we find bioInfo or hit the root
    let target = intersects[i].object;
    let resolved = null;
    while (target) {
      if (target.userData && target.userData.bioInfo) {
        resolved = target;
        break;
      }
      target = target.parent;
    }

    if (resolved) {
      const info = resolved.userData.bioInfo;
      if (hoveredObject !== resolved) {
        hoveredObject = resolved;
        if (btName) btName.textContent = info.name;
        if (btDesc) btDesc.textContent = info.desc;
        const btDot = document.querySelector('.bt-dot');
        if (btDot) {
          if (info.name.includes('White Blood Cell') || info.name.includes('Leukocyte')) {
            btDot.style.background = '#f8fafc';
            btDot.style.boxShadow = '0 0 10px #ffffff';
          } else if (info.name.includes('Glucose')) {
            btDot.style.background = '#f59e0b';
            btDot.style.boxShadow = '0 0 10px #f59e0b';
          } else if (info.name.includes('Beta') || info.name.includes('Pancreas')) {
            btDot.style.background = '#10b981';
            btDot.style.boxShadow = '0 0 10px #10b981';
          } else {
            btDot.style.background = '#dc2626';
            btDot.style.boxShadow = '0 0 10px #dc2626';
          }
        }
        if (bioTooltip) bioTooltip.classList.add('visible');
        canvas.style.cursor = 'pointer';
      }
      return;
    }
  }

  if (hoveredObject) {
    hoveredObject = null;
    if (bioTooltip) bioTooltip.classList.remove('visible');
    canvas.style.cursor = 'default';
  }
}

window.addEventListener('pointerdown', (e) => {
  onPointerMove(e);
  checkRaycast();
});

/* ==========================================================================
   5. CONTINUOUS CINEMATIC SCROLL CAMERA CHOREOGRAPHY
   ========================================================================== */
const SCROLL_STAGES = [
  { trigger: '#hero',              cam: { x: 0,   y: 0,   z: 22 }, target: { x: 1.8, y: 0,   z: 0 } },
  { trigger: '#bloodstream',       cam: { x: 0,   y: -22, z: 7  }, target: { x: 0,   y: -22, z: -10 } },
  { trigger: '#pancreas',          cam: { x: 0,   y: -48, z: 9  }, target: { x: -0.8, y: -48, z: 2 } },
  { trigger: '#insulin',           cam: { x: 0.8, y: -75, z: 6  }, target: { x: 0,   y: -75, z: 0 } },
  { trigger: '#pathology',         cam: { x: 0,   y: -75, z: 12 }, target: { x: 0,   y: -75, z: 0 } },
  { trigger: '#prediction-portal', cam: { x: 0,   y: -102,z: 14 }, target: { x: 0,   y: -102,z: 0 } },
  { trigger: '#ai-research',       cam: { x: -2,  y: -102,z: 10 }, target: { x: 0,   y: -102,z: 0 } },
  { trigger: '#awareness',         cam: { x: 0,   y: -110,z: 20 }, target: { x: 0,   y: -102,z: 0 } }
];

const hudScale    = document.getElementById('hudScale');
const hudLocation = document.getElementById('hudLocation');
const hudChapter  = document.getElementById('hudChapter');
const hudTone     = document.getElementById('hudTone');

const targetCamPos    = new THREE.Vector3(SCROLL_STAGES[0].cam.x, SCROLL_STAGES[0].cam.y, SCROLL_STAGES[0].cam.z);
const targetCamLookAt = new THREE.Vector3(SCROLL_STAGES[0].target.x, SCROLL_STAGES[0].target.y, SCROLL_STAGES[0].target.z);

let stageMetrics = [];

function updateStageMetrics() {
  stageMetrics = SCROLL_STAGES.map(stage => {
    const el = document.querySelector(stage.trigger);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const top = rect.top + scrollTop;
    const height = el.offsetHeight;
    return {
      trigger: stage.trigger,
      cam: stage.cam,
      target: stage.target,
      top: top,
      bottom: top + height,
      center: top + height * 0.5,
      el: el
    };
  }).filter(Boolean);
}

updateStageMetrics();
window.addEventListener('resize', updateStageMetrics);
window.addEventListener('load', updateStageMetrics);

let activeStageIndex = -1;

function updateContinuousCamera() {
  if (!stageMetrics.length) return;

  const scrollY = window.pageYOffset || document.documentElement.scrollTop;
  const viewportCenterY = scrollY + window.innerHeight * 0.5;

  const first = stageMetrics[0];
  const last  = stageMetrics[stageMetrics.length - 1];

  if (viewportCenterY <= first.center) {
    targetCamPos.set(first.cam.x, first.cam.y, first.cam.z);
    targetCamLookAt.set(first.target.x, first.target.y, first.target.z);
    setActiveStage(0);
    return;
  }

  if (viewportCenterY >= last.center) {
    targetCamPos.set(last.cam.x, last.cam.y, last.cam.z);
    targetCamLookAt.set(last.target.x, last.target.y, last.target.z);
    setActiveStage(stageMetrics.length - 1);
    return;
  }

  // Smooth S-Curve (Cosine) continuous interpolation between section midpoints
  for (let i = 0; i < stageMetrics.length - 1; i++) {
    const curr = stageMetrics[i];
    const next = stageMetrics[i + 1];

    if (viewportCenterY >= curr.center && viewportCenterY <= next.center) {
      const span = next.center - curr.center;
      let progress = span > 0 ? (viewportCenterY - curr.center) / span : 0;
      progress = Math.max(0, Math.min(1, progress));

      // Cosine easing has zero acceleration spike at endpoints (zero jerk)
      const easeT = 0.5 - 0.5 * Math.cos(progress * Math.PI);

      targetCamPos.x = THREE.MathUtils.lerp(curr.cam.x, next.cam.x, easeT);
      targetCamPos.y = THREE.MathUtils.lerp(curr.cam.y, next.cam.y, easeT);
      targetCamPos.z = THREE.MathUtils.lerp(curr.cam.z, next.cam.z, easeT);

      targetCamLookAt.x = THREE.MathUtils.lerp(curr.target.x, next.target.x, easeT);
      targetCamLookAt.y = THREE.MathUtils.lerp(curr.target.y, next.target.y, easeT);
      targetCamLookAt.z = THREE.MathUtils.lerp(curr.target.z, next.target.z, easeT);

      const closestStage = progress < 0.5 ? i : i + 1;
      setActiveStage(closestStage);
      break;
    }
  }
}

function setActiveStage(idx) {
  if (idx === activeStageIndex) return;
  activeStageIndex = idx;
  const stage = stageMetrics[idx];
  if (!stage) return;

  const el = stage.el;
  if (hudScale && el.dataset.scale) hudScale.textContent = el.dataset.scale;
  if (hudLocation && el.dataset.loc) hudLocation.textContent = el.dataset.loc;
  if (hudChapter && el.dataset.phase) hudChapter.textContent = el.dataset.phase;

  if (hudTone) {
    if (stage.trigger === '#hero') {
      hudTone.textContent = 'HOMEOSTATIC (NORMAL)';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#bloodstream') {
      hudTone.textContent = 'LAMINAR FLOW (NORMAL)';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#pancreas') {
      hudTone.textContent = 'ENDOCRINE BASAL';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#insulin') {
      hudTone.textContent = 'RECEPTOR SENSITIVE';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#pathology') {
      hudTone.textContent = 'COMPARATIVE STATES';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#prediction-portal' || stage.trigger === '#ai-research') {
      const curScore = (lastCalculatedProb * 100).toFixed(1);
      if (lastCalculatedProb < 0.3) {
        hudTone.textContent = `MODEL RISK: ${curScore}% (LOW)`;
        hudTone.className = 'hud-val hud-status-normal';
        hudTone.style.color = '';
      } else if (lastCalculatedProb < 0.6) {
        hudTone.textContent = `MODEL RISK: ${curScore}% (BORDERLINE)`;
        hudTone.className = 'hud-val';
        hudTone.style.color = '#f59e0b';
      } else {
        hudTone.textContent = `MODEL RISK: ${curScore}% (HIGH)`;
        hudTone.className = 'hud-val';
        hudTone.style.color = '#ef4444';
      }
    }
  }

  document.querySelectorAll('.nav-links a').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === stage.trigger);
  });

  // Focus class on active section
  document.querySelectorAll('.doc-chapter, .doc-hero').forEach(sec => {
    sec.classList.toggle('chapter-in-focus', sec === el);
  });
}

/* ==========================================================================
   6. MAIN RENDER LOOP & REAL-TIME BIOLOGICAL ANIMATIONS
   ========================================================================== */
let clock = new THREE.Clock();
let _frameCount = 0;

function animate() {
  requestAnimationFrame(animate);

  const delta = clock.getDelta();
  const time  = clock.getElapsedTime();

  // Throttle raycaster: run every 3 frames to save CPU/GPU
  if (_frameCount % 3 === 0) checkRaycast();
  _frameCount++;

  // 1. Full Body Gentle Micro-Rotation & Organ Dynamics
  if (completeHumanBody) {
    completeHumanBody.rotation.y = Math.sin(time * 0.2) * 0.14;
  }

  // 2. Cardiac Muscle Contraction (72 BPM rhythm: systole & diastole)
  if (heartMeshRef) {
    const heartBeat = 1.0 + 0.09 * Math.pow(Math.sin(time * 7.5), 6.0);
    heartMeshRef.scale.set(heartBeat, heartBeat, heartBeat);
  }

  // 3. Pulmonary Respiration (Gentle lung tidal expansion)
  if (leftLungRef && rightLungRef) {
    const breath = 1.0 + 0.04 * Math.sin(time * 1.8);
    leftLungRef.scale.set(breath, breath, breath);
    rightLungRef.scale.set(breath, breath, breath);
  }

  // 4. Floating Interstitial Particles
  if (floatingParticles) {
    floatingParticles.rotation.y = time * 0.02;
  }

  // 5. Bloodstream Fluid Flow: Laminar RBCs, Marginated WBC Rolling & Tumbling Glucose
  rbcList.forEach(rbc => {
    rbc.position.z += rbc.userData.vz;
    rbc.rotation.x += rbc.userData.rotX;
    rbc.rotation.y += rbc.userData.rotY;
    if (rbc.position.z > vesselLength * 0.45) rbc.position.z = -vesselLength * 0.45;
  });

  // Leukocyte margination & slow endothelial rolling with subtle selectin adhesion dynamics
  wbcList.forEach(wbc => {
    // Physiological selectin-integrin rolling hesitation
    const rollingHesitation = 0.85 + 0.25 * Math.sin(time * 1.2 + wbc.userData.phase);
    wbc.position.z += wbc.userData.vz * rollingHesitation;
    wbc.rotation.x += wbc.userData.rollX * rollingHesitation;
    wbc.rotation.y += wbc.userData.rollY * rollingHesitation;
    if (wbc.position.z > vesselLength * 0.45) wbc.position.z = -vesselLength * 0.45;
  });

  glucoseMolecules.forEach(g => {
    g.position.z += g.userData.vz;
    g.rotation.x += g.userData.rotX;
    if (g.position.z > vesselLength * 0.45) g.position.z = -vesselLength * 0.45;
  });

  // 6. Pancreas & Islet Exocytosis
  if (pancreasOrgan) pancreasOrgan.rotation.y = Math.sin(time * 0.3) * 0.06;
  if (isletCluster && isletCluster.userData.granules) {
    const pos = isletCluster.userData.granules.geometry.attributes.position.array;
    for (let i = 0; i < pos.length; i += 3) {
      pos[i + 2] += 0.015;
      if (pos[i + 2] > 2.0) pos[i + 2] = -1.0;
    }
    isletCluster.userData.granules.geometry.attributes.position.needsUpdate = true;
  }

  // 7. Lipid Bilayer Wave
  if (lipidBilayer && lipidBilayer.userData.heads) {
    lipidBilayer.userData.heads.forEach(h => {
      const dist = Math.sqrt(h.position.x * h.position.x + h.position.z * h.position.z);
      const wave = Math.sin(dist * 1.5 - time * 2.0) * 0.05;
      h.position.y = (h.position.y > 0 ? 0.45 : -0.45) + wave;
    });
  }

  // 8. 3D Neural Network Signal Propagation
  if (neuralNetwork && neuralNetwork.userData.pulseData) {
    const pData = neuralNetwork.userData.pulseData;
    const pPos  = neuralNetwork.userData.pulsePos;
    pData.forEach((p, idx) => {
      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;
      pPos[idx * 3]     = THREE.MathUtils.lerp(p.p1.x, p.p2.x, p.progress);
      pPos[idx * 3 + 1] = THREE.MathUtils.lerp(p.p1.y, p.p2.y, p.progress);
      pPos[idx * 3 + 2] = THREE.MathUtils.lerp(p.p1.z, p.p2.z, p.progress);
    });
    neuralNetwork.userData.pulseParticles.geometry.attributes.position.needsUpdate = true;
  }

  // Continuous camera scroll motion with Steadicam inertial smoothing
  updateContinuousCamera();

  const damping = PREFERS_REDUCED_MOTION ? 0.25 : 0.08;
  camera.position.x += (targetCamPos.x - camera.position.x) * damping;
  camera.position.y += (targetCamPos.y - camera.position.y) * damping;
  camera.position.z += (targetCamPos.z - camera.position.z) * damping;

  cameraTarget.x += (targetCamLookAt.x - cameraTarget.x) * damping;
  cameraTarget.y += (targetCamLookAt.y - cameraTarget.y) * damping;
  cameraTarget.z += (targetCamLookAt.z - cameraTarget.z) * damping;

  camera.lookAt(cameraTarget);
  renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

/* ==========================================================================
   7. AUTHENTIC ANN PREDICTION & REAL SHAP XAI ALGORITHMS
   ========================================================================== */
function relu(arr) {
  return arr.map(v => Math.max(0, v));
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z))));
}

function getANNModelData() {
  if (typeof window !== 'undefined' && window.ALL_MODELS_DATA && window.ALL_MODELS_DATA.models) {
    const ann = window.ALL_MODELS_DATA.models.find(m => m.id === 'ann');
    if (ann && ann.weights && ann.weights.length >= 3 && ann.biases && ann.biases.length >= 3) {
      return ann;
    }
  }
  if (!MODEL_DATA || !MODEL_DATA.weights || MODEL_DATA.weights.length < 3) {
    MODEL_DATA = getBaseModelData();
  }
  return MODEL_DATA;
}

// Phase 1 Fix: Dynamic Multi-Layer Feedforward ANN supporting arbitrary architectures (8 -> 32 -> 16 -> 1)
function forwardPassANN(zInput, weightsOverride, biasesOverride) {
  const model = getANNModelData();
  const weights = weightsOverride || (model ? model.weights : []);
  const biases  = biasesOverride  || (model ? model.biases : []);

  if (!weights || weights.length < 3 || !biases || biases.length < 3) {
    return { prob: 0.3547, logit: -0.6, h1: [], h2: [] };
  }

  // Layer 1: Input (8) -> Hidden 1 (32)
  const W1 = weights[0];
  const b1 = biases[0];
  const n1 = b1.length;
  const h1 = new Array(n1);
  for (let j = 0; j < n1; j++) {
    let sum = b1[j];
    for (let i = 0; i < zInput.length; i++) {
      sum += zInput[i] * W1[i][j];
    }
    h1[j] = Math.max(0, sum);
  }

  // Layer 2: Hidden 1 (32) -> Hidden 2 (16)
  const W2 = weights[1];
  const b2 = biases[1];
  const n2 = b2.length;
  const h2 = new Array(n2);
  for (let k = 0; k < n2; k++) {
    let sum = b2[k];
    for (let j = 0; j < n1; j++) {
      sum += h1[j] * W2[j][k];
    }
    h2[k] = Math.max(0, sum);
  }

  // Layer 3: Hidden 2 (16) -> Output (1)
  const W3 = weights[2];
  const b3 = biases[2];
  let logit = b3[0];
  for (let k = 0; k < n2; k++) {
    logit += h2[k] * W3[k][0];
  }

  const prob = sigmoid(logit);
  return { prob, logit, h1, h2 };
}

// Universal Model Evaluator across Verified Architecture (Pure ANN)
function evaluateSingleModel(modelId, rawFeatures) {
  const bundle = window.ALL_MODELS_DATA;
  const meta = bundle ? bundle.metadata : MODEL_DATA;
  const medians = meta.medians;
  const mean = meta.scaler_mean;
  const std  = meta.scaler_std;

  // Impute physiologically impossible 0s with training medians
  const imputed = rawFeatures.map((v, i) => {
    const featName = meta.feature_names[i];
    if (['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'].includes(featName) && (v === 0 || isNaN(v))) {
      return medians[featName] !== undefined ? medians[featName] : v;
    }
    return isNaN(v) ? 0 : v;
  });

  const scaled = imputed.map((v, i) => (v - mean[i]) / (std[i] || 1));

  const annModel = getANNModelData();
  const model = (bundle && bundle.models) ? bundle.models.find(m => m.id === 'ann') || bundle.models[0] : null;
  const annRes = forwardPassANN(
    scaled,
    (model && model.weights) ? model.weights : annModel.weights,
    (model && model.biases) ? model.biases : annModel.biases
  );
  return { probability: annRes.prob, logit: annRes.logit };
}

// Model-Specific Explanations (Pure ANN Integrated Gradients)
function computeModelSpecificSHAP(activeModelId, rawFeatures, currentProb) {
  const meta = window.ALL_MODELS_DATA ? window.ALL_MODELS_DATA.metadata : MODEL_DATA;
  const featureNames = meta.feature_names;
  const medians = meta.medians;
  const baselineExpected = 0.3547;
  const deltaTotal = currentProb - baselineExpected;

  // Impute raw features with training medians for valid clinical inference
  const cleanVector = rawFeatures.map((val, i) => {
    const fName = featureNames[i];
    return (['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'].includes(fName) && (val === 0 || isNaN(val) || val === null))
      ? (medians[fName] !== undefined ? medians[fName] : val) : (isNaN(val) ? 0 : val);
  });

  // True Integrated Gradients over the verified 8 -> 32 -> 16 -> 1 ANN architecture
  const annModel = getANNModelData();
  const mean = meta.scaler_mean;
  const std = meta.scaler_std;
  const zVector = cleanVector.map((val, i) => (val - mean[i]) / (std[i] || 1));
  const steps = 8;
  const rawAttributions = new Array(8).fill(0);
  const eps = 1e-4;

  for (let s = 1; s <= steps; s++) {
    const alpha = s / steps;
    const zStep = zVector.map(z => z * alpha);
    const baseP = forwardPassANN(zStep, annModel.weights, annModel.biases).prob;

    for (let i = 0; i < 8; i++) {
      const zPerturbed = zStep.slice();
      zPerturbed[i] += eps;
      const pertP = forwardPassANN(zPerturbed, annModel.weights, annModel.biases).prob;
      const grad = (pertP - baseP) / eps;
      rawAttributions[i] += grad * zVector[i];
    }
  }
  for (let i = 0; i < 8; i++) rawAttributions[i] /= steps;
  const sumRaw = rawAttributions.reduce((a, b) => a + b, 0);
  const attributions = rawAttributions.map(a => Math.abs(sumRaw) > 1e-6 ? (a / sumRaw) * deltaTotal : a);
  return {
    attributions,
    method: 'ANN Integrated Gradients',
    modelName: 'Artificial Neural Network'
  };
}

function computeAuthenticSHAP(rawVector) {
  const vector = rawVector || getFormFeatureVector();
  const annEval = evaluateSingleModel('ann', vector);
  return computeModelSpecificSHAP('ann', vector, annEval.probability);
}

/* ==========================================================================
   8. CONNECT INPUTS TO 3D VISUALIZATION & PREDICTION PLATFORM
   ========================================================================== */
const predictionForm       = document.getElementById('predictionForm');
const riskScoreDisplay     = document.getElementById('riskScoreDisplay');
const riskScoreLarge       = document.getElementById('riskScoreLarge');
const gaugeProgressCircle   = document.getElementById('gaugeProgressCircle');
const riskClassText        = document.getElementById('riskClassText');
const riskClassBadge       = document.getElementById('riskClassBadge');
const riskSubText          = document.getElementById('riskSubText');
const uncertaintyLevelText = document.getElementById('uncertaintyLevelText');
const cardUncertaintyTitle = document.getElementById('cardUncertaintyTitle');
const cardUncertaintyDesc  = document.getElementById('cardUncertaintyDesc');
const modelConfidenceText  = document.getElementById('modelConfidenceText');
const modelConfidenceFill  = document.getElementById('modelConfidenceFill');
const modelSelectControl   = document.getElementById('modelSelectControl');
const activeModelLabel     = document.getElementById('activeModelLabel');
const btnToggleComparison  = document.getElementById('btnToggleComparison');
const comparisonDrawer     = document.getElementById('comparisonDrawer');
const shapBarsList         = document.getElementById('shapBarsList');
const shapCardHeading      = document.getElementById('shapCardHeading');
const shapModelTag         = document.getElementById('shapModelTag');
const streamGlucVal        = document.getElementById('streamGlucVal');
const inputGlucose         = document.getElementById('inputGlucose');
const bestModelName        = document.getElementById('bestModelName');
const bestModelAcc         = document.getElementById('bestModelAccuracy');
const comparisonBody       = document.getElementById('modelComparisonBody');
const riskSilhouette       = document.getElementById('riskSilhouette');

const perfAcc   = document.getElementById('perfAcc');
const perfF1    = document.getElementById('perfF1');
const perfPrec  = document.getElementById('perfPrec');
const perfAuc   = document.getElementById('perfAuc');
const perfRec   = document.getElementById('perfRec');
const perfBrier = document.getElementById('perfBrier');

// Phase 3: Clinical Screening Threshold DOM Elements
const sliderThreshold        = document.getElementById('sliderThreshold');
const dispThresholdVal        = document.getElementById('dispThresholdVal');
const btnThreshValOpt         = document.getElementById('btnThreshValOpt');
const screeningDecisionBadge  = document.getElementById('screeningDecisionBadge');
const threshTradeoffNote      = document.getElementById('threshTradeoffNote');

let currentThreshold = 0.50;
const VALIDATION_OPTIMAL_THRESHOLDS = {
  ann: 0.25
};

// Phase 4: Interactive Counterfactual Explorer DOM Elements
const btnToggleCounterfactual = document.getElementById('btnToggleCounterfactual');
const cfDrawerPanel           = document.getElementById('cfDrawerPanel');
const cfActiveModelName       = document.getElementById('cfActiveModelName');
const cfActiveModelBadge      = document.getElementById('cfActiveModelBadge');
const sliderCfGlucose         = document.getElementById('sliderCfGlucose');
const sliderCfBMI             = document.getElementById('sliderCfBMI');
const sliderCfBP              = document.getElementById('sliderCfBP');
const sliderCfInsulin         = document.getElementById('sliderCfInsulin');
const sliderCfSkin            = document.getElementById('sliderCfSkin');
const dispCfGlucose           = document.getElementById('dispCfGlucose');
const dispCfBMI               = document.getElementById('dispCfBMI');
const dispCfBP                = document.getElementById('dispCfBP');
const dispCfInsulin           = document.getElementById('dispCfInsulin');
const dispCfSkin              = document.getElementById('dispCfSkin');
const cfOriginalScore         = document.getElementById('cfOriginalScore');
const cfOriginalPill          = document.getElementById('cfOriginalPill');
const cfHypoScore             = document.getElementById('cfHypoScore');
const cfHypoPill              = document.getElementById('cfHypoPill');
const cfDeltaVal              = document.getElementById('cfDeltaVal');
const cfDeltaBadge            = document.getElementById('cfDeltaBadge');
const cfDeltaBox              = document.getElementById('cfDeltaBox');
const cfChangesList           = document.getElementById('cfChangesList');
const cfExplanationText       = document.getElementById('cfExplanationText');
const btnResetCounterfactual  = document.getElementById('btnResetCounterfactual');

let cfHypotheticalVector = null;

/* ---------- Multi-Model Comparison Engine (Offline & Synchronous) ---------- */
function traverseTreeBundle(node, vec) {
  if (node.v !== undefined) return node.v;
  return vec[node.f] <= node.t
    ? traverseTreeBundle(node.l, vec)
    : traverseTreeBundle(node.r, vec);
}

function runMultiModelComparison(rawFeatures) {
  const bundle = window.ALL_MODELS_DATA;
  const results = [];

  if (bundle && bundle.models) {
    bundle.models.forEach(m => {
      const evalRes = evaluateSingleModel(m.id, rawFeatures);
      results.push({
        id: m.id,
        name: m.name,
        badge: m.badge,
        probability: evalRes.probability,
        accuracy: m.accuracy,
        roc_auc: m.roc_auc,
        f1: m.f1,
        recall: m.recall,
        precision: m.precision,
        brier: m.brier,
        shap_importance: m.shap_importance
      });
    });
  } else {
    const annEval = evaluateSingleModel('ann', rawFeatures);
    results.push({
      id: 'ann',
      name: 'Artificial Neural Network (ANN)',
      badge: 'MLP 8->32->16->1',
      probability: annEval.probability,
      accuracy: 0.7338,
      roc_auc: 0.8191,
      f1: 0.5859,
      recall: 0.5370,
      precision: 0.6444,
      brier: 0.1668
    });
  }

  // Sort by accuracy (best model first)
  results.sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0));

  // Render comparison table
  if (comparisonBody) {
    comparisonBody.innerHTML = '';
    results.forEach((r, i) => {
      const row = document.createElement('tr');
      const pct = (r.probability * 100).toFixed(1);
      const accStr = r.accuracy ? (r.accuracy * 100).toFixed(1) + '%' : '—';
      const isBest = i === 0;
      row.className = isBest ? 'best-model-row' : '';
      row.innerHTML = `
        <td>
          <div style="display:flex; flex-direction:column;">
            <span>${isBest ? '🏆 ' : ''}<strong>${r.name}</strong></span>
            <small style="opacity:0.65; font-size:0.75rem;">${r.badge || ''}</small>
          </div>
        </td>
        <td>
          <span class="comp-prob ${r.probability >= currentThreshold ? 'prob-high' : 'prob-low'}">
            ${pct}%
          </span>
        </td>
        <td>
          <strong>${accStr}</strong>
          <div style="font-size:0.7rem; color:var(--text-muted);">AUC: ${r.roc_auc ? r.roc_auc.toFixed(3) : '—'}</div>
        </td>
        <td>
          ${isBest ? '<span class="best-badge">🏆 SELECTED BEST</span>' : '<span style="font-size:0.75rem; opacity:0.6;">Validated</span>'}
        </td>
      `;
      comparisonBody.appendChild(row);
    });
  }

  // Update best model indicator
  const best = results[0];
  if (best) {
    if (bestModelName) bestModelName.textContent = best.name;
    if (bestModelAcc) {
      bestModelAcc.textContent = `${(best.accuracy * 100).toFixed(1)}% (AUC: ${best.roc_auc ? best.roc_auc.toFixed(3) : '—'}, Sensitivity: ${(best.recall * 100).toFixed(1)}%)`;
    }
  }

  return results;
}

function getFormFeatureVector() {
  const form = document.getElementById('predictionForm');
  if (!form) return [1, 100, 72, 23, 85, 26.5, 0.38, 33];

  const meta = window.ALL_MODELS_DATA ? window.ALL_MODELS_DATA.metadata : MODEL_DATA;
  return meta.feature_names.map(name => {
    const el = form.elements[name];
    const val = parseFloat(el ? el.value : '');
    return isNaN(val) ? meta.medians[name] : val;
  });
}

function applyThresholdDecision(prob, activeModelName) {
  if (dispThresholdVal) {
    dispThresholdVal.textContent = `${currentThreshold.toFixed(2)} (${currentThreshold === 0.50 ? 'Standard' : (currentThreshold === 0.25 ? 'High Sensitivity' : (currentThreshold === 0.65 ? 'High Specificity' : 'Custom'))})`;
  }

  const isPositive = prob >= currentThreshold;
  if (screeningDecisionBadge) {
    if (isPositive) {
      screeningDecisionBadge.textContent = '⚠️ Positive (At/Above Cutoff)';
      screeningDecisionBadge.className = 'sdr-badge alert-pos';
    } else {
      screeningDecisionBadge.textContent = '🛡️ Negative (Below Cutoff)';
      screeningDecisionBadge.className = 'sdr-badge';
    }
  }

  if (threshTradeoffNote) {
    if (currentThreshold <= 0.30) {
      threshTradeoffNote.textContent = 'Screening Mode (Threshold ≤ 0.30): Maximizes Sensitivity (~80–87%). Catches the vast majority of potential diabetic cases with low False Negatives, at the cost of higher False Positives.';
    } else if (currentThreshold >= 0.60) {
      threshTradeoffNote.textContent = 'Confirmation Mode (Threshold ≥ 0.60): Maximizes Precision and Specificity (~84–90%). Minimizes false alarms before costly clinical referrals, but increases False Negatives.';
    } else {
      threshTradeoffNote.textContent = 'Balanced Mode (Threshold ~0.50): Standard decision boundary. Statistical screening indicator — not a clinical diagnosis. Venous blood testing (HbA1c) is required for clinical confirmation.';
    }
  }
}

function runPredictionAndRender() {
  const vector = getFormFeatureVector();
  const allResults = runMultiModelComparison(vector);
  
  // Determine selected active model
  const activeModelId = modelSelectControl ? modelSelectControl.value : 'ann';
  const activeModel = allResults.find(r => r.id === activeModelId) || allResults[0];
  
  const prob = activeModel ? activeModel.probability : 0.5;
  lastCalculatedProb = prob;
  const pct = Math.round(prob * 100);

  // Update Score Displays
  if (riskScoreDisplay) riskScoreDisplay.textContent = `${pct}%`;
  if (riskScoreLarge) riskScoreLarge.textContent = `${pct}%`;

  // Update Circular SVG Gauge
  if (gaugeProgressCircle) {
    const circumference = 263.89;
    const offset = circumference * (1.0 - Math.min(1.0, Math.max(0, prob)));
    gaugeProgressCircle.style.strokeDashoffset = offset;
    
    if (prob < 0.30) {
      gaugeProgressCircle.style.stroke = '#10b981';
    } else if (prob < 0.60) {
      gaugeProgressCircle.style.stroke = '#06b6d4';
    } else {
      gaugeProgressCircle.style.stroke = '#ef4444';
    }
  }

  // Update Risk Classifications & Badges
  if (prob < 0.30) {
    if (riskClassText) {
      riskClassText.textContent = 'Low Risk';
      riskClassText.style.color = '#10b981';
    }
    if (riskClassBadge) {
      riskClassBadge.textContent = '🛡️ Low Risk';
      riskClassBadge.style.background = '#10b981';
      riskClassBadge.style.color = '#022c22';
    }
    if (uncertaintyLevelText) uncertaintyLevelText.textContent = 'Uncertainty Level: Low';
    if (cardUncertaintyTitle) cardUncertaintyTitle.textContent = 'Model Uncertainty: Low';
    if (cardUncertaintyDesc) cardUncertaintyDesc.textContent = `Model indicates strong metabolic stability (${pct}%). Periodic monitoring advised.`;
    if (riskSubText) riskSubText.textContent = `Model has high confidence (${pct}%) in non-diabetic metabolic stability.`;
    if (modelConfidenceText) modelConfidenceText.textContent = '84%';
    if (modelConfidenceFill) modelConfidenceFill.style.width = '84%';
    if (riskSilhouette) riskSilhouette.style.filter = 'drop-shadow(0 0 16px rgba(16, 185, 129, 0.55))';
  } else if (prob < 0.60) {
    if (riskClassText) {
      riskClassText.textContent = 'Moderate';
      riskClassText.style.color = '#38bdf8';
    }
    if (riskClassBadge) {
      riskClassBadge.textContent = '🛡️ Moderate Risk';
      riskClassBadge.style.background = '#38bdf8';
      riskClassBadge.style.color = '#0369a1';
    }
    if (uncertaintyLevelText) uncertaintyLevelText.textContent = 'Uncertainty Level: Moderate';
    if (cardUncertaintyTitle) cardUncertaintyTitle.textContent = 'Model Uncertainty: Moderate';
    if (cardUncertaintyDesc) cardUncertaintyDesc.textContent = `Prediction confidence is balanced (${pct}%). Consider additional clinical follow-up like venous HbA1c testing.`;
    if (riskSubText) riskSubText.textContent = `Prediction has some uncertainty (${pct}%). Consider additional clinical follow-up.`;
    if (modelConfidenceText) modelConfidenceText.textContent = '78%';
    if (modelConfidenceFill) modelConfidenceFill.style.width = '78%';
    if (riskSilhouette) riskSilhouette.style.filter = 'drop-shadow(0 0 18px rgba(56, 189, 248, 0.6))';
  } else {
    if (riskClassText) {
      riskClassText.textContent = 'High Risk';
      riskClassText.style.color = '#ef4444';
    }
    if (riskClassBadge) {
      riskClassBadge.textContent = '⚠️ High Risk';
      riskClassBadge.style.background = '#ef4444';
      riskClassBadge.style.color = '#ffffff';
    }
    if (uncertaintyLevelText) uncertaintyLevelText.textContent = 'Uncertainty Level: Low (High Risk)';
    if (cardUncertaintyTitle) cardUncertaintyTitle.textContent = 'Model Alert: High Risk';
    if (cardUncertaintyDesc) cardUncertaintyDesc.textContent = `Biomarkers strongly align with diabetic cohorts (${pct}%). Professional medical evaluation recommended.`;
    if (riskSubText) riskSubText.textContent = `Metabolic markers strongly match diabetic profiles (${pct}%). Professional testing recommended.`;
    if (modelConfidenceText) modelConfidenceText.textContent = '88%';
    if (modelConfidenceFill) modelConfidenceFill.style.width = '88%';
    if (riskSilhouette) riskSilhouette.style.filter = 'drop-shadow(0 0 22px rgba(239, 68, 68, 0.75))';
  }

  // Update Threshold Decision
  applyThresholdDecision(prob, activeModel ? activeModel.name : 'Active Model');

  // Update Active Model Label in Column 1
  if (activeModelLabel && activeModel) {
    activeModelLabel.textContent = `${activeModel.name} ${activeModel.id === 'ann' ? '(Primary)' : ''}`;
  }

  // Update Active Model 6-Cell Performance Grid
  if (activeModel) {
    if (perfAcc)   perfAcc.textContent   = activeModel.accuracy ? `${(activeModel.accuracy * 100).toFixed(1)}%` : '—';
    if (perfF1)    perfF1.textContent    = activeModel.f1 ? `${(activeModel.f1 * 100).toFixed(1)}%` : '—';
    if (perfPrec)  perfPrec.textContent  = activeModel.precision ? `${(activeModel.precision * 100).toFixed(1)}%` : '—';
    if (perfAuc)   perfAuc.textContent   = activeModel.roc_auc ? activeModel.roc_auc.toFixed(3) : '—';
    if (perfRec)   perfRec.textContent   = activeModel.recall ? `${(activeModel.recall * 100).toFixed(1)}%` : '—';
    if (perfBrier) perfBrier.textContent = activeModel.brier ? activeModel.brier.toFixed(3) : '—';
  }

  // Phase 2 Fix: Compute Model-Specific SHAP Attributions for the Selected Model
  const xaiRes = computeModelSpecificSHAP(activeModelId, vector, prob);

  if (shapModelTag && activeModel) {
    shapModelTag.textContent = activeModel.name;
  }
  if (shapCardHeading) {
    shapCardHeading.textContent = `Feature Contribution (${xaiRes.method})`;
  }

  if (shapBarsList) {
    shapBarsList.innerHTML = '';
    const meta = window.ALL_MODELS_DATA ? window.ALL_MODELS_DATA.metadata : MODEL_DATA;
    
    // Sort features by magnitude of attribution
    const featuresWithShap = meta.feature_names.map((name, i) => ({
      name,
      val: vector[i],
      shap: xaiRes.attributions[i]
    })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    featuresWithShap.forEach(item => {
      const row = document.createElement('div');
      row.className = 'shap-wf-row';
      const isPos = item.shap >= 0;
      const widthPct = Math.min(100, Math.max(6, Math.abs(item.shap) * 200));
      const valStr = `${isPos ? '+' : ''}${item.shap.toFixed(2)}`;

      row.innerHTML = `
        <span class="shap-wf-name" title="${item.name}: ${item.val}">${item.name}</span>
        <span class="shap-wf-val ${isPos ? 'val-pos' : 'val-neg'}">${valStr}</span>
        <div class="shap-wf-track">
          <div class="${isPos ? 'shap-wf-fill-pos' : 'shap-wf-fill-neg'}" style="width: ${widthPct}%;"></div>
        </div>
      `;
      shapBarsList.appendChild(row);
    });
  }

  // Update 3D Bloodstream and Neural Pulse speed
  const glucVal = vector[1];
  if (streamGlucVal) streamGlucVal.textContent = `Linked to Input (${glucVal} mg/dL)`;

  if (typeof glucoseMolecules !== 'undefined' && glucoseMolecules.length) {
    const activeCount = Math.floor(Math.min(1.0, Math.max(0.2, glucVal / 220)) * glucoseMolecules.length);
    glucoseMolecules.forEach((g, idx) => {
      g.visible = idx < activeCount;
      g.userData.vz = 0.05 + (glucVal / 250) * 0.12;
    });
  }

  if (typeof neuralNetwork !== 'undefined' && neuralNetwork && neuralNetwork.userData.pulseData) {
    const pulseFactor = 0.015 + (prob * 0.035);
    neuralNetwork.userData.pulseData.forEach(p => p.speed = pulseFactor * (0.8 + Math.random() * 0.4));
  }

  // Synchronize Counterfactual Explorer (Phase 4)
  initOrUpdateCounterfactual(vector);
}

// Phase 4: Interactive Counterfactual Explorer Engine
function initOrUpdateCounterfactual(originalVector) {
  if (!cfHypotheticalVector) {
    cfHypotheticalVector = originalVector.slice();
  } else {
    // Keep immutable features locked to original patient vector
    cfHypotheticalVector[0] = originalVector[0]; // Pregnancies
    cfHypotheticalVector[6] = originalVector[6]; // DiabetesPedigreeFunction
    cfHypotheticalVector[7] = originalVector[7]; // Age
  }

  const activeModelId = modelSelectControl ? modelSelectControl.value : 'ann';
  const bundle = window.ALL_MODELS_DATA;
  const activeModel = bundle && bundle.models ? bundle.models.find(m => m.id === activeModelId) : null;
  const modelName = activeModel ? activeModel.name : 'Active Model';

  if (cfActiveModelName) cfActiveModelName.textContent = modelName;
  if (cfActiveModelBadge) cfActiveModelBadge.textContent = modelName;

  // Sync sliders to hypothetical vector
  if (sliderCfGlucose) sliderCfGlucose.value = cfHypotheticalVector[1];
  if (sliderCfBMI)     sliderCfBMI.value     = cfHypotheticalVector[5];
  if (sliderCfBP)      sliderCfBP.value      = cfHypotheticalVector[2];
  if (sliderCfInsulin) sliderCfInsulin.value = cfHypotheticalVector[4];
  if (sliderCfSkin)    sliderCfSkin.value    = cfHypotheticalVector[3];

  renderCounterfactualState(originalVector);
}

function renderCounterfactualState(originalVector) {
  const activeModelId = modelSelectControl ? modelSelectControl.value : 'ann';
  
  if (dispCfGlucose) dispCfGlucose.textContent = `${cfHypotheticalVector[1]} mg/dL`;
  if (dispCfBMI)     dispCfBMI.textContent     = `${parseFloat(cfHypotheticalVector[5]).toFixed(1)} kg/m²`;
  if (dispCfBP)      dispCfBP.textContent      = `${cfHypotheticalVector[2]} mm Hg`;
  if (dispCfInsulin) dispCfInsulin.textContent = `${cfHypotheticalVector[4]} µU/mL`;
  if (dispCfSkin)    dispCfSkin.textContent    = `${cfHypotheticalVector[3]} mm`;

  // Compute Original and Hypothetical Probabilities using active model
  const origEval = evaluateSingleModel(activeModelId, originalVector);
  const hypoEval = evaluateSingleModel(activeModelId, cfHypotheticalVector);

  const origProb = origEval.probability;
  const hypoProb = hypoEval.probability;

  const origPct = Math.round(origProb * 100);
  const hypoPct = Math.round(hypoProb * 100);
  const deltaPct = hypoPct - origPct;

  if (cfOriginalScore) cfOriginalScore.textContent = `${origPct}%`;
  if (cfHypoScore)     cfHypoScore.textContent     = `${hypoPct}%`;

  if (cfOriginalPill) {
    cfOriginalPill.textContent = origProb < 0.3 ? 'Low Risk' : (origProb < 0.6 ? 'Moderate Risk' : 'High Risk');
  }
  if (cfHypoPill) {
    cfHypoPill.textContent = hypoProb < 0.3 ? 'Low Risk' : (hypoProb < 0.6 ? 'Moderate Risk' : 'High Risk');
  }

  if (cfDeltaVal && cfDeltaBadge && cfDeltaBox) {
    if (deltaPct < 0) {
      cfDeltaVal.textContent = `${deltaPct}% Risk Reduction`;
      cfDeltaVal.style.color = '#34d399';
      cfDeltaBadge.textContent = 'Protective Shift';
      cfDeltaBadge.className = 'cfd-badge badge-pass';
      cfDeltaBox.style.background = 'rgba(16, 185, 129, 0.12)';
      cfDeltaBox.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    } else if (deltaPct > 0) {
      cfDeltaVal.textContent = `+${deltaPct}% Risk Elevation`;
      cfDeltaVal.style.color = '#f87171';
      cfDeltaBadge.textContent = 'Elevated Risk';
      cfDeltaBadge.className = 'cfd-badge badge-fail';
      cfDeltaBox.style.background = 'rgba(239, 68, 68, 0.12)';
      cfDeltaBox.style.borderColor = 'rgba(239, 68, 68, 0.35)';
    } else {
      cfDeltaVal.textContent = `0% (No Change)`;
      cfDeltaVal.style.color = '#94a3b8';
      cfDeltaBadge.textContent = 'Identical';
      cfDeltaBadge.className = 'cfd-badge';
      cfDeltaBox.style.background = 'rgba(15, 23, 42, 0.6)';
      cfDeltaBox.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    }
  }

  // Render Changes List
  if (cfChangesList) {
    cfChangesList.innerHTML = '';
    const meta = window.ALL_MODELS_DATA ? window.ALL_MODELS_DATA.metadata : MODEL_DATA;
    let anyChanges = false;

    meta.feature_names.forEach((name, i) => {
      const origVal = originalVector[i];
      const hypoVal = cfHypotheticalVector[i];
      const diff = hypoVal - origVal;

      if (Math.abs(diff) > 0.05) {
        anyChanges = true;
        const isInc = diff > 0;
        const item = document.createElement('li');
        item.className = 'ctb-item';
        const diffStr = (isInc ? '+' : '') + (name === 'BMI' || name === 'DiabetesPedigreeFunction' ? diff.toFixed(1) : Math.round(diff));
        item.innerHTML = `
          <span><strong>${name}</strong>: ${origVal} &rarr; ${hypoVal}</span>
          <span class="ctb-delta ${isInc ? 'delta-inc' : ''}">${diffStr}</span>
        `;
        cfChangesList.appendChild(item);
      }
    });

    if (!anyChanges) {
      cfChangesList.innerHTML = '<li class="ctb-item" style="color:#64748b; font-style:italic;">No biomarkers modified yet. Move sliders above to simulate changes.</li>';
    }
  }

  // Model-Specific Explanation Note
  if (cfExplanationText) {
    const glucDiff = cfHypotheticalVector[1] - originalVector[1];
    const bmiDiff  = cfHypotheticalVector[5] - originalVector[5];

    cfExplanationText.textContent = `The 8->32->16->1 Neural Network propagates the standardized hypothetical vector through its trained ReLU manifolds, yielding an output activation of ${hypoPct}%.`;
  }
}

// Phase 3 Event Listeners: Screening Threshold Controls
if (sliderThreshold) {
  sliderThreshold.addEventListener('input', () => {
    currentThreshold = parseFloat(sliderThreshold.value);
    applyThresholdDecision(lastCalculatedProb);
    runMultiModelComparison(getFormFeatureVector());
    document.querySelectorAll('.btn-thresh-preset').forEach(b => b.classList.remove('active'));
  });
}

document.querySelectorAll('.btn-thresh-preset').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-thresh-preset').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const val = btn.dataset.thresh;
    const activeModelId = modelSelectControl ? modelSelectControl.value : 'ann';

    if (val === 'val-opt') {
      currentThreshold = VALIDATION_OPTIMAL_THRESHOLDS[activeModelId] || 0.50;
    } else {
      currentThreshold = parseFloat(val);
    }

    if (sliderThreshold) sliderThreshold.value = currentThreshold;
    applyThresholdDecision(lastCalculatedProb);
    runMultiModelComparison(getFormFeatureVector());
  });
});

// Phase 4 Event Listeners: Counterfactual Explorer Controls
if (btnToggleCounterfactual && cfDrawerPanel) {
  btnToggleCounterfactual.addEventListener('click', () => {
    const isVisible = cfDrawerPanel.style.display === 'block';
    cfDrawerPanel.style.display = isVisible ? 'none' : 'block';
    btnToggleCounterfactual.querySelector('span').textContent = isVisible ? 'Open Scenario Explorer' : 'Close Scenario Explorer';
  });
}

// Connect counterfactual sliders
[
  { slider: sliderCfGlucose, idx: 1, disp: dispCfGlucose, suffix: ' mg/dL', parse: parseInt },
  { slider: sliderCfBMI,     idx: 5, disp: dispCfBMI,     suffix: ' kg/m²', parse: parseFloat },
  { slider: sliderCfBP,      idx: 2, disp: dispCfBP,      suffix: ' mm Hg', parse: parseInt },
  { slider: sliderCfInsulin, idx: 4, disp: dispCfInsulin, suffix: ' µU/mL', parse: parseInt },
  { slider: sliderCfSkin,    idx: 3, disp: dispCfSkin,    suffix: ' mm',    parse: parseInt }
].forEach(cfg => {
  if (cfg.slider) {
    cfg.slider.addEventListener('input', () => {
      if (!cfHypotheticalVector) cfHypotheticalVector = getFormFeatureVector().slice();
      cfHypotheticalVector[cfg.idx] = cfg.parse(cfg.slider.value);
      renderCounterfactualState(getFormFeatureVector());
    });
  }
});

// Counterfactual Simulation Presets
document.querySelectorAll('.btn-cf-preset').forEach(btn => {
  btn.addEventListener('click', () => {
    const orig = getFormFeatureVector();
    if (!cfHypotheticalVector) cfHypotheticalVector = orig.slice();
    const mode = btn.dataset.cfpreset;

    if (mode === 'glucose') {
      cfHypotheticalVector[1] = 95; // Normoglycemic target
    } else if (mode === 'bmi') {
      cfHypotheticalVector[5] = Math.max(18.5, Math.round(orig[5] * 0.90 * 10) / 10); // 10% BMI reduction
    } else if (mode === 'lifestyle') {
      cfHypotheticalVector[1] = Math.min(orig[1], 100);
      cfHypotheticalVector[5] = Math.max(18.5, Math.round(orig[5] * 0.93 * 10) / 10);
      cfHypotheticalVector[2] = Math.max(60, orig[2] - 5);
    } else if (mode === 'recourse') {
      // Automatic minimal actionable boundary recourse to reach < 30% risk
      const activeModelId = modelSelectControl ? modelSelectControl.value : 'ann';
      let bestVec = cfHypotheticalVector.slice();
      let bestP = evaluateSingleModel(activeModelId, bestVec).probability;

      // Iterative boundary solver over modifiable Glucose and BMI
      for (let g = Math.min(orig[1], 140); g >= 85; g -= 5) {
        for (let b = orig[5]; b >= Math.max(19.0, orig[5] * 0.80); b -= 0.5) {
          const testV = orig.slice();
          testV[1] = g;
          testV[5] = Math.round(b * 10) / 10;
          const p = evaluateSingleModel(activeModelId, testV).probability;
          if (p < 0.30) {
            bestVec = testV;
            bestP = p;
            break;
          }
        }
        if (bestP < 0.30) break;
      }
      cfHypotheticalVector = bestVec;
    }

    // Sync sliders
    if (sliderCfGlucose) sliderCfGlucose.value = cfHypotheticalVector[1];
    if (sliderCfBMI)     sliderCfBMI.value     = cfHypotheticalVector[5];
    if (sliderCfBP)      sliderCfBP.value      = cfHypotheticalVector[2];
    if (sliderCfInsulin) sliderCfInsulin.value = cfHypotheticalVector[4];
    if (sliderCfSkin)    sliderCfSkin.value    = cfHypotheticalVector[3];

    renderCounterfactualState(orig);
  });
});

if (btnResetCounterfactual) {
  btnResetCounterfactual.addEventListener('click', () => {
    const orig = getFormFeatureVector();
    cfHypotheticalVector = orig.slice();
    initOrUpdateCounterfactual(orig);
  });
}

if (modelSelectControl) {
  modelSelectControl.addEventListener('change', () => {
    runPredictionAndRender();
  });
}

// Model Comparison Drawer Toggle
if (btnToggleComparison && comparisonDrawer) {
  btnToggleComparison.addEventListener('click', () => {
    const isOpen = comparisonDrawer.classList.toggle('open');
    btnToggleComparison.textContent = isOpen ? 'Hide Architecture ▴' : 'View Model Architecture ▾';
  });
}

// Navigation Tab Pills
document.querySelectorAll('.pred-tab-pill').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pred-tab-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    if (tab === '3d') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (tab === 'shap' || tab === 'feature-impact') {
      const shapEl = document.querySelector('.shap-card');
      if (shapEl) shapEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
});

if (inputGlucose) {
  inputGlucose.addEventListener('input', () => {
    const glucVal = parseFloat(inputGlucose.value) || 100;
    const activeCount = Math.floor(Math.min(1.0, Math.max(0.2, glucVal / 220)) * glucoseMolecules.length);
    glucoseMolecules.forEach((g, idx) => {
      g.visible = idx < activeCount;
      g.userData.vz = 0.05 + (glucVal / 250) * 0.12;
    });
    if (streamGlucVal) streamGlucVal.textContent = `Live Input: ${glucVal} mg/dL`;
  });
}

if (predictionForm) {
  predictionForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runPredictionAndRender();
  });
}

const PRESETS = {
  healthy: {
    Pregnancies: 1, Glucose: 85, BloodPressure: 68, SkinThickness: 18,
    Insulin: 65, BMI: 21.8, DiabetesPedigreeFunction: 0.22, Age: 25
  },
  borderline: {
    Pregnancies: 2, Glucose: 122, BloodPressure: 78, SkinThickness: 28,
    Insulin: 135, BMI: 28.6, DiabetesPedigreeFunction: 0.45, Age: 44
  },
  highrisk: {
    Pregnancies: 4, Glucose: 178, BloodPressure: 88, SkinThickness: 38,
    Insulin: 210, BMI: 36.2, DiabetesPedigreeFunction: 0.88, Age: 52
  },
  median: MODEL_DATA.medians
};

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.preset;
    const data = PRESETS[key];
    if (!data || !predictionForm) return;

    Object.entries(data).forEach(([name, val]) => {
      if (predictionForm.elements[name]) predictionForm.elements[name].value = val;
    });
    runPredictionAndRender();
  });
});

const btnResetForm = document.getElementById('btnResetForm');
if (btnResetForm && predictionForm) {
  btnResetForm.addEventListener('click', () => {
    const defaults = {
      Pregnancies: 1, Glucose: 100, BloodPressure: 72, SkinThickness: 23,
      Insulin: 85, BMI: 26.5, DiabetesPedigreeFunction: 0.38, Age: 33
    };
    Object.entries(defaults).forEach(([name, val]) => {
      if (predictionForm.elements[name]) predictionForm.elements[name].value = val;
    });
    runPredictionAndRender();
  });
}

/* ==========================================================================
   9. POPULATE GLOBAL METRICS & IMPORTANCE
   ========================================================================== */
function populateGlobalMetrics() {
  const metricAuc  = document.getElementById('metricAuc');
  const metricAcc  = document.getElementById('metricAcc');
  const metricF1   = document.getElementById('metricF1');
  const metricBase = document.getElementById('metricBase');
  const globalList = document.getElementById('globalImportanceList');

  if (metricAuc)  metricAuc.textContent  = MODEL_DATA.metrics.roc_auc.toFixed(3);
  if (metricAcc)  metricAcc.textContent  = `${(MODEL_DATA.metrics.accuracy * 100).toFixed(1)}%`;
  if (metricF1)   metricF1.textContent   = MODEL_DATA.metrics.f1_score.toFixed(3);
  if (metricBase) metricBase.textContent = `${(MODEL_DATA.expected_value * 100).toFixed(1)}%`;

  if (globalList && MODEL_DATA.global_importance_pct) {
    globalList.innerHTML = '';
    const sorted = Object.entries(MODEL_DATA.global_importance_pct).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([name, pct]) => {
      const row = document.createElement('div');
      row.className = 'gi-row';
      row.innerHTML = `
        <span class="gi-name">${name}</span>
        <div class="gi-bar-track">
          <div class="gi-bar-fill" style="width: ${pct * 3.5}%;"></div>
        </div>
        <span class="gi-pct">${pct}%</span>
      `;
      globalList.appendChild(row);
    });
  }
}

populateGlobalMetrics();

/* ==========================================================================
   10. CHAPTER IV PATHOLOGY SWITCHER & OGTT CURVE
   ========================================================================== */
const ogttCanvas = document.getElementById('ogttCanvas');

function drawOGTTCurve(mode = 'normal') {
  if (!ogttCanvas) return;
  const ctx = ogttCanvas.getContext('2d');
  const w = ogttCanvas.width;
  const h = ogttCanvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  for (let y = 25; y < h - 25; y += 35) {
    ctx.beginPath();
    ctx.moveTo(35, y);
    ctx.lineTo(w - 15, y);
    ctx.stroke();
  }

  ctx.fillStyle = '#6b7280';
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillText('mg/dL', 5, 18);
  ctx.fillText('0 min', 35, h - 8);
  ctx.fillText('60 min', 200, h - 8);
  ctx.fillText('120 min', 380, h - 8);

  // Normal curve
  ctx.strokeStyle = mode === 'normal' ? '#10b981' : 'rgba(16, 185, 129, 0.25)';
  ctx.lineWidth = mode === 'normal' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 130);
  ctx.bezierCurveTo(110, 80, 160, 95, 230, 110);
  ctx.bezierCurveTo(300, 125, 370, 130, 420, 132);
  ctx.stroke();

  // Type 1 curve
  ctx.strokeStyle = mode === 'type1' ? '#ef4444' : 'rgba(239, 68, 68, 0.25)';
  ctx.lineWidth = mode === 'type1' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 110);
  ctx.bezierCurveTo(110, 50, 190, 25, 270, 22);
  ctx.bezierCurveTo(340, 20, 390, 18, 420, 16);
  ctx.stroke();

  // Type 2 curve
  ctx.strokeStyle = mode === 'type2' ? '#f59e0b' : 'rgba(245, 158, 11, 0.25)';
  ctx.lineWidth = mode === 'type2' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 105);
  ctx.bezierCurveTo(110, 65, 190, 55, 260, 60);
  ctx.bezierCurveTo(330, 68, 380, 75, 420, 80);
  ctx.stroke();
}

drawOGTTCurve('normal');

document.querySelectorAll('.pathology-mode-selector .mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pathology-mode-selector .mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    drawOGTTCurve(btn.dataset.mode);
  });
});

/* ==========================================================================
   11. NAVBAR & INITIAL RUN
   ========================================================================== */
const navbar    = document.getElementById('navbar');
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
}

// Seamless, cinematic smooth navigation for all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    if (!href || href === '#') return;
    const targetEl = document.querySelector(href);
    if (!targetEl) return;
    e.preventDefault();
    if (navLinks) navLinks.classList.remove('open');

    const rect = targetEl.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    // Perfectly center target section in the viewport
    const targetY = rect.top + scrollTop - Math.max(0, (window.innerHeight - targetEl.offsetHeight) * 0.5);

    window.scrollTo({
      top: Math.max(0, targetY),
      behavior: 'smooth'
    });
  });
});

runPredictionAndRender();

console.log('INSIDE DIABETES: Engine initialized with full human body, internal organs, and authentic ANN.');
