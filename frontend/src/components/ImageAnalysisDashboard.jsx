import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  UploadCloud,
  RotateCcw,
  Layers,
  Activity,
  Flame,
  Droplets,
  Mountain,
  Wind,
  CheckCircle2,
  AlertTriangle,
  Sliders,
  ZoomIn,
  Eye,
  Microscope,
  Crosshair,
  Sparkles,
  ArrowRight,
  Maximize2,
  Scan,
  ShieldAlert,
  Zap,
  Cpu,
  BarChart3,
  Compass
} from 'lucide-react';

/* ── Hazard Spectral Profiles ──
   Each hazard radiates its own tactical mission-control wavelength,
   transforming the entire atmospheric glow, laser reticles, dials, and telemetry. */
const HAZARD_PROFILES = {
  wildfire: {
    id: 'wildfire',
    name: 'Wildfire',
    code: 'HZ-FIRE-01',
    badgeText: 'Thermal Radiation Alert',
    Icon: Flame,
    accent: '#ff5a1f',
    accentSoft: '#ff8a4c',
    secondary: '#ff3b30',
    glow: 'rgba(255, 90, 31, 0.35)',
    glowSoft: 'rgba(255, 90, 31, 0.12)',
    reticle: '#ff7849',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(255, 90, 31, 0.28), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(255, 59, 48, 0.16), transparent 60%)',
    border: 'rgba(255, 90, 31, 0.35)',
    borderHover: 'rgba(255, 90, 31, 0.65)',
    tagBg: 'rgba(255, 90, 31, 0.12)',
  },
  flood: {
    id: 'flood',
    name: 'Flood',
    code: 'HZ-AQUA-02',
    badgeText: 'Hydrological Surge Alert',
    Icon: Droplets,
    accent: '#39a9ff',
    accentSoft: '#70c0ff',
    secondary: '#00d2ff',
    glow: 'rgba(57, 169, 255, 0.35)',
    glowSoft: 'rgba(57, 169, 255, 0.12)',
    reticle: '#60b5ff',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(57, 169, 255, 0.28), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(0, 210, 255, 0.16), transparent 60%)',
    border: 'rgba(57, 169, 255, 0.35)',
    borderHover: 'rgba(57, 169, 255, 0.65)',
    tagBg: 'rgba(57, 169, 255, 0.12)',
  },
  earthquake: {
    id: 'earthquake',
    name: 'Earthquake',
    code: 'HZ-SEIS-03',
    badgeText: 'Tectonic Displacement Alert',
    Icon: Activity,
    accent: '#f6b84b',
    accentSoft: '#ffd17a',
    secondary: '#e6a23c',
    glow: 'rgba(246, 184, 75, 0.35)',
    glowSoft: 'rgba(246, 184, 75, 0.12)',
    reticle: '#ffc86b',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(246, 184, 75, 0.26), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(230, 162, 60, 0.16), transparent 60%)',
    border: 'rgba(246, 184, 75, 0.35)',
    borderHover: 'rgba(246, 184, 75, 0.65)',
    tagBg: 'rgba(246, 184, 75, 0.12)',
  },
  hurricane: {
    id: 'hurricane',
    name: 'Hurricane',
    code: 'HZ-CYCL-04',
    badgeText: 'Barometric Storm Vortex',
    Icon: Wind,
    accent: '#a370f7',
    accentSoft: '#c49cfc',
    secondary: '#7952b3',
    glow: 'rgba(163, 112, 247, 0.35)',
    glowSoft: 'rgba(163, 112, 247, 0.12)',
    reticle: '#ba8eff',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(163, 112, 247, 0.28), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(121, 82, 179, 0.16), transparent 60%)',
    border: 'rgba(163, 112, 247, 0.35)',
    borderHover: 'rgba(163, 112, 247, 0.65)',
    tagBg: 'rgba(163, 112, 247, 0.12)',
  },
  landslide: {
    id: 'landslide',
    name: 'Landslide',
    code: 'HZ-GEOT-05',
    badgeText: 'Slope Stability Failure',
    Icon: Mountain,
    accent: '#31d07d',
    accentSoft: '#5ee69e',
    secondary: '#10b981',
    glow: 'rgba(49, 208, 125, 0.35)',
    glowSoft: 'rgba(49, 208, 125, 0.12)',
    reticle: '#4be090',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(49, 208, 125, 0.28), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(16, 185, 129, 0.16), transparent 60%)',
    border: 'rgba(49, 208, 125, 0.35)',
    borderHover: 'rgba(49, 208, 125, 0.65)',
    tagBg: 'rgba(49, 208, 125, 0.12)',
  },
  default: {
    id: 'default',
    name: 'Mission Baseline',
    code: 'SYS-ORBIT-00',
    badgeText: 'Multimodal Sensor Ready',
    Icon: Crosshair,
    accent: '#ff5a1f',
    accentSoft: '#ff8a4c',
    secondary: '#4fd1b0',
    glow: 'rgba(255, 90, 31, 0.25)',
    glowSoft: 'rgba(255, 90, 31, 0.08)',
    reticle: '#ff8a4c',
    atmosphere:
      'radial-gradient(1100px 650px at 85% -10%, rgba(255, 90, 31, 0.22), transparent 65%), radial-gradient(800px 500px at -10% 110%, rgba(79, 209, 176, 0.10), transparent 60%)',
    border: 'rgba(239, 230, 211, 0.15)',
    borderHover: 'rgba(255, 90, 31, 0.45)',
    tagBg: 'rgba(255, 255, 255, 0.06)',
  },
};

function titleCase(str) {
  if (!str) return '';
  return String(str)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function ImageAnalysisDashboard({ onImageClick, onSwitchToChat }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [backend, setBackend] = useState('provided');
  const [selectedDisaster, setSelectedDisaster] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState('original');
  const [sliderPos, setSliderPos] = useState(50);
  const [activeRegion, setActiveRegion] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);
  const sliderContainerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  // Sync slider container dimensions for the split comparison view
  useEffect(() => {
    if (!sliderContainerRef.current) return;
    const update = () => {
      if (sliderContainerRef.current) {
        setContainerWidth(sliderContainerRef.current.offsetWidth);
      }
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [result]);

  // Derive dynamic hazard profile
  const activeHazardKey = useMemo(() => {
    if (result?.disaster_context?.label && result.disaster_context.label !== 'unknown') {
      return result.disaster_context.label.toLowerCase();
    }
    if (result?.disaster_type?.prediction && result.disaster_type.prediction !== 'unknown') {
      return result.disaster_type.prediction.toLowerCase();
    }
    if (selectedDisaster) {
      return selectedDisaster.toLowerCase();
    }
    return 'default';
  }, [result, selectedDisaster]);

  const hazard = HAZARD_PROFILES[activeHazardKey] || HAZARD_PROFILES.default;

  const handleFileSelect = (f) => {
    if (!f || !f.type.startsWith('image/')) return;
    setFile(f);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('localization_backend', backend);
    if (selectedDisaster) {
      formData.append('disaster_type', selectedDisaster);
    }

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || `Inference error: HTTP ${response.status}`);
      }
      setResult(data);
      setActiveView('original');
      setActiveRegion(null);
    } catch (err) {
      setError(err.message || 'Inspection pipeline failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setActiveView('original');
    setActiveRegion(null);
  };

  // Structured data accessors
  const pred =
    result?.disaster_type && result.disaster_type.prediction !== 'unknown'
      ? result.disaster_type
      : result?.informativeness || result?.disaster_type || {};

  const damagePred = result?.damage || {};
  const damage = result?.damage_assessment || {};
  const context = result?.disaster_context || { label: 'unknown' };

  const eventLabel =
    context.label && context.label !== 'unknown'
      ? `${titleCase(context.label)} Assessment`
      : damagePred.prediction && damagePred.prediction !== 'unknown'
      ? `${titleCase(damagePred.prediction)} Severity Analysis`
      : 'Disaster Assessment';

  return (
    <div className="relative min-h-screen text-bone">
      {/* Dynamic Peril Atmospheric Canvas */}
      <div
        className="pointer-events-none fixed inset-0 z-0 transition-all duration-1000"
        style={{ backgroundImage: hazard.atmosphere }}
      />
      <div className="topo pointer-events-none fixed inset-0 z-0 opacity-40" />
      <div className="grain pointer-events-none fixed inset-0 z-0 opacity-07" />

      {/* Tactical Sub-Bar: Peril Spectrum Selector & Active Telemetry Status */}
      <div className="relative z-10 mx-auto max-w-6xl px-4 pt-4 pb-2 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-bone/10 bg-basalt-2/70 px-4 py-2.5 backdrop-blur-2xl">
          <div className="flex items-center gap-2.5 font-mono text-xs">
            <span
              className="flex h-2 w-2 rounded-full transition-colors duration-500"
              style={{ backgroundColor: hazard.accent, boxShadow: `0 0 10px ${hazard.accent}` }}
            />
            <span className="font-bold text-bone">{hazard.name} Telemetry Active</span>
            <span className="text-ash">•</span>
            <span className="text-ash/80">{hazard.badgeText}</span>
          </div>

          {/* Quick Peril Context Bias Selector Pills */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-ash mr-1 hidden sm:inline">
              Peril Bias:
            </span>
            <button
              onClick={() => setSelectedDisaster('')}
              className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-mono tracking-wider transition-all ${
                !selectedDisaster
                  ? 'border border-bone/30 bg-white/10 font-bold text-bone shadow-sm'
                  : 'text-ash hover:text-bone hover:bg-white/5 border border-transparent'
              }`}
            >
              <Crosshair size={11} />
              <span>Auto-Detect</span>
            </button>
            {Object.values(HAZARD_PROFILES)
              .filter((h) => h.id !== 'default')
              .map((h) => {
                const isCurrent = (selectedDisaster || activeHazardKey) === h.id;
                return (
                  <button
                    key={h.id}
                    onClick={() => setSelectedDisaster(h.id === selectedDisaster ? '' : h.id)}
                    title={`Bias analysis pipeline toward ${h.name}`}
                    className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-mono tracking-wider transition-all ${
                      isCurrent
                        ? 'font-bold shadow-md'
                        : 'text-ash hover:text-bone hover:bg-white/5 border border-transparent'
                    }`}
                    style={
                      isCurrent
                        ? {
                            backgroundColor: h.tagBg,
                            color: h.accent,
                            border: `1px solid ${h.border}`,
                            boxShadow: `0 0 10px ${h.glowSoft}`,
                          }
                        : {}
                    }
                  >
                    <h.Icon size={11} strokeWidth={2.2} />
                    <span>{h.name}</span>
                  </button>
                );
              })}
          </div>
        </div>
      </div>

      {/* Main Analysis Container */}
      <div className="relative z-10 mx-auto max-w-6xl px-4 py-4 sm:px-6 lg:px-8">
        {/* PHASE 1: Targeted Upload & Ingestion Console */}
        {!result && (
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
            className="mx-auto max-w-3xl space-y-6"
          >
            {/* Header intro */}
            <div className="text-center space-y-2">
              <span className="font-mono text-[11px] uppercase tracking-[0.24em] text-ash">
                Aerial &amp; Satellite Surveillance Deck
              </span>
              <h2 className="font-display text-3xl font-extrabold tracking-tight text-bone sm:text-4xl">
                Inspect Visual Damage
              </h2>
              <p className="font-serif text-lg italic text-ash max-w-xl mx-auto">
                Submit an optical or satellite scene. The hybrid model extracts convolutional activations,
                computes LBP texture entropy, and maps damaged boundaries.
              </p>
            </div>

            {/* Targeting Reticle Dropzone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`group relative cursor-pointer overflow-hidden rounded-3xl border bg-basalt-2/80 p-8 text-center backdrop-blur-2xl transition-all duration-300 sm:p-12 ${
                isDragging ? 'scale-[1.01]' : 'hover:border-bone/30'
              }`}
              style={{
                borderColor: isDragging ? hazard.accent : hazard.border,
                boxShadow: isDragging ? `0 0 40px ${hazard.glow}` : '0 20px 60px -25px rgba(0,0,0,0.8)',
              }}
            >
              {/* Corner Targeting Crosshairs */}
              <div className="pointer-events-none absolute left-4 top-4 font-mono text-xs text-ash/60">+</div>
              <div className="pointer-events-none absolute right-4 top-4 font-mono text-xs text-ash/60">+</div>
              <div className="pointer-events-none absolute bottom-4 left-4 font-mono text-xs text-ash/60">+</div>
              <div className="pointer-events-none absolute bottom-4 right-4 font-mono text-xs text-ash/60">+</div>

              <div className="relative mx-auto flex flex-col items-center justify-center">
                <div
                  className="mb-5 grid h-20 w-20 place-items-center rounded-2xl border transition-all duration-500 group-hover:scale-105"
                  style={{
                    backgroundColor: hazard.tagBg,
                    borderColor: hazard.border,
                    boxShadow: `0 0 30px ${hazard.glowSoft}`,
                    color: hazard.accent,
                  }}
                >
                  <Scan size={36} strokeWidth={1.8} />
                </div>

                <h3 className="font-display text-xl font-bold tracking-tight text-bone sm:text-2xl">
                  {file ? file.name : 'Select or drop disaster surveillance frame'}
                </h3>
                <p className="mt-1.5 font-serif text-sm italic text-ash sm:text-base">
                  High-altitude UAV footage, multispectral orbit imagery, or ground-level survey
                </p>

                <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                  <span className="rounded-md border border-bone/10 bg-basalt-3/80 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-ash">
                    MAX 32 MB
                  </span>
                  <span className="rounded-md border border-bone/10 bg-basalt-3/80 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-ash">
                    PNG · JPG · WEBP · TIFF
                  </span>
                  <span
                    className="rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider"
                    style={{
                      borderColor: hazard.border,
                      backgroundColor: hazard.tagBg,
                      color: hazard.accent,
                    }}
                  >
                    M2 RESOLUTION ACTIVE
                  </span>
                </div>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.length) handleFileSelect(e.target.files[0]);
                }}
              />
            </div>

            {/* Thumbnail Preview Tile */}
            {previewUrl && (
              <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                className="relative overflow-hidden rounded-2xl border border-bone/15 bg-basalt-2/90 p-3 shadow-xl backdrop-blur-xl"
              >
                <div className="relative aspect-[16/9] w-full overflow-hidden rounded-xl bg-black/60">
                  <img
                    src={previewUrl}
                    alt="Loaded frame preview"
                    className="h-full w-full object-contain"
                  />
                  <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-basalt/80 via-transparent to-transparent" />
                  <div className="absolute bottom-3 left-4 flex items-center gap-2 font-mono text-xs text-bone">
                    <Crosshair size={14} style={{ color: hazard.accent }} />
                    <span>TARGET LOCKED · READY FOR FEATURE EXTRACTION</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                    className="absolute right-3 top-3 rounded-lg border border-bone/20 bg-basalt-3/90 px-2.5 py-1 font-mono text-xs text-ash transition-colors hover:border-ember hover:bg-ember hover:text-basalt"
                  >
                    Clear Target
                  </button>
                </div>
              </motion.div>
            )}

            {/* Pipeline Configuration Console */}
            <div className="grid grid-cols-1 gap-4 rounded-2xl border border-bone/10 bg-basalt-2/80 p-5 backdrop-blur-xl sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-ash">
                  <Cpu size={13} style={{ color: hazard.accent }} />
                  <span>Localization Backend</span>
                </label>
                <select
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                  className="w-full rounded-xl border border-bone/15 bg-basalt-3/90 px-3.5 py-2.5 font-mono text-xs text-bone outline-none transition focus:border-ember"
                >
                  <option value="provided">Fast Saliency (Production Default)</option>
                  <option value="sun_ica">SUN + ICA (Saliency Using Natural Stats)</option>
                  <option value="opencv">OpenCV Spectral Residual</option>
                  <option value="gradcam">Grad-CAM (Gradient Class Activation)</option>
                  <option value="gradcam_plus_plus">Grad-CAM++ (Pixel-Accurate Attention)</option>
                  <option value="scorecam">Score-CAM (Gradient-Free Scoring)</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-ash">
                  <Compass size={13} style={{ color: hazard.accent }} />
                  <span>Peril Context Bias</span>
                </label>
                <select
                  value={selectedDisaster}
                  onChange={(e) => setSelectedDisaster(e.target.value)}
                  className="w-full rounded-xl border border-bone/15 bg-basalt-3/90 px-3.5 py-2.5 font-mono text-xs text-bone outline-none transition focus:border-ember"
                >
                  <option value="">Auto-Detect from Scene Radiance</option>
                  <option value="wildfire">Wildfire (Vegetation loss, burn scar)</option>
                  <option value="flood">Flood (Water inundation, sediment)</option>
                  <option value="earthquake">Earthquake (Rubble, shear fracture)</option>
                  <option value="hurricane">Hurricane / Storm (Wind wash, debris)</option>
                  <option value="landslide">Landslide (Terrain displacement, mud)</option>
                </select>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2.5 rounded-xl border border-ember/40 bg-ember/10 p-3.5 font-mono text-xs text-ember-soft">
                <AlertTriangle size={16} className="shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Launch Ingestion Button */}
            <div className="space-y-3">
              <button
                disabled={!file || isAnalyzing}
                onClick={handleAnalyze}
                className="group relative flex w-full items-center justify-center gap-2.5 overflow-hidden rounded-2xl py-4 font-mono text-sm font-bold uppercase tracking-[0.16em] text-basalt shadow-2xl transition-all disabled:opacity-40 hover:scale-[1.01] active:scale-[0.99]"
                style={{
                  background: `linear-gradient(135deg, ${hazard.accent}, ${hazard.secondary})`,
                  boxShadow: `0 10px 40px -10px ${hazard.glow}`,
                }}
              >
                <Microscope size={18} strokeWidth={2.4} />
                <span>{isAnalyzing ? 'Extracting Saliency & Texture Vectors…' : 'Initiate Damage Analysis'}</span>
              </button>

              {isAnalyzing && (
                <div className="flex items-center justify-center gap-2 font-mono text-xs text-ash">
                  <div className="flex gap-1.5">
                    <span
                      className="h-2 w-2 animate-bounce rounded-full"
                      style={{ backgroundColor: hazard.accent, animationDelay: '0ms' }}
                    />
                    <span
                      className="h-2 w-2 animate-bounce rounded-full"
                      style={{ backgroundColor: hazard.secondary, animationDelay: '150ms' }}
                    />
                    <span
                      className="h-2 w-2 animate-bounce rounded-full"
                      style={{ backgroundColor: hazard.accentSoft, animationDelay: '300ms' }}
                    />
                  </div>
                  <span className="shimmer">
                    Running InceptionResNetV2 + Saliency M2 connected component cluster…
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* PHASE 2: Production Telemetry & Damage Research Deck */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
            className="space-y-6"
          >
            {/* Mission Deck Header */}
            <div
              className="relative overflow-hidden rounded-3xl border border-bone/15 bg-basalt-2/90 p-6 backdrop-blur-2xl sm:p-8"
              style={{
                boxShadow: `0 20px 70px -30px ${hazard.glowSoft}`,
              }}
            >
              {/* Tactical glow pill */}
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                <div>
                  <div className="flex items-center gap-2">
                    <span
                      className="font-mono text-[11px] font-bold uppercase tracking-[0.24em]"
                      style={{ color: hazard.accent }}
                    >
                      {hazard.code} · Research Inference
                    </span>
                    <span className="font-mono text-xs text-ash/60">•</span>
                    <span className="font-mono text-xs uppercase tracking-widest text-ash">
                      M2 Localization
                    </span>
                  </div>

                  <h2 className="mt-1 font-display text-3xl font-extrabold tracking-tight text-bone sm:text-5xl">
                    {eventLabel}
                  </h2>
                  <p className="mt-2 max-w-3xl font-serif text-base italic leading-relaxed text-bone/80 sm:text-lg">
                    {result.disaster_description}
                  </p>
                </div>

                <button
                  onClick={handleReset}
                  className="group flex shrink-0 items-center gap-2 self-start rounded-xl border border-bone/15 bg-basalt-3/80 px-4 py-2.5 font-mono text-xs uppercase tracking-wider text-bone shadow-md backdrop-blur-md transition-all hover:border-ember/60 hover:bg-basalt-3 sm:self-center active:scale-95"
                >
                  <RotateCcw size={14} className="transition-transform group-hover:-rotate-90" />
                  <span>Inspect New Scene</span>
                </button>
              </div>
            </div>

            {/* 4 Primary Telemetry HUD Cards */}
            <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              {/* Card 1: Damage Severity Class */}
              <div
                className="relative flex min-h-[140px] flex-col justify-between rounded-2xl border p-5 backdrop-blur-xl transition-all"
                style={{
                  backgroundColor: 'rgba(20, 17, 15, 0.85)',
                  borderColor: hazard.border,
                  boxShadow: `0 10px 30px -15px ${hazard.glowSoft}`,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-widest text-ash">
                    Damage Severity
                  </span>
                  <ShieldAlert size={15} style={{ color: hazard.accent }} />
                </div>
                <div>
                  <strong className="block font-display text-2xl font-bold tracking-tight text-bone sm:text-3xl">
                    {titleCase(damagePred.prediction || 'Unclassified')}
                  </strong>
                  <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-basalt-3">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${damagePred.confidence || 0}%` }}
                      transition={{ duration: 1.2, ease: 'easeOut' }}
                      className="h-full rounded-full"
                      style={{
                        background: `linear-gradient(90deg, ${hazard.secondary}, ${hazard.accent})`,
                      }}
                    />
                  </div>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-ash">
                  Calibrated Confidence {damagePred.confidence || 0}%
                </span>
              </div>

              {/* Card 2: DEM Score */}
              <div className="relative flex min-h-[140px] flex-col justify-between rounded-2xl border border-bone/10 bg-basalt-2/85 p-5 backdrop-blur-xl">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-widest text-ash">
                    DEM Metric Score
                  </span>
                  <Zap size={15} style={{ color: hazard.accent }} />
                </div>
                <div>
                  <strong className="font-mono text-3xl font-extrabold tracking-tight text-bone sm:text-4xl">
                    {damage.dem_score}
                  </strong>
                  <span className="ml-1 font-mono text-xs text-ash">/ 100</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: hazard.accent, boxShadow: `0 0 8px ${hazard.accent}` }}
                  />
                  <span className="font-mono text-xs font-bold uppercase tracking-wider" style={{ color: hazard.accent }}>
                    {damage.severity_level || 'Moderate'}
                  </span>
                </div>
              </div>

              {/* Card 3: Affected Footprint Area */}
              <div className="relative flex min-h-[140px] flex-col justify-between rounded-2xl border border-bone/10 bg-basalt-2/85 p-5 backdrop-blur-xl">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-widest text-ash">
                    Affected Footprint
                  </span>
                  <Crosshair size={15} style={{ color: hazard.accent }} />
                </div>
                <div>
                  <strong className="font-mono text-3xl font-extrabold tracking-tight text-bone sm:text-4xl">
                    {damage.affected_area_percentage}%
                  </strong>
                </div>
                <span className="font-mono text-xs text-ash">
                  {damage.damaged_region_count || 0} discrete damage clusters
                </span>
              </div>

              {/* Card 4: Inference Latency */}
              <div className="relative flex min-h-[140px] flex-col justify-between rounded-2xl border border-bone/10 bg-basalt-2/85 p-5 backdrop-blur-xl">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-widest text-ash">
                    Latency &amp; Features
                  </span>
                  <Cpu size={15} style={{ color: hazard.accent }} />
                </div>
                <div>
                  <strong className="font-mono text-3xl font-extrabold tracking-tight text-bone sm:text-4xl">
                    {result.processing_time_ms}
                  </strong>
                  <span className="ml-1 font-mono text-xs text-ash">ms</span>
                </div>
                <span className="font-mono text-xs text-ash truncate">
                  Backend: {damage.localization_backend || 'Fast Saliency'}
                </span>
              </div>
            </section>

            {/* Visual Evidence Workspace & Telemetry Deck */}
            <section className="grid grid-cols-1 gap-6 lg:grid-cols-[1.85fr_0.85fr]">
              {/* Left Viewer Panel */}
              <div className="flex flex-col gap-4 rounded-3xl border border-bone/15 bg-basalt-2/90 p-5 shadow-2xl backdrop-blur-2xl sm:p-6">
                {/* Spectral View Selector */}
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Layers size={16} style={{ color: hazard.accent }} />
                    <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-bone">
                      Spectral Evidence Layers
                    </h3>
                  </div>

                  <div className="flex flex-wrap gap-1 rounded-xl border border-bone/10 bg-basalt-3/80 p-1">
                    {[
                      { key: 'original', label: '01 Original' },
                      { key: 'saliency_overlay', label: '02 Saliency' },
                      { key: 'lbp', label: '03 LBP Texture' },
                      { key: 'gradcam', label: '04 GradCAM++' },
                      { key: 'ddm', label: '05 DDM Overlay' },
                      { key: 'damage_mask', label: '06 Binary Mask' },
                    ].map((tab) => {
                      const isActive = activeView === tab.key;
                      return (
                        <button
                          key={tab.key}
                          onClick={() => setActiveView(tab.key)}
                          className={`rounded-lg px-2.5 py-1 font-mono text-[11px] tracking-wider transition-all ${
                            isActive
                              ? 'font-bold shadow-md'
                              : 'text-ash hover:text-bone hover:bg-white/5'
                          }`}
                          style={
                            isActive
                              ? {
                                  backgroundColor: hazard.tagBg,
                                  color: hazard.accent,
                                  border: `1px solid ${hazard.border}`,
                                  boxShadow: `0 0 10px ${hazard.glowSoft}`,
                                }
                              : {}
                          }
                        >
                          {tab.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Primary Optical Display Stage */}
                <div
                  onClick={() => onImageClick?.(result.visualizations[activeView] || result.visualizations.original)}
                  className="group relative aspect-[16/10] w-full cursor-zoom-in overflow-hidden rounded-2xl border border-bone/15 bg-black/80 shadow-2xl"
                  title="Click to view full-resolution magnification"
                >
                  <img
                    src={result.visualizations[activeView] || result.visualizations.original}
                    alt={activeView}
                    className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-[1.01]"
                  />

                  {/* Corner Target Markers */}
                  <div className="pointer-events-none absolute left-3 top-3 font-mono text-[10px] text-bone/40">
                    SRC: {activeView.toUpperCase()}
                  </div>
                  <div className="pointer-events-none absolute right-3 bottom-3 font-mono text-[10px] text-bone/40">
                    DIM: {result.image_dimensions?.analysis ? result.image_dimensions.analysis.join('×') : 'WGS84'}
                  </div>

                  <div className="absolute right-3 top-3 rounded-lg border border-bone/20 bg-basalt-3/80 p-1.5 text-bone opacity-0 backdrop-blur-md transition-opacity group-hover:opacity-100">
                    <Maximize2 size={14} />
                  </div>
                </div>

                {/* Dual-Spectrum Curtain Slider (Original vs DDM) */}
                <div className="mt-2 space-y-2 rounded-2xl border border-bone/10 bg-basalt-3/40 p-4">
                  <div className="flex items-center justify-between font-mono text-xs text-ash">
                    <span className="flex items-center gap-1.5">
                      <Sliders size={13} style={{ color: hazard.accent }} />
                      Dual-Spectrum Curtain: Baseline RGB vs. Damage Density Map
                    </span>
                    <span className="font-bold" style={{ color: hazard.accent }}>
                      {sliderPos}% DDM Blend
                    </span>
                  </div>

                  <div
                    ref={sliderContainerRef}
                    className="relative aspect-[16/10] w-full select-none overflow-hidden rounded-xl border border-bone/20 bg-black/80"
                  >
                    {/* Baseline original image */}
                    <img
                      src={result.visualizations.original}
                      alt="Baseline"
                      className="absolute inset-0 h-full w-full object-contain"
                    />

                    {/* DDM overlay sliced at slider position */}
                    <div
                      className="absolute inset-y-0 left-0 overflow-hidden pointer-events-none transition-none"
                      style={{
                        width: `${sliderPos}%`,
                        borderRight: `2px solid ${hazard.accent}`,
                        boxShadow: `0 0 16px ${hazard.accent}`,
                      }}
                    >
                      <div
                        className="h-full"
                        style={{
                          width: containerWidth > 0 ? `${containerWidth}px` : '100%',
                        }}
                      >
                        <img
                          src={result.visualizations.ddm}
                          alt="DDM Overlay"
                          className="h-full w-full object-contain"
                        />
                      </div>
                    </div>

                    {/* HUD labels */}
                    <div className="pointer-events-none absolute left-3 top-2.5 rounded bg-black/75 px-2 py-0.5 font-mono text-[10px] uppercase text-white/90">
                      DDM Density
                    </div>
                    <div className="pointer-events-none absolute right-3 top-2.5 rounded bg-black/75 px-2 py-0.5 font-mono text-[10px] uppercase text-white/90">
                      Optical RGB
                    </div>

                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={sliderPos}
                      onChange={(e) => setSliderPos(Number(e.target.value))}
                      className="absolute inset-x-4 bottom-3.5 z-20 h-1.5 cursor-ew-resize rounded-lg bg-black/80 opacity-80 hover:opacity-100 transition-opacity"
                      style={{ accentColor: hazard.accent }}
                      aria-label="Split spectrum comparison"
                    />
                  </div>
                </div>
              </div>

              {/* Right Telemetry Column */}
              <aside className="space-y-4">
                {/* Circular DEM Radial Dial */}
                <div className="flex items-center gap-5 rounded-3xl border border-bone/15 bg-basalt-2/90 p-5 shadow-xl backdrop-blur-2xl">
                  {/* Conic Ring */}
                  <div
                    className="relative flex h-24 w-24 shrink-0 items-center justify-center rounded-full shadow-lg transition-all duration-700"
                    style={{
                      background: `conic-gradient(${hazard.accent} ${Math.min(
                        damage.dem_score || 0,
                        100
                      )}%, rgba(239, 230, 211, 0.08) 0)`,
                      boxShadow: `0 0 25px ${hazard.glowSoft}`,
                    }}
                  >
                    <div className="flex h-[76px] w-[76px] flex-col items-center justify-center rounded-full bg-basalt shadow-inner">
                      <span className="font-mono text-2xl font-black leading-none text-bone">
                        {Math.round(damage.dem_score || 0)}
                      </span>
                      <span className="font-mono text-[9px] uppercase tracking-widest text-ash">
                        DEM-100
                      </span>
                    </div>
                  </div>

                  <div>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-ash">
                      Damage Index
                    </span>
                    <strong className="block font-display text-xl font-bold text-bone">
                      {damage.severity_level || 'Evaluating'}
                    </strong>
                    <p className="mt-1 font-serif text-xs italic text-ash">
                      Density coefficient: {damage.average_damage_density || '0.00'}
                    </p>
                  </div>
                </div>

                {/* Top 5 Neural Predictions */}
                <div className="rounded-3xl border border-bone/15 bg-basalt-2/90 p-5 shadow-xl backdrop-blur-2xl">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5">
                      <BarChart3 size={14} style={{ color: hazard.accent }} />
                      Class Probability
                    </span>
                    <span className="font-mono text-[10px] text-ash">SOFTMAX</span>
                  </div>

                  <div className="space-y-2.5">
                    {pred.top5?.map((row, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex items-center justify-between font-mono text-xs">
                          <span className="capitalize text-bone/90">{titleCase(row.label)}</span>
                          <span className="font-bold" style={{ color: idx === 0 ? hazard.accent : undefined }}>
                            {row.confidence}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-basalt-3">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${row.confidence}%` }}
                            transition={{ duration: 0.8, delay: idx * 0.08 }}
                            className="h-full rounded-full"
                            style={{
                              background:
                                idx === 0
                                  ? `linear-gradient(90deg, ${hazard.secondary}, ${hazard.accent})`
                                  : 'rgba(239, 230, 211, 0.25)',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Surface Texture Diagnostics */}
                <div className="rounded-3xl border border-bone/15 bg-basalt-2/90 p-5 shadow-xl backdrop-blur-2xl">
                  <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5 mb-3">
                    <Activity size={14} style={{ color: hazard.accent }} />
                    LBP Texture Diagnostics
                  </span>

                  <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                    {Object.entries(result.texture_statistics || {}).map(([key, val]) => (
                      <div
                        key={key}
                        className="rounded-xl border border-bone/10 bg-basalt-3/60 p-2.5"
                      >
                        <span className="block text-[10px] uppercase text-ash">{titleCase(key)}</span>
                        <strong className="text-sm font-semibold text-bone">{val}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </aside>
            </section>

            {/* Detail Grid: Damage Clusters, Impact, Pipeline Breakdown & Recommendations */}
            <section className="grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* Connected Damage Clusters (Clickable region targets) */}
              <div className="rounded-3xl border border-bone/15 bg-basalt-2/90 p-6 shadow-xl backdrop-blur-2xl">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5">
                    <Crosshair size={14} style={{ color: hazard.accent }} />
                    Connected Damage Clusters
                  </span>
                  <span className="font-mono text-[10px] text-ash">M2 CLUSTERING</span>
                </div>

                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {damage.regions?.slice(0, 6).map((region, idx) => {
                    const isSel = activeRegion === idx;
                    return (
                      <button
                        key={idx}
                        onClick={() => {
                          setActiveRegion(idx);
                          setActiveView('ddm');
                        }}
                        className={`rounded-xl border p-2.5 text-left font-mono transition-all ${
                          isSel
                            ? 'shadow-lg'
                            : 'border-bone/10 bg-basalt-3/60 hover:border-bone/25'
                        }`}
                        style={
                          isSel
                            ? {
                                borderColor: hazard.accent,
                                backgroundColor: hazard.tagBg,
                                boxShadow: `0 0 14px ${hazard.glowSoft}`,
                              }
                            : {}
                        }
                      >
                        <div className="flex items-center justify-between text-xs font-bold">
                          <span>α-{idx + 1}</span>
                          <span style={{ color: hazard.accent }}>{region.area_percentage}%</span>
                        </div>
                        <span className="block text-[10px] text-ash truncate mt-0.5">
                          Centroid: {region.centroid?.join(', ')}
                        </span>
                      </button>
                    );
                  })}
                  {(!damage.regions || damage.regions.length === 0) && (
                    <div className="col-span-3 py-4 text-center font-mono text-xs text-ash">
                      No discrete connected damage clusters detected above threshold.
                    </div>
                  )}
                </div>

                <div className="mt-4 space-y-1.5 divide-y divide-bone/5 font-mono text-xs">
                  <div className="flex justify-between py-1">
                    <span className="text-ash">Texture Entropy</span>
                    <strong className="text-bone">{damage.texture_entropy}</strong>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-ash">Boundary Complexity</span>
                    <strong className="text-bone">{damage.boundary_complexity}</strong>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-ash">Localization Score</span>
                    <strong className="text-bone">{damage.localization_score}</strong>
                  </div>
                </div>
              </div>

              {/* Operational Impact Estimates */}
              <div className="rounded-3xl border border-bone/15 bg-basalt-2/90 p-6 shadow-xl backdrop-blur-2xl">
                <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5 mb-3">
                  <ShieldAlert size={14} style={{ color: hazard.accent }} />
                  Operational Impact Estimates
                </span>

                <div className="space-y-2 divide-y divide-bone/5 font-mono text-xs">
                  {Object.entries(damage.impact_estimates || {}).map(([k, v]) => (
                    <div key={k} className="flex justify-between py-2">
                      <span className="text-ash">{titleCase(k)}</span>
                      <strong className="font-bold text-bone" style={{ color: hazard.accent }}>
                        {v}
                      </strong>
                    </div>
                  ))}
                  {(!damage.impact_estimates || Object.keys(damage.impact_estimates).length === 0) && (
                    <div className="py-4 text-center font-mono text-xs text-ash">
                      No structural impact flags identified in visual footprint.
                    </div>
                  )}
                </div>
              </div>

              {/* Neural Architecture Breakdown (Full Width) */}
              <div className="col-span-1 rounded-3xl border border-bone/15 bg-basalt-2/90 p-6 shadow-xl backdrop-blur-2xl md:col-span-2">
                <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5 mb-4">
                  <Cpu size={14} style={{ color: hazard.accent }} />
                  Pipeline Feature Extraction Trace
                </span>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 font-mono text-xs">
                  {Object.entries(result.prediction_breakdown || {}).map(([key, desc], idx) => (
                    <div
                      key={idx}
                      className="rounded-2xl border border-bone/10 bg-basalt-3/60 p-4 space-y-1.5"
                      style={{ borderLeftColor: hazard.accent, borderLeftWidth: 3 }}
                    >
                      <strong className="block font-bold uppercase tracking-wider text-bone">
                        {titleCase(key)}
                      </strong>
                      <p className="font-serif text-xs italic leading-relaxed text-ash">
                        {desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Incident Command Directives (Full Width) */}
              <div className="col-span-1 rounded-3xl border border-bone/15 bg-basalt-2/90 p-6 shadow-xl backdrop-blur-2xl md:col-span-2">
                <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-bone flex items-center gap-1.5 mb-4">
                  <CheckCircle2 size={14} style={{ color: hazard.accent }} />
                  Incident Commander Tactical Directives
                </span>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {damage.emergency_recommendations?.map((rec, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 rounded-2xl border border-bone/10 bg-basalt-3/70 p-4 text-xs text-bone/90 shadow-sm leading-relaxed"
                    >
                      <span
                        className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full text-[10px] font-bold text-basalt"
                        style={{ backgroundColor: hazard.accent }}
                      >
                        {idx + 1}
                      </span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </motion.div>
        )}
      </div>
    </div>
  );
}
