# DisasterRes-Net: Comprehensive System Architecture & Agentic Workflow

This document provides a technical overview of **DisasterRes-Net**, covering model architecture, feature extraction, fusion strategies, multi-task prediction pipelines, deterministic & CrewAI agentic frameworks, and recommendations for future enhancements.

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [What is Being Predicted (Objectives & Targets)](#2-what-is-being-predicted-objectives--targets)
3. [How Predictions are Generated (End-to-End Pipeline)](#3-how-predictions-are-generated-end-to-end-pipeline)
   - [3.1 Saliency-Attended Visual Stream](#31-saliency-attended-visual-stream)
   - [3.2 Deep CNN Embeddings (Inception-ResNet-V2)](#32-deep-cnn-embeddings-inception-resnet-v2)
   - [3.3 Handcrafted Texture Descriptors (GLCM & LBP)](#33-handcrafted-texture-descriptors-glcm--lbp)
   - [3.4 Early Feature Fusion & Normalization](#34-early-feature-fusion--normalization)
   - [3.5 Damage Localization & DEM Scoring (Module M2)](#35-damage-localization--dem-scoring-module-m2)
4. [Prediction Classifiers & Optimization Methods](#4-prediction-classifiers--optimization-methods)
   - [4.1 Classifiers & Loss / Weighting Mechanics](#41-classifiers--loss--weighting-mechanics)
   - [4.2 Cross-Validation & Generalization Protocol (Stratified 5-Fold)](#42-cross-validation--generalization-protocol-stratified-5-fold)
   - [4.3 Computational & Training Optimizations](#43-computational--training-optimizations)
5. [Agentic Framework Implementation (CrewAI & Deterministic Runner)](#5-agentic-framework-implementation-crewai--deterministic-runner)
   - [5.1 Architecture Overview: Is There a Main Agent?](#51-architecture-overview-is-there-a-main-agent)
   - [5.2 Subagent Taxonomy & Roles](#52-subagent-taxonomy--roles)
   - [5.3 Tooling Registry](#53-tooling-registry)
6. [Current Storyline & Execution Lifecycle of an Agent](#6-current-storyline--execution-lifecycle-of-an-agent)
   - [6.1 Mode A: Conversational Query Flow (Autonomous Web/Satellite Ingestion)](#61-mode-a-conversational-query-flow-autonomous-websatellite-ingestion)
   - [6.2 Mode B: Direct Multimodal Evidence Flow (Upload Path)](#62-mode-b-direct-multimodal-evidence-flow-upload-path)
   - [6.3 Hallucination Defense: The Numeric Fidelity Gate](#63-hallucination-defense-the-numeric-fidelity-gate)
7. [Future Roadmap: Model, Training & Agentic Optimizations](#7-future-roadmap-model-training--agentic-optimizations)
   - [7.1 Machine Learning & Computer Vision Enhancements](#71-machine-learning--computer-vision-enhancements)
   - [7.2 Agentic Framework Enhancements](#72-agentic-framework-enhancements)

---

## 1. Executive Summary & High-Level Architecture

DisasterRes-Net is an emergency intelligence system combining **hybrid computer vision** with **agentic situational reasoning**. Inspired by the research of *Gupta & Roy (2024)*, the system processes overhead satellite, aerial drone, and ground-level disaster imagery to estimate disaster perils, classify damage severity, localize rubble/inundation/burn scars, and draft situational reports.

```
                              USER INTERFACES
         +--------------------------------------------------------+
         |  Flask Research Dashboard  |  Vite Modern React Desk   |
         |  (Interactive Overlays)    |  (SSE Streaming Chat UI)  |
         +--------------------------------------------------------+
                                     |
                       HTTP POST / REST / SSE Streams
                                     |
+------------------------------------+------------------------------------+
|                                                                         |
v                                                                         v
[app.py: Single Image Dashboard API]           [crew/runner.py: Agentic Execution Pool]
|                                              |
|-- Max Edge Resizing (1280px / 960px)         |-- ThreadPoolExecutor (max_workers=2)
|-- Direct Hybrid Pipeline Inference           |-- Bounded Queue per Session
|-- Connected Component Mask (M2)              |-- Server-Sent Events (SSE) Pusher
|-- DDM / DEM Damage Scores                    |
v                                              v
[HybridDisasterPipeline]                       [crew/crew.py: Agent Orchestrator]
(2036-dim Saliency-CNN-LBP-GLCM Fusion)        (Resolution -> Vision -> Aggregation -> Report)
```

---

## 2. What is Being Predicted (Objectives & Targets)

The system predicts qualitative classifications and spatial damage assessments across five key targets:

| Objective / Metric | Type | Classes / Range | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Disaster Type (Peril)** | Multi-Class Classification | `earthquake`, `flood`, `hurricane`, `landslide`, `wildfire`, `not_disaster` | Identifies the physical hazard to select peril-specific response protocols. |
| **Damage Severity** | Multi-Class Classification | `severe`, `mild`, `little_or_none` | Measures structural and environmental destruction for triage prioritization. |
| **Informativeness** | Binary Classification | `informative`, `not_informative` | Filters out irrelevant, uninformative, or obscured crisis imagery. |
| **Damage Evaluation Metric (DEM)** | Continuous Regression Score | `0.00` to `100.00` | Standardized composite score reflecting overall severity and spatial disruption. |
| **Damage Localization Mask & DDM** | Spatial Binary & Heatmap Segmentation | $H \times W$ 2D spatial coordinates | Localizes contiguous damage patches, spatial centroids, and affected area percentage. |

---

## 3. How Predictions are Generated (End-to-End Pipeline)

Prediction relies on an early-fusion hybrid architecture that couples deep spatial features with handcrafted texture descriptors and neuro-inspired visual attention.

```
+---------------------------------------------------------------------------------------------------+
|                                      INPUT RGB IMAGE (299 x 299)                                  |
+---------------------------------------------------------------------------------------------------+
       |                                       |                                    |
       v                                       v                                    v
[LMS Saliency Attention]            [Uniform LBP Descriptors]            [GLCM Texture Matrices]
       |                                       |                                    |
       +-------------------+                   | (Sub-band statistics)              | (4 distances, 4 angles)
       |                   |                   v                                    v
       v                   v            [LBP Stats (16-D)]                  [GLCM Vector (20-D)]
 [Original RGB]     [Saliency RGB]             |                                    |
       |                   |                   |                                    |
       v                   v                   |                                    |
[InceptionResNetV2] [InceptionResNetV2]        |                                    |
       |                   |                   |                                    |
       v                   v                   |                                    |
   [df1 (1000-D)]      [df2 (1000-D)]          |                                    |
       |                   |                   |                                    |
       +-------------------+-------------------+------------------------------------+
                                           |
                                           v
                       [Early Concatenation: 2036 Dimensions]
                                           |
                                           v
                           [StandardScaler Normalization]
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
                    v                      v                      v
             [Random Forest:        [Random Forest:        [Random Forest:
             Disaster Type]         Damage Severity]       Informativeness]
                    |                      |                      |
              6 Class Probs          3 Class Probs          2 Class Probs
```

### 3.1 Saliency-Attended Visual Stream
1. **LMS Color Space Transformation:** RGB values are normalized to $[0, 1]$ and mapped into the human long-, medium-, and short-wavelength (LMS) cone response space using the SUN color matrix:
   $$\begin{bmatrix} L \\ M \\ S \end{bmatrix} = \begin{bmatrix} 0.300 & 0.622 & 0.078 \\ 0.230 & 0.692 & 0.078 \\ 0.243 & 0.205 & 0.553 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$
2. **Color Opponent & Intensity Pyramids:** Derives Red-Green ($RG = L - M$), Blue-Yellow ($BY = S - (L + M)/2$), and Luminance ($I = (L + M + S)/3$) channels.
3. **Multi-Scale Center-Surround Differences:** Generates feature maps across dyadic Gaussian pyramid levels ($2$ through $5$) to isolate visual anomalies against background clutter.
4. **Attention Generation:** Produces a normalized 2D saliency map $S \in [0, 1]^{H \times W}$ and blends it with the original input:
   $$I_{\text{enhanced}} = \alpha \cdot I_{\text{RGB}} + (1 - \alpha) \cdot (I_{\text{RGB}} \odot S)$$

### 3.2 Deep CNN Embeddings (Inception-ResNet-V2)
The network utilizes a frozen `inception_resnet_v2` backbone loaded via `timm`:
- **Stream 1 ($df_1$ - 1000-D):** Extracted from the standard input image $I_{\text{RGB}}$.
- **Stream 2 ($df_2$ - 1000-D):** Extracted from the attention-weighted representation $I_{\text{enhanced}}$ or converted saliency RGB.

### 3.3 Handcrafted Texture Descriptors (GLCM & LBP)
Disaster scenes feature texture patterns such as rubble granularity, flood water smoothness, and wildfire burn edges:
- **Gray-Level Co-occurrence Matrix (GLCM - 20-D):**
  - Evaluated at offsets $\delta \in \{1, 2, 3, 5\}$ across four orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.
  - Extracts 5 second-order statistical properties per matrix: Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
- **Local Binary Patterns (LBP - 16-D):**
  - Evaluates circular uniform patterns with radius $R=1, P=8$ and $R=2, P=16$.
  - Extracts sub-band statistical moments: Mean, Standard Deviation, Skewness, Kurtosis, Energy, and Texture Entropy.

### 3.4 Early Feature Fusion & Normalization
The individual feature representations are concatenated into a single feature vector:
$$\mathbf{x}_{\text{fused}} = \left[ \mathbf{df}_1 \,(1000) \;\parallel\; \mathbf{df}_2 \,(1000) \;\parallel\; \mathbf{glcm} \,(20) \;\parallel\; \mathbf{lbp}_{\text{stats}} \,(16) \right] \in \mathbb{R}^{2036}$$
The concatenated representation is standardized via `StandardScaler` fitted on the training split:
$$\mathbf{z} = \frac{\mathbf{x}_{\text{fused}} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

### 3.5 Damage Localization & DEM Scoring (Module M2)
Localization operates without requiring dense ground-truth segmentation masks:
1. **Activation Map:** Extracts saliency or class activation maps (Grad-CAM, Grad-CAM++, Score-CAM).
2. **Otsu / Adaptive Thresholding:** Binarizes the activation field into a candidate damage mask.
3. **Morphological Filtering:** Morphological opening and connected-component analysis discard small noise artifacts (area $< 0.5\%$).
4. **Damage Evaluation Metric (DEM):** Calculates a unified severity score:
   $$\text{DEM} = w_a \cdot A_{\text{affected}} + w_d \cdot D_{\text{damage}} + w_e \cdot H_{\text{entropy}} + w_c \cdot C_{\text{classifier}}$$
   - $A_{\text{affected}}$: Percentage of image pixels identified as damaged.
   - $D_{\text{damage}}$: Mean activation intensity inside detected damage polygons.
   - $H_{\text{entropy}}$: Texture entropy within damaged boundaries.
   - $C_{\text{classifier}}$: Confidence of the Random Forest damage severity prediction.

---

## 4. Prediction Classifiers & Optimization Methods

### 4.1 Classifiers & Loss / Weighting Mechanics
- **Ensemble Architecture:** `RandomForestClassifier` with $300$ estimators per objective (`max_features='sqrt'`).
- **Class Imbalance Mitigation:**
  - Standard maximum likelihood estimators over-index on dominant classes.
  - The model uses `class_weight='balanced'`, scaling weights inversely proportional to class frequencies:
    $$w_j = \frac{N}{K \cdot n_j}$$
    where $N$ is total samples, $K$ is number of classes, and $n_j$ is samples in class $j$.
  - This ensures minority perils (e.g., landslides, wildfires) receive balanced gradient updates alongside earthquakes and floods.
- **LRU In-Memory Checkpoint Bundle:**
  Models and scalers are loaded through `@lru_cache(maxsize=3)` with automatic dimension contract validation (`resolve_artifact_spec`), preventing redundant disk I/O while ensuring feature compatibility across 2030-dim legacy and 2036-dim modern pipelines.

### 4.2 Cross-Validation & Generalization Protocol (Stratified 5-Fold)
To prevent data leakage, evaluate out-of-fold generalization, and ensure class distribution consistency across splits, the training pipeline implements a **Stratified 5-Fold Cross-Validation** protocol (`sklearn.model_selection.StratifiedKFold`):
- **Stratification Requirement:** Due to acute imbalance across disaster perils (e.g., $28$ landslide test samples vs. $2,553$ non-disaster samples), random K-Fold would risk folds with zero minority-class instances. Stratified splitting preserves class proportions identically in every fold:
  $$\frac{n_{j,\text{fold}}}{N_{\text{fold}}} \approx \frac{n_j}{N}$$
- **Cross-Validation Execution:**
  - Configured with $K=5$ folds, `shuffle=True`, and seed `random_state=42` prior to the final fit on the training split.
  - Accuracy metrics are tracked across all 5 iterations:
    $$\mu_{\text{CV}} = \frac{1}{5} \sum_{k=1}^5 \text{Acc}_k, \quad \sigma_{\text{CV}} = \sqrt{\frac{1}{5} \sum_{k=1}^5 (\text{Acc}_k - \mu_{\text{CV}})^2}$$
  - Logged into MLflow tracking metadata as `cv_accuracy_mean` and `cv_accuracy_std`.
- **Validation Results:**
  - For `disaster_type`, 5-fold cross-validation yields **$68.06\% \pm 1.06\%$** ($[68.39\%, 68.47\%, 66.10\%, 69.30\%, 68.03\%]$), proving that the holdout test accuracy ($68.45\%$) reflects genuine out-of-sample generalization rather than split-dependent artifacts.

### 4.3 Computational & Training Optimizations
1. **Compressed Feature Matrix Caching (`.npz`):**
   CNN forward passes and texture computations across 18,103 dataset images are computed once and serialized to disk (`mlruns/2/.../damage_train_df1_df2_glcm_lbp_stats_features.npz`). Retraining classifiers takes seconds instead of hours.
2. **Parallel CPU Multi-Threading:**
   Batch inference and GLCM/LBP calculations leverage `joblib.Parallel(n_jobs=-1, backend="threading")`, distributing feature extraction across available CPU cores.
3. **Bounded Dedicated Thread Pools:**
   Inference in Flask and CrewAI runs in an isolated `ThreadPoolExecutor(max_workers=2)` (`pipeline_singleton.py`), preventing compute-heavy PyTorch passes from blocking HTTP request event loops or SSE streams.
4. **Adaptive Image Downsampling:**
   Images exceeding high resolutions are downsampled before feature extraction ($1280\text{px}$ analysis edge, $960\text{px}$ response edge) using bicubic filtering to bound memory and CPU footprint.

---

## 5. Agentic Framework Implementation (CrewAI & Deterministic Runner)

### 5.1 Architecture Overview: Is There a Main Agent?
In the DisasterRes-Net framework:
- **Orchestrator Pattern:** The system uses a centralized **Workflow Coordinator (`run_disaster_crew` in `crew/crew.py`)**.
- **Execution Modality:**
  - **Deterministic Engine (Primary & Active):** A state machine that executes specialized tools in sequence, generates structured evidence blocks, and enforces numeric validation.
  - **CrewAI Declarative Layer (`crew/agents.py`, `crew/tasks.py`):** Configures formal `crewai.Agent` personas and `crewai.Task` graphs for autonomous multi-agent environments.

```
                    +---------------------------------------+
                    |           run_disaster_crew           |
                    |        (Central Coordinator)          |
                    +---------------------------------------+
                                        |
       +--------------------------------+--------------------------------+
       |                                |                                |
       v                                v                                v
[Agent 1: Scout]             [Agent 2: Analyst]             [Agent 3: Writer]
Role: Intelligence Officer   Role: Vision & Metrics         Role: Communications Officer
Tools:                       Tools:                         Tools:
- Query Resolver             - Suitability Gate             - Deterministic Formatter
- ReliefWeb / GDACS          - Hybrid Pipeline Tool         - OpenAI LLM Narrator
- Image Downloader Tool      - Metric Aggregator            - Numeric Fidelity Gate
```

### 5.2 Subagent Taxonomy & Roles

| Agent Identifier | Operational Role | Backstory & Persona | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Disaster Intelligence Scout** | Intelligence & Reconnaissance Officer | Specialized emergency scout skilled in open-source intelligence and satellite catalog search. | Resolves geographical entities, identifies ongoing disaster events, coordinates bounding boxes, and scrapes candidate imagery. |
| **Damage Analyst** | Computer Vision & Quantitative Specialist | Rigorous spatial data analyst adhering to strict optical validation standards. | Enforces image suitability checks, triggers 2036-dim hybrid inference, calculates DEM scores, and aggregates damage statistics. |
| **Disaster Report Writer** | Emergency Communications Officer | Technical communications specialist drafting concise, fact-constrained disaster situation reports. | Synthesizes qualitative findings and quantitative facts into a structured Markdown SITREP without hallucinating data. |

### 5.3 Tooling Registry

- `ImageSuitabilityGate`: Evaluates image resolution, variance of Laplacian (blur check), and aspect ratio to filter unsuitable overhead imagery.
- `ReliefWebEventSearchTool`: Connects to UN OCHA ReliefWeb and GDACS APIs to retrieve disaster metadata.
- `GoogleImageScrapeTool` / `ImageDownloaderTool`: Searches and retrieves aerial and satellite disaster imagery.
- `DisasterResPipelineTool`: Invokes the frozen Inception-ResNet-V2, GLCM, and LBP feature extractors and Random Forest classifiers.
- `DamageAssessmentTool`: Generates connected-component masks, damage contours, and DEM scores.
- `MetricAggregatorTool`: Aggregates multi-image predictions into regional statistics and assigns confidence bands (`High`, `Medium`, `Moderate`).
- `NumericFidelityValidator`: Validates LLM-generated text against verified facts before delivering output to the user.

---

## 6. Current Storyline & Execution Lifecycle of an Agent

The system operates across two execution lifecycles based on user input:

```
MODE A: Conversational Query Flow
=================================
[User Prompt] 
      │ "Explain damage in Wayanad"
      ▼
[Scout Agent] ──► resolve_query() & geocode() ──► Extract coordinates
      │
      ▼
[Scout Agent] ──► Query ReliefWeb / GDACS / Serper.dev ──► Candidate Image URLs
      │
      ▼
[Analyst Agent] ─► ImageSuitabilityGate ──► Reject blurry / corrupt images
      │
      ▼
[Analyst Agent] ─► analyze_image() ──► 2036-D Fused Inference & M2 Localization
      │
      ▼
[Analyst Agent] ─► aggregate_assessments() ──► Compute Mean DEM & Confidence
      │
      ▼
[FactsBlock Assembly] ──► Construct verified ground-truth data payload
      │
      ▼
[Writer Agent] ──► Deterministic Template & LLM Narration (GPT-4o-mini)
      │
      ▼
[Numeric Gate] ──► Validate numbers against FactsBlock ──► (Fallback if invalid)
      │
      ▼
[SSE Output] ──► Stream report tokens to Vite Chat UI


MODE B: Direct Multimodal Evidence Flow
=======================================
[Image Upload + Prompt] 
      │ (Base64 JPEG/PNG + optional query)
      ▼
[Coordinator] ──► Decode buffer & check suitability gate
      │
      ▼
[Analyst Agent] ──► Fused Model Classification:
      │               • Predicts Peril (e.g., Wildfire)
      │               • Predicts Severity (e.g., Mild)
      │               • Computes DEM & Localization Overlays
      ▼
[Context Resolver] ──► Updates Scout event_type with model prediction
      │               • Sets location: "Uploaded Scene" (or parsed place)
      │               • Sets date: Current date
      ▼
[Writer Agent] ──► Generate Situation Report (SITREP)
      │
      ▼
[SSE Stream] ──► Stream Markdown SITREP + Overlays (DDM, Saliency, Mask)
```

### 6.1 Mode A: Conversational Query Flow (Autonomous Web/Satellite Ingestion)
1. **Intent & Location Resolution:**
   The user submits a prompt (e.g., *"Assess the recent flood impact in Wayanad"*). The Scout executes `resolve_query()` to isolate the target location (`Wayanad`) and disaster type (`flood`).
2. **Event & Coordinate Geocoding:**
   The Scout queries `resolve_event()` to fetch UN OCHA / GDACS crisis identifiers and retrieves latitude/longitude bounding boxes.
3. **Imagery Retrieval:**
   The Scout queries aerial image providers and search engines (`GoogleImageScrapeTool` / Serper.dev) to gather candidate disaster scene URLs.
4. **Suitability Gating:**
   The Analyst filters candidates through `ImageSuitabilityGate`. Images that are blurry, corrupted, or non-aerial are rejected.
5. **Computer Vision Inference:**
   Accepted images are processed through `analyze_image()`. The pipeline extracts $df_1$, $df_2$, GLCM, and LBP features, predicts peril and severity distributions, and generates damage masks.
6. **Multi-Image Evidence Aggregation:**
   The Analyst calls `aggregate_assessments()`, calculating mean DEM scores, affected area percentages, and setting the confidence band (`High`, `Medium`, `Moderate`).

### 6.2 Mode B: Direct Multimodal Evidence Flow (Upload Path)
1. **Direct Image Decoding:**
   The user uploads an image via the prompt box or file upload panel. Base64 bytes are decoded to RGB format.
2. **Autonomous Peril Inference:**
   If the user does not specify the peril or location, the image is passed to `HybridDisasterPipeline`. The classifier identifies the peril (`wildfire`, `flood`, `earthquake`, `hurricane`, `landslide`, or `not_disaster`) and severity rating (`severe`, `mild`, `little_or_none`).
3. **Metadata Synchronization:**
   The Coordinator populates the `ScoutResult` dataclass with the classifier's predicted peril, current date, and location context.

### 6.3 Hallucination Defense: The Numeric Fidelity Gate
To prevent LLM hallucinations during crisis reporting:
1. **FactsBlock Serialization:**
   All ground-truth metrics (DEM scores, affected area percentages, region counts, confidence percentages) are frozen in a `FactsBlock`.
2. **Numeric Extraction:**
   `facts.values()` extracts all valid numeric tokens formatted as integers, decimals, and percentages.
3. **LLM Narration & Auditing:**
   The LLM generates narrative text constrained strictly to the facts JSON.
4. **Validation Check:**
   `NumericFidelityValidator` scans every numeric token in the generated response using regex. If the LLM generates a metric that does not exist in `FactsBlock`, the text is rejected and the system falls back to the deterministic report.

---

## 7. Future Roadmap: Model, Training & Agentic Optimizations

### 7.1 Machine Learning & Computer Vision Enhancements

```
Current Architecture (Frozen Backbone + RF)
[Frozen Inception-ResNet-V2] ──► [2036-D Features] ──► [Random Forest Classifier]

Recommended Target Architecture (End-to-End Multimodal)
[Multi-Band Satellite Image] ──► [Swin-UNet / SegFormer] ──► Dense 4-Class Segmentation (xBD)
                                          │
                                          ▼
                               [LoRA-Adapted Vision LLM] ──► Autonomous Direct Narration
```

1. **Transition from Patch-Level Saliency to Dense Building Segmentation:**
   - *Current Limitation:* Saliency thresholds capture broad damage areas but cannot evaluate individual building footprints.
   - *Recommendation:* Train a **SegFormer (B3/B4)** or **Mask2Former** model on the **xBD dataset** to predict FEMA 4-class building damage masks (`no-damage`, `minor-damage`, `major-damage`, `destroyed`).
2. **End-to-End Backbone Fine-Tuning with Focal Loss:**
   - *Current Limitation:* Inception-ResNet-V2 weights remain frozen from ImageNet pretraining.
   - *Recommendation:* Unfreeze the top layers and train end-to-end with **Focal Loss** to address visual ambiguity between landslide mud, flood waters, and earthquake debris:
     $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
3. **Multi-Spectral & Multi-Sensor Fusion (SAR + Optical):**
   - *Current Limitation:* Relies solely on 3-channel RGB imagery, which is susceptible to cloud cover and nighttime constraints.
   - *Recommendation:* Integrate **Sentinel-1 Synthetic Aperture Radar (SAR)** (VV/VH polarization) with **Sentinel-2 multispectral bands** (NIR, RedEdge, SWIR). SAR penetrates cloud cover and smoke to detect water extent and structural collapse.
4. **Pre-Disaster / Post-Disaster Bi-Temporal Siamese Networks:**
   - *Recommendation:* Implement a Siamese architecture comparing registered pre-event and post-event imagery to calculate normalized difference vegetation indices ($\Delta\text{NDVI}$) and water indices ($\Delta\text{NDWI}$).

### 7.2 Agentic Framework Enhancements

1. **Multi-Agent Consensus & Debate Loops:**
   - Implement an autonomous **Debate Protocol**: If the Computer Vision Analyst flags `severe damage` but the Intelligence Scout finds ReliefWeb reports indicating only minor flooding, the agents can trigger a consensus iteration to request secondary imagery or adjust confidence bands.
2. **Direct Satellite STAC API Integration:**
   - Connect the Scout agent directly to SpatioTemporal Asset Catalog (STAC) endpoints (e.g., Planet Labs, Sentinel Hub, USGS EarthExplorer). When given a location and date, the agent can query satellite scenes, filter by cloud cover ($< 15\%$), and download tiles automatically.
3. **Spatial Vector Memory (RAG with GeoPandas & ChromaDB):**
   - Store historical SITREPs and regional baselines in a geospatial vector database. If a new disaster occurs in an area analyzed previously, the agent can compare damage levels against historical baselines.
4. **Dynamic Tool Calling & Self-Correcting Feedback:**
   - Equip the Scout with self-correcting query mechanisms: If an image search yields ground photos instead of satellite imagery, the agent detects the perspective mismatch via the suitability gate and adjusts search parameters to focus on overhead imagery.

