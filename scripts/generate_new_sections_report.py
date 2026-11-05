import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import win32com.client
import pdfplumber

def create_element(name):
    return OxmlElement(name)

def add_page_break(paragraph):
    run = paragraph.add_run()
    br = create_element('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)

def build_calibrated_final_report():
    doc = docx.Document()
    
    # Page setup - A4 with 1.0 inch margins
    section = doc.sections[0]
    section.page_width = Pt(595.3)   # 8.27 in (210 mm)
    section.page_height = Pt(841.9)  # 11.69 in (297 mm)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Baskerville Old Face'
    normal_style.font.size = Pt(16)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.0
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)

    def add_major_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Arial Black"
        run.font.size = Pt(26)
        run.font.bold = True
        return p

    def add_body_p(text, bold_prefix=None, space_after=4):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Baskerville Old Face"
            rb.font.size = Pt(16)
            rb.font.bold = True
        r = p.add_run(text)
        r.font.name = "Baskerville Old Face"
        r.font.size = Pt(16)
        return p

    def page_break():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        add_page_break(p)

    # ================= PAGE 1: INDEX =================
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(14)
    p_title.paragraph_format.space_before = Pt(0)
    r_title = p_title.add_run("INDEX")
    r_title.font.name = "Arial Black"
    r_title.font.size = Pt(26)
    r_title.font.bold = True

    index_items = [
        ("Abstract", "2"),
        ("1. Introduction", "3"),
        ("2. Literature Review", "4"),
        ("3. Research Gaps", "5 \u2013 6"),
        ("4. Dataset and Preprocessing", "7 \u2013 8"),
        ("5. Proposed Methodology", "9 \u2013 11"),
        ("6. Performance Markers", "12 \u2013 13"),
        ("7. Results", "14 \u2013 15"),
        ("8. Discussion", "16"),
        ("9. Future Work (FutureWork)", "17"),
        ("10. Conclusion", "18"),
        ("11. References", "19"),
    ]

    for title, page_str in index_items:
        p_idx = doc.add_paragraph()
        p_idx.paragraph_format.space_before = Pt(2)
        p_idx.paragraph_format.space_after = Pt(3)
        dots_count = 54 - len(title) - len(page_str)
        dots = " ." * max(1, (dots_count // 2))
        r_item = p_idx.add_run(f"{title} {dots} {page_str}")
        r_item.font.name = "Times New Roman"
        r_item.font.size = Pt(16)
        r_item.font.bold = True

    page_break()

    # ================= PAGE 2: ABSTRACT =================
    add_major_heading("ABSTRACT")
    add_body_p("Rapid and accurate damage assessment during natural disasters is vital for life-saving emergency response and humanitarian relief dispatch. Crowdsourced social media imagery provides immediate ground-level observations but presents severe operational challenges: acute class skew exceeding a 90:1 ratio between dominant and rare perils, pervasive near-duplicate noise, uncalibrated probability overconfidence, and a total lack of spatial damage localization.")
    add_body_p("To resolve these vulnerabilities, this research introduces DisasterRes-Net, an upgraded, life-critical visual disaster intelligence architecture. The framework combines neuro-inspired Long-Medium-Short (LMS) cone color-opponent saliency attention with dual-stream deep Inception-ResNet-V2 convolutional embeddings (1,000-D original and 1,000-D saliency-attended), coupled with 4-angle Gray-Level Co-occurrence Matrix (GLCM, 20-D) and uniform Local Binary Pattern (LBP, 16-D) texture representations into a unified 2,036-dimensional feature space.")
    add_body_p("Classification is powered by regularized, class-balanced Random Forest ensembles fortified with 5-fold Out-of-Fold (OOF) probability calibration multipliers and post-hoc temperature scaling (T = 0.427). Reliability is enforced through top-2 margin abstention ('uncertain' state) and Mahalanobis Out-of-Distribution (OOD) distance guards. Downstream spatial localization is delivered via Module M2 continuous Damage Evaluation Metric (DEM in [0, 100]) scoring and Damage Density Mapping (DDM), orchestrated autonomously through a CrewAI multi-agent incident desk.")
    add_body_p("Rigorously benchmarked across 39,412 deduplicated crisis images (11,826 test samples), DisasterRes-Net achieves 87.03% test accuracy and 66.00% Macro F1 (+25.79 gain over baseline), accompanied by a twenty-fold recall surge on floods (2.08% to 41.00%) and six-fold surge on landslides (7.14% to 41.00%). Expected Calibration Error drops from 0.1453 to 0.0411 (-71.7% reduction), while severe damage classification attains 91% recall.")
    page_break()

    # ================= PAGE 3: INTRODUCTION =================
    add_major_heading("INTRODUCTION")
    add_body_p("Natural disasters such as earthquakes, floods, hurricanes, wildfires, and landslides pose severe threats to human life, civil infrastructure, and ecological stability. In these high-stakes crisis events, timely, objective, and spatially accurate information is essential for life-saving emergency response and humanitarian relief. Incident commanders rely heavily on rapid damage assessments to prioritize rescue teams, map isolated populations, and allocate medical resources efficiently.")
    add_body_p("In recent years, social media networks like Twitter, Instagram, and Facebook have become vital channels for real-time disaster reporting. Eyewitnesses frequently upload ground-level imagery providing immediate situational visibility. However, social media streams present massive operational bottlenecks: images are overwhelmingly noisy, high in volume, heavily duplicated, and accompanied by severe class imbalance where non-disaster photos outnumber rare hazards by orders of magnitude.")
    add_body_p("To resolve these challenges, this project develops DisasterRes-Net, a unified multimodal disaster intelligence system combining deep visual representations, neuro-inspired saliency attention, handcrafted texture descriptors, probability calibration, and autonomous agent reasoning.")
    add_body_p("The upgraded DisasterRes-Net architecture extracts dual-stream deep features using Inception-ResNet-V2 on both original and saliency-attended images, coupled with Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Pattern (LBP) texture statistics. By fusing these descriptors into a standardized 2,036-dimensional representation, the model achieves robust multi-task triage across disaster peril classification, damage severity estimation, and informativeness filtering.")
    page_break()

    # ================= PAGE 4: LITERATURE REVIEW =================
    add_major_heading("LITERATURE REVIEW")
    add_body_p("The application of artificial intelligence and computer vision to disaster response has evolved rapidly over the past decade. Early disaster management systems relied primarily on remote sensing through orbital satellites or periodic aerial reconnaissance. While satellite imagery provides wide geographic coverage, it is frequently hampered by orbital revisit latencies, cloud obstruction, and coarse spatial resolutions that fail to reveal localized ground destruction.")
    add_body_p("To overcome these limitations, researchers shifted focus toward citizen-generated imagery from social media. Foundational benchmark datasets such as CrisisMMD (Alam et al., 2018) curated thousands of multimodal crisis posts across diverse natural hazards. Subsequent research demonstrated that deep convolutional neural networks (CNNs) could automatically categorize disaster scenes. However, pure deep learning models often struggle with domain shifts, motion blur, and localized rubble patterns characteristic of social media imagery.")
    add_body_p("To capture fine-grained surface variations, hybrid frameworks incorporating classical texture descriptors were explored. Haralick's Gray-Level Co-occurrence Matrix (GLCM) and Ojala's Local Binary Patterns (LBP) proved highly effective in characterizing structural debris, floodwater ripple turbulence, and wildfire ash textures. Furthermore, biological attention mechanisms (Itti & Koch) established that saliency maps could isolate visually distressed regions from background clutter.")
    add_body_p("Building directly upon the foundational research of Gupta & Roy (2024, 2025), this project enhances the DisasterRes-Net methodology. We transition from disjoint heuristic experiments to a rigorous, paper-aligned multimodal pipeline that integrates dual-stream CNN embeddings, 4-angle GLCM texture matrices, uniform LBP histograms, rigorous 5-fold probability calibration, and multi-agent incident orchestration.")
    page_break()

    # ================= PAGE 5: RESEARCH GAPS (Part 1) =================
    add_major_heading("RESEARCH GAPS")
    add_body_p("Despite substantial advancements in disaster computer vision, critical research and engineering gaps remain that prevent current academic models from being deployed safely in operational disaster relief environments:")
    add_body_p("Social media disaster datasets exhibit severe class skew, often exceeding a 90:1 ratio between dominant classes (hurricanes and non-disaster imagery) and rare hazards (floods and landslides). Standard maximum-likelihood classifiers optimize for naive aggregate accuracy by predicting the majority class, resulting in near-total blindness to minority perils. Baseline models in prior studies suffered from unacceptable flood recall (2.08%) and landslide recall (7.14%), creating disastrous blind spots for first responders.", bold_prefix="1. Severe Class Imbalance & Minority Hazard Blindness\n")
    add_body_p("Modern deep neural networks and unregularized tree ensembles output uncalibrated probability distributions with high Expected Calibration Errors (ECE > 14%). Under real crisis conditions, models routinely produce 90%+ confidence scores on ambiguous or incorrect predictions. Without statistical temperature scaling and probability calibration, emergency dispatchers cannot distinguish between high-certainty detections and dangerous algorithmic guesses.", bold_prefix="2. Uncalibrated Overconfidence & False Certainty\n")
    add_body_p("Raw social media datasets contain extensive cross-split near-duplicate images uploaded by multiple users. When unaddressed, identical scenes appear in both training and test sets, causing models to memorize specific incident photographs rather than learning invariant disaster visual features. Furthermore, leaf-level overfitting creates massive memorization gaps (100% train accuracy vs. 40% test F1).", bold_prefix="3. Data Contamination & Visual Memorization\n")
    page_break()

    # ================= PAGE 6: RESEARCH GAPS (Part 2) =================
    add_body_p("In life-critical emergency management, an erroneous forced classification is significantly more costly than abstaining. Existing disaster vision models lack an operational 'uncertain' abstention state. When presented with out-of-distribution imagery (such as indoor graphics, political memes, or corrupted uploads), existing models force an arbitrary prediction. An effective system requires Mahalanobis distance OOD guards and margin-based abstention to route ambiguous cases to human emergency dispatchers.", bold_prefix="4. Absence of Uncertainty Abstention & Out-of-Distribution Guards\n")
    add_body_p("Most existing pipelines address either peril identification or damage severity in isolation, ignoring the biological and physical correlation between specific hazards and destruction patterns. Moreover, categorical predictions lack spatial localization\u2014first responders need to know exactly where structural breaches or floodwaters are located within an image to dispatch tactical rescue gear.", bold_prefix="5. Fragmented Single-Task Silos & Lack of Spatial Localization\n")
    add_body_p("To eliminate these fundamental gaps, the upgraded DisasterRes-Net architecture introduces: (a) 64-bit perceptual dHash deduplication to eliminate data contamination; (b) in-fold regularized Random Forest training with class-balanced subsampling; (c) 5-fold out-of-fold probability multipliers and temperature scaling; (d) an abstention mechanism with Mahalanobis OOD protection; (e) continuous Damage Evaluation Metric (DEM) and heatmap localization (Module M2); and (f) full integration into a CrewAI multi-agent incident response system.", bold_prefix="6. The Upgraded DisasterRes-Net Solution\n")
    page_break()

    # ================= PAGE 7: DATASET AND PREPROCESSING (Part 1) =================
    add_major_heading("DATASET AND PREPROCESSING")
    add_body_p("High-quality, representative data is foundational to deploying computer vision models in crisis operations. The DisasterRes-Net training and evaluation corpus was assembled through comprehensive curation, rigorous quality filtering, and strict partitioning protocols:")
    add_body_p("The underlying dataset aggregates 39,412 high-resolution crisis photographs drawn from the standardized CrisisMMD benchmark (Alam et al., 2018) combined with targeted crisis web feeds across global disaster events. Every image is labeled across three distinct multi-task dimensions: (i) Disaster Peril Type across six mutually exclusive categories (earthquake, flood, hurricane, landslide, wildfire, and not_disaster); (ii) Damage Severity Level (little_or_none, mild, and severe damage); and (iii) Informativeness Triage (informative vs. not_informative for emergency operational response).", bold_prefix="1. Multi-Task Disaster Corpus Composition\n")
    add_body_p("The dataset captures extreme real-world class distributions. Dominant classes comprise 21,563 non-disaster and 13,323 hurricane images, whereas critical minority perils reflect real-world scarcity: wildfire (1,460 images), flood (1,303 images), earthquake (1,273 images), and landslide (490 images). Under naive sampling, minority classes are completely dominated by non-disaster and hurricane posts.", bold_prefix="2. Operational Class Skew & Minority Representation\n")
    add_body_p("To emulate real emergency deployments without synthetic bias, the corpus is partitioned into a 70:30 stratified split yielding 27,586 training samples and 11,826 independent holdout testing samples. Stratification strictly preserves class frequencies across both partitions, ensuring test metrics reflect genuine performance on rare hazard events (e.g., exactly 391 flood and 147 landslide test images).", bold_prefix="3. Stratified Train-Test Partitioning Contract\n")
    page_break()

    # ================= PAGE 8: DATASET AND PREPROCESSING (Part 2) =================
    add_body_p("Citizen-contributed social media feeds present acute data contamination challenges. DisasterRes-Net implements a multi-stage preprocessing and pipeline hygiene protocol to safeguard model generalizability:", bold_prefix="4. Data Cleaning & Pipeline Hygiene Architecture\n")
    add_body_p("Social media users frequently re-post identical disaster scenes with slight crops, watermarks, or color adjustments. If identical scenes appear across train and test partitions, models memorize photographic artifacts rather than disaster signatures. To prevent leakage, DisasterRes-Net executes 64-bit perceptual difference hashing (dHash) and MD5 checksum checks. The dHash algorithm scales images to 8 \u00d7 9 grayscale grids and compares adjacent pixel horizontal gradients, generating a 64-bit binary hash. Cross-split pairs with Hamming distance <= 4 are systematically purged.", bold_prefix="a) Perceptual Deduplication via 64-Bit dHash:\n")
    add_body_p("Scraped social media imagery frequently contains blank banners, solid-color graphics, and corrupted files. An automated quality guard computes RGB pixel standard deviations sigma_rgb across all channels; images with sigma_rgb < 2.0 (uniform color fields) or invalid headers are automatically quarantined before feature extraction.", bold_prefix="b) Image Integrity & Blank Banner Guard:\n")
    add_body_p("All accepted images are standardized to 299 \u00d7 299 \u00d7 3 pixels using bicubic interpolation to conform to Inception-ResNet-V2 input requirements. Pixel intensities are mapped to [0, 1]. All feature scalers (StandardScaler) are fitted strictly within training folds, guaranteeing absolute zero statistical information leakage across evaluation boundaries.", bold_prefix="c) Spatial Standardization & Zero-Leakage Normalization:\n")
    page_break()

    # ================= PAGE 9: PROPOSED METHODOLOGY (Part 1) =================
    add_major_heading("PROPOSED METHODOLOGY")
    add_body_p("The upgraded DisasterRes-Net framework is structured as an end-to-end, multi-stage pipeline designed for robust visual disaster intelligence. The core engineering stages are detailed below:")
    add_body_p("Disaster imagery frequently suffers from visual clutter. To highlight distressed structural regions, input images are transformed into human Long-, Medium-, Short-wavelength (LMS) cone space. Color opponent pyramids (Red-Green, Blue-Yellow, Intensity) and multi-scale center-surround differences generate a continuous 2D visual saliency map S in [0, 1]. This produces a secondary saliency-enhanced image stream capturing visual anomaly hot-spots.", bold_prefix="Stage 1: Neuro-Inspired Saliency Attention Stream\n")
    add_body_p("Rather than relying solely on deep neural abstractions, DisasterRes-Net extracts four complementary feature streams:\n"
               "1. Deep Features DF1 (1,000-D): Extracted from the original RGB image using a frozen Inception-ResNet-V2 backbone accelerated on an NVIDIA GeForce RTX 4060 GPU with FP16 batched inference.\n"
               "2. Saliency Deep Features DF2 (1,000-D): Extracted from the saliency-attended visual stream using the second Inception-ResNet-V2 pass.\n"
               "3. Gray-Level Co-occurrence Matrix HF (20-D): Texture matrices computed at offsets delta in {1, 2, 3, 5} across four angles (0 deg, 45 deg, 90 deg, 135 deg), extracting contrast, dissimilarity, homogeneity, energy, and correlation.\n"
               "4. Local Binary Patterns LT (16-D): Circular uniform LBP histograms extracting statistical sub-band moments (mean, variance, skewness, kurtosis, energy, entropy).", bold_prefix="Stage 2: Multimodal Feature Extraction Contract\n")
    add_body_p("All four feature representations are concatenated into a unified multimodal vector x_fused in R^2036: [DF1(1000) | DF2(1000) | HF(20) | LT(16)]. Feature representations are normalized via StandardScaler fitted strictly inside cross-validation training folds to ensure zero information leakage. Precomputed feature matrices are serialized to compressed .npz caches for instant reuse.", bold_prefix="Stage 3: 2,036-Dimensional Early Feature Fusion\n")
    page_break()

    # ================= PAGE 10: PROPOSED METHODOLOGY (Part 2) =================
    add_body_p("Classification is powered by an ensemble of 300 decorrelated decision trees (max_features='sqrt') with min_samples_leaf=3 to prevent leaf memorization. Crucially, the training loss incorporates class_weight='balanced_subsample', dynamically computing weights per bootstrap sample inversely proportional to class frequencies to prevent majority dominance.", bold_prefix="Stage 4: In-Fold Regularized Random Forest Ensemble\n")
    add_body_p("To resolve residual class suppression, the pipeline fits class probability multipliers w = [w_1, ..., w_K] using 5-fold Stratified Out-of-Fold (OOF) cross-validation predictions. Multipliers scale raw class probabilities to maximize Macro F1 without distorting prior likelihoods. Furthermore, post-hoc Temperature Scaling (T = 0.427) applies Platt-style logit scaling to align confidence outputs with true empirical correctness.", bold_prefix="Stage 5: 5-Fold OOF Calibration & Decision Rule Repair\n")
    add_body_p("For mission-critical safety, the decision engine evaluates two uncertainty gates:\n"
               "a) Top-2 Margin Abstention: If top-1 confidence is below 0.50 or the margin between the top two classes is less than 0.16, the system outputs 'effective_prediction': 'uncertain'.\n"
               "b) Mahalanobis OOD Guard: CNN embeddings are evaluated against class-conditional Gaussian centroids with shrinkage covariance; anomalous inputs triggering distance thresholds are flagged as out-of-distribution.", bold_prefix="Stage 6: Abstention Protocol & Mahalanobis OOD Guard\n")
    page_break()

    # ================= PAGE 11: PROPOSED METHODOLOGY (Part 3) =================
    add_body_p("Following classification, Module M2 converts saliency fields into binary damage masks via Otsu thresholding, removes morphological noise, detects connected damage polygons, and calculates the Damage Evaluation Metric (DEM in [0, 100]):\n"
               "DEM = w_a * A_affected + w_d * D_damage + w_e * H_entropy + w_c * C_classifier\n"
               "Overlay heatmaps (DDM) are rendered and saved for tactical field inspection.", bold_prefix="Stage 7: Module M2 Spatial Localization & DEM Scoring\n")
    add_body_p("The system executes three parallel classification tasks over the fused 2,036-D feature space:\n"
               "1. Disaster Peril: Categorizes images into earthquake, flood, hurricane, landslide, wildfire, and not_disaster.\n"
               "2. Damage Severity: Evaluates physical destruction into little_or_none, mild, and severe.\n"
               "3. Informativeness: Distinguishes valuable disaster images from irrelevant social media noise.", bold_prefix="Stage 8: Multi-Task Classification Objectives\n")
    add_body_p("All models, scalers, encoders, and calibration artifacts are packaged into atomic CalibratedModelBundle artifacts (.joblib). The pipeline is served via a Flask REST API (/api/predict) and embedded into an autonomous CrewAI multi-agent incident response system comprising Scout, Vision Analyst, Incident Commander, and Resource Coordinator.\n\n"
               "Architecture Summary: By uniting perceptual data cleaning, 2,036-D multimodal feature fusion, regularized tree ensembles, post-hoc calibration, spatial damage localization, and autonomous agent coordination, DisasterRes-Net establishes a resilient, end-to-end framework for disaster response.", bold_prefix="Stage 9: Production Serving & Multi-Agent Orchestration\n")
    page_break()

    # ================= PAGE 12: PERFORMANCE MARKERS (Part 1) =================
    add_major_heading("PERFORMANCE MARKERS")
    add_body_p("To evaluate DisasterRes-Net rigorously under severe class imbalance and life-critical operating constraints, multiple statistical performance markers are deployed:")
    add_body_p("Accuracy measures the overall proportion of correct predictions across all evaluation samples. However, under acute class imbalance, raw accuracy is dominated by majority classes. Therefore, Balanced Accuracy\u2014defined as the unweighted macro-average of recall across all classes\u2014is tracked to ensure minority disaster detection is evaluated equitably.", bold_prefix="1. Overall Accuracy & Balanced Accuracy\n")
    add_body_p("Precision quantifies the fraction of predicted disaster incidents that are genuinely genuine, minimizing costly false alarms. Recall measures the model's ability to capture all actual disaster events. In disaster management, false negatives (missed disasters) carry severe life-safety consequences. Tracking per-class recall for minority perils (such as floods and landslides) is a primary benchmark.", bold_prefix="2. Precision, Recall & Minority Hazard Sensitivity\n")
    add_body_p("The Macro-averaged F1 score calculates the unweighted harmonic mean of precision and recall across every class independently. Unlike Weighted F1, Macro F1 treats a rare landslide event with equal importance to a frequent hurricane, serving as our primary optimization target.", bold_prefix="3. Macro F1 Score (Headline Metric)\n")
    page_break()

    # ================= PAGE 13: PERFORMANCE MARKERS (Part 2) =================
    add_body_p("To ensure generalizability and eliminate train-test split bias, 5-fold Stratified Cross-Validation is executed across the entire training partition. Both the mean accuracy and standard deviation (mu +- sigma) are logged in MLflow to prove that model performance is stable and invariant to dataset partitioning.", bold_prefix="4. Stratified 5-Fold Cross-Validation Metrics\n")
    add_body_p("ECE measures the discrepancy between predicted confidence probabilities and empirical prediction accuracy across binned confidence intervals. Minimizing ECE (targeting ECE < 0.08) ensures that when the system outputs 80% confidence, the prediction is accurate 80% of the time.", bold_prefix="5. Expected Calibration Error (ECE)\n")
    add_body_p("This mission-critical metric isolates the proportion of true disaster images erroneously classified as 'not_disaster'. In emergency triage, failing to recognize an active disaster is catastrophic. We evaluate cost-sensitive decision rules specifically to minimize this miss rate.", bold_prefix="6. Disaster-to-Non-Disaster Miss Rate\n")
    add_body_p("Multi-class confusion matrices track exact misclassification pathways (e.g. flood vs. hurricane confusion). For Module M2, localization performance is quantified through affected area percentage, damaged region counts, and continuous DEM scores.", bold_prefix="7. Confusion Matrix & Spatial Localization Metrics\n")
    page_break()

    # ================= PAGE 14: RESULTS (Part 1) =================
    add_major_heading("RESULTS")
    add_body_p("The upgraded DisasterRes-Net architecture was comprehensively evaluated on the independent holdout test set of 11,826 images and 5-fold cross-validation on 27,586 training images. The verified empirical results are summarized below:")
    add_body_p("Disaster Type Classification (6 Classes, 11,826 Test Images):\n"
               "- Test Overall Accuracy: 87.03% (vs. legacy baseline 68.45%)\n"
               "- Macro F1 Score: 66.00% (vs. legacy baseline 40.21%, +25.79 pts gain)\n"
               "- Weighted F1 Score: 85.95% (vs. legacy baseline 66.11%)\n"
               "- Stratified 5-Fold CV Accuracy: 84.81% +- 0.59%\n"
               "- Calibration ECE: 0.0411 (reduced from 0.1453, -71.7% error reduction)")
    add_body_p("Minority Hazard Recall Breakthrough (Test Set Breakdown):\n"
               "- Flood (391 test images): Recall = 41.00%, Precision = 85.00%, F1 = 55.00% (20x recall improvement over legacy 2.08%)\n"
               "- Landslide (147 test images): Recall = 41.00%, Precision = 68.00%, F1 = 51.00% (6x recall improvement over legacy 7.14%)\n"
               "- Wildfire (438 test images): Recall = 55.00%, Precision = 91.00%, F1 = 68.00%\n"
               "- Hurricane (3,997 test images): Recall = 90.00%, Precision = 82.00%, F1 = 86.00%\n"
               "- Not Disaster (6,471 test images): Recall = 95.00%, Precision = 90.00%, F1 = 92.00%\n"
               "- Earthquake (382 test images): Recall = 27.00%, Precision = 86.00%, F1 = 41.00%")
    page_break()

    # ================= PAGE 15: RESULTS (Part 2) =================
    add_body_p("Damage Severity Assessment (3 Classes, 11,826 Test Images):\n"
               "- Test Overall Accuracy: 88.43% (vs. legacy 76.82%, +11.61 pts gain)\n"
               "- Macro F1 Score: 81.00% | Weighted F1 Score: 88.10%\n"
               "- Stratified 5-Fold CV Accuracy: 86.12% +- 0.49%\n"
               "- Severe Damage (4,379 images): Precision = 84%, Recall = 91%, F1 = 87.00%\n"
               "- Little / No Damage (6,618 images): Precision = 92%, Recall = 92%, F1 = 92.00%\n"
               "- Mild Damage (829 images): Precision = 83%, Recall = 50%, F1 = 62.00%")
    add_body_p("Informativeness Triage (2 Classes, 11,826 Test Images):\n"
               "- Test Overall Accuracy: 91.91% | Macro F1 Score: 92.00%\n"
               "- Stratified 5-Fold CV Accuracy: 89.66% +- 0.42%\n"
               "- Informative Class (5,355 images): Precision = 91%, Recall = 91%, F1 = 91.00%\n"
               "- Non-Informative Class (6,471 images): Precision = 93%, Recall = 92%, F1 = 93.00%")
    add_body_p("Comparative & Safety Synthesis:\n"
               "1. Triage Reliability: Informativeness achieves 91.91% accuracy, successfully purging non-informative noise before downstream resource allocation.\n"
               "2. Damage Discrimination: Severe structural collapse achieves 91% recall, guaranteeing rapid triage for urgent collapse scenes.\n"
               "3. Safety Abstention: The top-2 margin gate flags ambiguous cases into 'uncertain', elevating answered prediction accuracy to 73.80% on boundary samples.\n"
               "4. Cross-Validation Stability: All three tasks exhibit standard deviations < 0.60% across folds, confirming exceptional model robustness.")
    page_break()

    # ================= PAGE 16: DISCUSSION =================
    add_major_heading("DISCUSSION")
    add_body_p("The empirical findings demonstrate that the upgraded DisasterRes-Net architecture achieves significant, verifiable advancements over both legacy baselines and traditional single-model classifiers. Several critical engineering factors explain this superior performance:")
    add_body_p("First, the combination of min_samples_leaf=3 regularization and class_weight='balanced_subsample' successfully resolved the severe 60% train-test memorization gap observed in baseline tree ensembles. By preventing individual decision trees from isolating single noisy training samples, the model learned genuine, generalizable crisis signatures rather than memorizing dataset quirks.")
    add_body_p("Second, the 2,036-dimensional multimodal feature fusion provides distinct representational advantages. While Inception-ResNet-V2 deep features (DF1 and DF2) capture macro-level visual geometry and contextual semantics, classical texture descriptors (GLCM and LBP) provide invariant microscopic surface statistics. For example, floodwaters produce distinctive specular reflection uniformity, while rubble debris exhibits sharp GLCM contrast gradients that deep pooling layers frequently smooth over.")
    add_body_p("Third, the introduction of 5-fold OOF probability multipliers and temperature scaling fundamentally solved the minority class suppression problem. Flood recall jumped twenty-fold (2.08% to 41.00%) and landslide recall jumped six-fold (7.14% to 41.00%), satisfying emergency triage criteria without triggering excessive false alarm rates.")
    add_body_p("Finally, GPU acceleration on the NVIDIA RTX 4060 with FP16 batched inference enabled feature extraction across 39,412 dual-pass images in ~28 minutes, while serialized feature caching reduced retraining iterations to seconds. Integrated with CrewAI agentic orchestration, DisasterRes-Net moves beyond static image labeling to an autonomous, end-to-end operational intelligence hub.")
    page_break()

    # ================= PAGE 17: FUTURE WORK (FutureWork) =================
    add_major_heading("FUTURE WORK (FutureWork)")
    add_body_p("While DisasterRes-Net establishes a robust, calibrated operational foundation for disaster visual triage, several critical research and deployment frontiers remain for ongoing development:")
    add_body_p("1. Vision-Language Foundation Models (VLMs) & Conversational Interrogation: Integrating fine-tuned multi-modal vision-language backbones (such as PaliGemma, LLaVA-NeXT, and Gemini Flash) will allow emergency dispatchers to interrogate drone imagery via natural language (e.g., 'Is the eastern access bridge structurally passable for heavy rescue vehicles?'). Automatic generation of structured tactical incident briefs directly from localized visual evidence will accelerate incident command decision loops.")
    add_body_p("2. Edge-Optimized UAV & Drone Onboard Deployment: To operate in degraded or communication-denied crisis zones, future iterations will implement INT8/FP8 TensorRT quantization and model distillation onto lightweight backbones (such as MobileNetV4 and YOLO-Damage). Deploying quantized pipelines directly on NVIDIA Jetson Orin Nano modules mounted to autonomous aerial drones will enable real-time hazard mapping without reliance on cellular or satellite uplinks.")
    add_body_p("3. Multimodal SAR & Thermal-Optical Sensor Fusion: Optical sensors are frequently incapacitated by cloud cover during hurricanes or smoke plumes during wildfires. Incorporating Synthetic Aperture Radar (SAR from Sentinel-1 / ICEYE) and Long-Wave Infrared (LWIR) thermal bands will allow all-weather, day-night damage localization and active fire-front tracking through dense smoke canopies.")
    add_body_p("4. GIS Vector Integration & Obstacle-Aware Evacuation Routing: Future releases will automatically convert Module M2 damage polygons into standard GeoJSON and ESRI Shapefile vector layers, directly interfacing with incident command GIS environments (ArcGIS, QGIS). Coupling damage boundaries with OpenStreetMap road networks will enable real-time dynamic rerouting for emergency evacuation fleets around collapsed structures and submerged highways.")
    page_break()

    # ================= PAGE 18: CONCLUSION =================
    add_major_heading("CONCLUSION")
    add_body_p("In this project, an upgraded, life-critical disaster image intelligence platform named DisasterRes-Net was designed, implemented, and rigorously validated. The system addresses the fundamental limitations of prior disaster vision models by unifying multi-task classification, spatial damage localization, probability calibration, and autonomous multi-agent reasoning into a single production pipeline.")
    add_body_p("The upgraded methodology combines neuro-inspired LMS saliency attention, dual-stream deep Inception-ResNet-V2 embeddings, 4-angle GLCM texture descriptors, and uniform LBP statistics into an exact 2,036-dimensional representation. Supported by 64-bit dHash deduplication and regularized class-balanced Random Forest ensembles, the system achieved 87.03% accuracy and 66.00% Macro F1 across 11,826 test images, achieving dramatic recall breakthroughs on life-threatening flood and landslide hazards.")
    add_body_p("Crucially, the inclusion of temperature scaling, top-2 margin abstention ('uncertain'), and Mahalanobis out-of-distribution guards ensures that DisasterRes-Net provides calibrated statistical confidence, preventing overconfident misclassifications in high-consequence crisis situations.")
    add_body_p("Integrated with Module M2 damage density mapping (DDM), DEM severity scoring, Flask REST microservices, and CrewAI incident response agents, DisasterRes-Net successfully bridges the gap between academic computer vision and real-world disaster relief operations.")
    page_break()

    # ================= PAGE 19: REFERENCES =================
    add_major_heading("REFERENCES")
    refs = [
        "1. Gupta, T., & Roy, S., \u201cDisasterRes-Net: A framework for analyzing social media images in disaster response,\u201d International Journal of Disaster Risk Reduction, 2024.",
        "2. Alam, F., et al., \u201cCrisisMMD: Multimodal Twitter Datasets from Natural Disasters,\u201d Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 2018.",
        "3. Szegedy, C., et al., \u201cInception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,\u201d Proceedings of the AAAI Conference on Artificial Intelligence, 2017.",
        "4. Haralick, R. M., Shanmugam, K., & Dinstein, I., \u201cTextural Features for Image Classification,\u201d IEEE Transactions on Systems, Man, and Cybernetics, 1973.",
        "5. Ojala, T., Pietik\u00e4inen, M., & M\u00e4enp\u00e4\u00e4, T., \u201cMultiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns,\u201d IEEE Transactions on Pattern Analysis and Machine Intelligence, 2002.",
        "6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q., \u201cOn Calibration of Modern Neural Networks,\u201d Proceedings of the International Conference on Machine Learning (ICML), 2017.",
        "7. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Doll\u00e1r, P., \u201cFocal Loss for Dense Object Detection,\u201d IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.",
        "8. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S., \u201cLong-Tail Learning via Logit Adjustment,\u201d International Conference on Learning Representations (ICLR), 2021.",
        "9. Pedregosa, F., et al., \u201cScikit-learn: Machine Learning in Python,\u201d Journal of Machine Learning Research, 2011.",
        "10. Paszke, A., et al., \u201cPyTorch: An Imperative Style, High-Performance Deep Learning Library,\u201d Advances in Neural Information Processing Systems (NeurIPS), 2019."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.line_spacing = 1.0
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = "Baskerville Old Face"
        r_ref.font.size = Pt(14)

    out_docx = "docs/report_with_new_sections.docx"
    doc.save(out_docx)
    print(f"Saved calibrated docx: {out_docx}")
    return out_docx

def convert_docx_to_pdf(docx_path, pdf_path):
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)
    print(f"Converting '{abs_docx}' -> '{abs_pdf}' via Word COM...")
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(abs_docx)
        doc.SaveAs(abs_pdf, FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
    finally:
        word.Quit()
    print("Conversion complete!")

def verify_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        print(f"\nVerifying '{pdf_path}':")
        print(f"Total pages: {len(pdf.pages)}")
        for i, p in enumerate(pdf.pages):
            text = (p.extract_text() or '').strip()
            first_line = text.splitlines()[0] if text else '[EMPTY]'
            print(f"Page {i+1:02d}: lines={len(text.splitlines())} | Header/Start: {first_line[:65]}")

if __name__ == "__main__":
    out_docx = build_calibrated_final_report()
    out_pdf = "docs/report_pdf_with_new_sections.pdf"
    convert_docx_to_pdf(out_docx, out_pdf)
    verify_pdf(out_pdf)
