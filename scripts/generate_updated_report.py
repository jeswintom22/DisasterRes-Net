import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def create_element(name):
    return OxmlElement(name)

def add_page_break(paragraph):
    run = paragraph.add_run()
    br = create_element('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)

def build_calibrated_report():
    doc = docx.Document()
    
    # Page setup - A4 with 1.0 inch margins
    section = doc.sections[0]
    section.page_width = Pt(595.3)   # 8.27 in
    section.page_height = Pt(841.9)  # 11.69 in
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
        p.paragraph_format.space_after = Pt(10)
        p.paragraph_format.space_before = Pt(0)
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

    # ================= PAGE 1: INDEX =================
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(20)
    p_title.paragraph_format.space_before = Pt(0)
    r_title = p_title.add_run("INDEX")
    r_title.font.name = "Arial Black"
    r_title.font.size = Pt(26)
    r_title.font.bold = True

    index_items = [
        ("1. Introduction", "2"),
        ("2. Literature Review", "3"),
        ("3. Research Gaps", "4 \u2013 5"),
        ("4. Proposed Methodology and Data", "6 \u2013 9"),
        ("5. Performance Markers", "10 \u2013 11"),
        ("6. Results", "12 \u2013 13"),
        ("7. Discussion", "14"),
        ("8. Conclusion", "15"),
        ("9. References", "16"),
    ]

    for title, page_str in index_items:
        p_idx = doc.add_paragraph()
        p_idx.paragraph_format.space_before = Pt(4)
        p_idx.paragraph_format.space_after = Pt(4)
        dots_count = 55 - len(title) - len(page_str)
        dots = " ." * (dots_count // 2)
        r_item = p_idx.add_run(f"{title} {dots} {page_str}")
        r_item.font.name = "Times New Roman"
        r_item.font.size = Pt(18)
        r_item.font.bold = True

    p_br1 = doc.add_paragraph()
    add_page_break(p_br1)

    # ================= PAGE 2: INTRODUCTION =================
    add_major_heading("INTRODUCTION")
    add_body_p("Natural disasters such as earthquakes, floods, hurricanes, wildfires, and landslides pose severe threats to human life, civil infrastructure, and ecological stability. In these high-stakes crisis events, timely, objective, and spatially accurate information is essential for life-saving emergency response and humanitarian relief. Incident commanders rely heavily on rapid damage assessments to prioritize rescue teams, map isolated populations, and allocate medical resources efficiently.")
    add_body_p("In recent years, social media networks like Twitter, Instagram, and Facebook have become vital channels for real-time disaster reporting. Eyewitnesses frequently upload ground-level imagery providing immediate situational visibility. However, social media streams present massive operational bottlenecks: images are overwhelmingly noisy, high in volume, heavily duplicated, and accompanied by severe class imbalance where non-disaster photos outnumber rare hazards by orders of magnitude.")
    add_body_p("To resolve these challenges, this project develops DisasterRes-Net, a unified multimodal disaster intelligence system combining deep visual representations, neuro-inspired saliency attention, handcrafted texture descriptors, probability calibration, and autonomous agent reasoning.")
    add_body_p("The upgraded DisasterRes-Net architecture extracts dual-stream deep features using Inception-ResNet-V2 on both original and saliency-attended images, coupled with Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Pattern (LBP) texture statistics. By fusing these descriptors into a standardized 2,036-dimensional representation, the model achieves robust multi-task triage across disaster peril classification, damage severity estimation, and informativeness filtering.")

    p_br2 = doc.add_paragraph()
    add_page_break(p_br2)

    # ================= PAGE 3: LITERATURE REVIEW =================
    add_major_heading("LITERATURE REVIEW")
    add_body_p("The application of artificial intelligence and computer vision to disaster response has evolved rapidly over the past decade. Early disaster management systems relied primarily on remote sensing through orbital satellites or periodic aerial reconnaissance. While satellite imagery provides wide geographic coverage, it is frequently hampered by orbital revisit latencies, cloud obstruction, and coarse spatial resolutions that fail to reveal localized ground destruction.")
    add_body_p("To overcome these limitations, researchers shifted focus toward citizen-generated imagery from social media. Foundational benchmark datasets such as CrisisMMD (Alam et al., 2018) curated thousands of multimodal crisis posts across diverse natural hazards. Subsequent research demonstrated that deep convolutional neural networks (CNNs) could automatically categorize disaster scenes. However, pure deep learning models often struggle with domain shifts, motion blur, and localized rubble patterns characteristic of social media imagery.")
    add_body_p("To capture fine-grained surface variations, hybrid frameworks incorporating classical texture descriptors were explored. Haralick's Gray-Level Co-occurrence Matrix (GLCM) and Ojala's Local Binary Patterns (LBP) proved highly effective in characterizing structural debris, floodwater ripple turbulence, and wildfire ash textures. Furthermore, biological attention mechanisms (Itti & Koch) established that saliency maps could isolate visually distressed regions from background clutter.")
    add_body_p("Building directly upon the foundational research of Gupta & Roy (2024, 2025), this project enhances the DisasterRes-Net methodology. We transition from disjoint heuristic experiments to a rigorous, paper-aligned multimodal pipeline that integrates dual-stream CNN embeddings, 4-angle GLCM texture matrices, uniform LBP histograms, rigorous 5-fold probability calibration, and multi-agent incident orchestration.")

    p_br3 = doc.add_paragraph()
    add_page_break(p_br3)

    # ================= PAGE 4: RESEARCH GAPS (Part 1) =================
    add_major_heading("RESEARCH GAPS")
    add_body_p("Despite substantial advancements in disaster computer vision, critical research and engineering gaps remain that prevent current academic models from being deployed safely in operational disaster relief environments:")
    add_body_p("Social media disaster datasets exhibit severe class skew, often exceeding a 90:1 ratio between dominant classes (hurricanes and non-disaster imagery) and rare hazards (floods and landslides). Standard maximum-likelihood classifiers optimize for naive aggregate accuracy by predicting the majority class, resulting in near-total blindness to minority perils. Baseline models in prior studies suffered from unacceptable flood recall (2.08%) and landslide recall (7.14%), creating disastrous blind spots for first responders.", bold_prefix="1. Severe Class Imbalance & Minority Hazard Blindness\n")
    add_body_p("Modern deep neural networks and unregularized tree ensembles output uncalibrated probability distributions with high Expected Calibration Errors (ECE > 14%). Under real crisis conditions, models routinely produce 90%+ confidence scores on ambiguous or incorrect predictions. Without statistical temperature scaling and probability calibration, emergency dispatchers cannot distinguish between high-certainty detections and dangerous algorithmic guesses.", bold_prefix="2. Uncalibrated Overconfidence & False Certainty\n")
    add_body_p("Raw social media datasets contain extensive cross-split near-duplicate images uploaded by multiple users. When unaddressed, identical scenes appear in both training and test sets, causing models to memorize specific incident photographs rather than learning invariant disaster visual features. Furthermore, leaf-level overfitting creates massive memorization gaps (100% train accuracy vs. 40% test F1).", bold_prefix="3. Data Contamination & Visual Memorization\n")

    p_br4 = doc.add_paragraph()
    add_page_break(p_br4)

    # ================= PAGE 5: RESEARCH GAPS (Part 2) =================
    add_body_p("In life-critical emergency management, an erroneous forced classification is significantly more costly than abstaining. Existing disaster vision models lack an operational 'uncertain' abstention state. When presented with out-of-distribution imagery (such as indoor graphics, political memes, or corrupted uploads), existing models force an arbitrary prediction. An effective system requires Mahalanobis distance OOD guards and margin-based abstention to route ambiguous cases to human emergency dispatchers.", bold_prefix="4. Absence of Uncertainty Abstention & Out-of-Distribution Guards\n")
    add_body_p("Most existing pipelines address either peril identification or damage severity in isolation, ignoring the biological and physical correlation between specific hazards and destruction patterns. Moreover, categorical predictions lack spatial localization—first responders need to know exactly where structural breaches or floodwaters are located within an image to dispatch tactical rescue gear.", bold_prefix="5. Fragmented Single-Task Silos & Lack of Spatial Localization\n")
    add_body_p("To eliminate these fundamental gaps, the upgraded DisasterRes-Net architecture introduces: (a) 64-bit perceptual dHash deduplication to eliminate data contamination; (b) in-fold regularized Random Forest training with class-balanced subsampling; (c) 5-fold out-of-fold probability multipliers and temperature scaling; (d) an abstention mechanism with Mahalanobis OOD protection; (e) continuous Damage Evaluation Metric (DEM) and heatmap localization (Module M2); and (f) full integration into a CrewAI multi-agent incident response system.", bold_prefix="6. The Upgraded DisasterRes-Net Solution\n")

    p_br5 = doc.add_paragraph()
    add_page_break(p_br5)

    # ================= PAGE 6: PROPOSED METHODOLOGY AND DATA (Part 1) =================
    add_major_heading("PROPOSED METHODOLOGY AND DATA")
    add_body_p("The upgraded DisasterRes-Net framework is structured as an end-to-end, multi-stage pipeline designed for robust visual disaster intelligence. The core engineering stages are detailed below:")
    add_body_p("The system operates on an expanded dataset comprising 39,412 disaster images collected from CrisisMMD and targeted crisis web streams across six hazard categories: earthquake, flood, hurricane, landslide, wildfire, and non-disaster. The data is partitioned into a 70:30 stratified split yielding 27,586 training images and 11,826 holdout testing images, preserving identical class proportions across both sets.", bold_prefix="Stage 1: Scaled Data Ingestion & Partitioning\n")
    add_body_p("To guarantee zero cross-split leakage, the ingestion engine implements 64-bit perceptual difference hashing (dHash) alongside MD5 checksums, eliminating exact and near-duplicate pairs across splits. In addition, an automated image quality guard computes RGB pixel standard deviations, filtering out corrupted files and solid-color blank filler banners (std < 2.0). All valid images are standardized to 299 \u00d7 299 pixels using bicubic interpolation.", bold_prefix="Stage 2: Perceptual Deduplication & Pipeline Hygiene\n")
    add_body_p("Disaster imagery frequently suffers from visual clutter. To highlight distressed structural regions, input images are transformed into human Long-, Medium-, Short-wavelength (LMS) cone space. Color opponent pyramids (Red-Green, Blue-Yellow, Intensity) and multi-scale center-surround differences generate a continuous 2D visual saliency map S in [0, 1]. This produces a secondary saliency-enhanced image stream capturing visual anomaly hot-spots.", bold_prefix="Stage 3: Neuro-Inspired Saliency Attention Stream\n")

    p_br6 = doc.add_paragraph()
    add_page_break(p_br6)

    # ================= PAGE 7: PROPOSED METHODOLOGY AND DATA (Part 2) =================
    add_body_p("Rather than relying solely on deep neural abstractions, DisasterRes-Net extracts four complementary feature streams:", bold_prefix="Stage 4: Multimodal Feature Extraction Contract\n")
    add_body_p("1. Deep Features DF1 (1,000-D): Extracted from the original RGB image using a frozen Inception-ResNet-V2 backbone accelerated on an NVIDIA GeForce RTX 4060 GPU with FP16 batched inference.\n"
               "2. Saliency Deep Features DF2 (1,000-D): Extracted from the saliency-attended visual stream using the second Inception-ResNet-V2 pass.\n"
               "3. Gray-Level Co-occurrence Matrix HF (20-D): Texture matrices computed at offsets delta in {1, 2, 3, 5} across four angles (0 deg, 45 deg, 90 deg, 135 deg), extracting contrast, dissimilarity, homogeneity, energy, and correlation.\n"
               "4. Local Binary Patterns LT (16-D): Circular uniform LBP histograms extracting statistical sub-band moments (mean, variance, skewness, kurtosis, energy, entropy).")
    add_body_p("All four feature representations are concatenated into a unified multimodal vector x_fused in R^2036: [DF1(1000) | DF2(1000) | HF(20) | LT(16)]. Feature representations are normalized via StandardScaler fitted strictly inside cross-validation training folds to ensure zero information leakage. Precomputed feature matrices are serialized to compressed .npz caches for instant reuse.", bold_prefix="Stage 5: 2,036-Dimensional Early Feature Fusion\n")
    add_body_p("Classification is powered by an ensemble of 300 decorrelated decision trees (max_features='sqrt') with min_samples_leaf=3 to prevent leaf memorization. Crucially, the training loss incorporates class_weight='balanced_subsample', dynamically computing weights per bootstrap sample inversely proportional to class frequencies to prevent majority dominance.", bold_prefix="Stage 6: In-Fold Regularized Random Forest Ensemble\n")

    p_br7 = doc.add_paragraph()
    add_page_break(p_br7)

    # ================= PAGE 8: PROPOSED METHODOLOGY AND DATA (Part 3) =================
    add_body_p("To resolve residual class suppression, the pipeline fits class probability multipliers w = [w_1, ..., w_K] using 5-fold Stratified Out-of-Fold (OOF) cross-validation predictions. Multipliers scale raw class probabilities to maximize Macro F1 without distorting prior likelihoods. Furthermore, post-hoc Temperature Scaling (T = 0.427) applies Platt-style logit scaling to align confidence outputs with true empirical correctness.", bold_prefix="Stage 7: 5-Fold OOF Calibration & Decision Rule Repair\n")
    add_body_p("For mission-critical safety, the decision engine evaluates two uncertainty gates:\n"
               "a) Top-2 Margin Abstention: If top-1 confidence is below 0.50 or the margin between the top two classes is less than 0.16, the system outputs 'effective_prediction': 'uncertain'.\n"
               "b) Mahalanobis OOD Guard: CNN embeddings are evaluated against class-conditional Gaussian centroids with shrinkage covariance; anomalous inputs triggering distance thresholds are flagged as out-of-distribution.", bold_prefix="Stage 8: Abstention Protocol & Mahalanobis OOD Guard\n")
    add_body_p("Following classification, Module M2 converts saliency fields into binary damage masks via Otsu thresholding, removes morphological noise, detects connected damage polygons, and calculates the Damage Evaluation Metric (DEM in [0, 100]):\n"
               "DEM = w_a * A_affected + w_d * D_damage + w_e * H_entropy + w_c * C_classifier\n"
               "Overlay heatmaps (DDM) are rendered and saved for tactical field inspection.", bold_prefix="Stage 9: Module M2 Spatial Localization & DEM Scoring\n")

    p_br8 = doc.add_paragraph()
    add_page_break(p_br8)

    # ================= PAGE 9: PROPOSED METHODOLOGY AND DATA (Part 4) =================
    add_body_p("The system executes three parallel classification tasks over the fused 2,036-D feature space:\n"
               "1. Disaster Type (Peril): Categorizes images into earthquake, flood, hurricane, landslide, wildfire, and not_disaster.\n"
               "2. Damage Severity: Evaluates physical destruction into little_or_none, mild, and severe.\n"
               "3. Informativeness Triage: Distinguishes operationally valuable disaster images from irrelevant social media noise (informative vs. not_informative).", bold_prefix="Stage 10: Multi-Task Classification Objectives\n")
    add_body_p("All models, scalers, encoders, and calibration artifacts are packaged into atomic CalibratedModelBundle artifacts (.joblib). The pipeline is served via a Flask REST API (/api/predict) and embedded into an autonomous CrewAI multi-agent incident response system:\n"
               "- Scout Agent: Ingests visual feeds and validates query resolution.\n"
               "- Vision Analyst: Executes hybrid inference, DEM scoring, and OOD checks.\n"
               "- Incident Commander: Assesses peril-severity matrices and triggers protocols.\n"
               "- Resource Coordinator: Formulates resource allocation recommendations.", bold_prefix="Stage 11: Production Serving & Multi-Agent Orchestration\n")
    add_body_p("By uniting perceptual data cleaning, 2,036-D multimodal feature fusion, regularized tree ensembles, post-hoc calibration, spatial damage localization, and autonomous agent coordination, DisasterRes-Net establishes a resilient, end-to-end framework for disaster response.", bold_prefix="Architecture Summary\n")

    p_br9 = doc.add_paragraph()
    add_page_break(p_br9)

    # ================= PAGE 10: PERFORMANCE MARKERS (Part 1) =================
    add_major_heading("PERFORMANCE MARKERS")
    add_body_p("To evaluate DisasterRes-Net rigorously under severe class imbalance and life-critical operating constraints, multiple statistical performance markers are deployed:")
    add_body_p("Accuracy measures the overall proportion of correct predictions across all evaluation samples. However, under acute class imbalance, raw accuracy is dominated by majority classes. Therefore, Balanced Accuracy\u2014defined as the unweighted macro-average of recall across all classes\u2014is tracked to ensure minority disaster detection is evaluated equitably.", bold_prefix="1. Overall Accuracy & Balanced Accuracy\n")
    add_body_p("Precision quantifies the fraction of predicted disaster incidents that are genuinely genuine, minimizing costly false alarms. Recall measures the model's ability to capture all actual disaster events. In disaster management, false negatives (missed disasters) carry severe life-safety consequences. Tracking per-class recall for minority perils (such as floods and landslides) is a primary benchmark.", bold_prefix="2. Precision, Recall & Minority Hazard Sensitivity\n")
    add_body_p("The Macro-averaged F1 score calculates the unweighted harmonic mean of precision and recall across every class independently. Unlike Weighted F1, Macro F1 treats a rare landslide event with equal importance to a frequent hurricane, serving as our primary optimization target.", bold_prefix="3. Macro F1 Score (Headline Metric)\n")

    p_br10 = doc.add_paragraph()
    add_page_break(p_br10)

    # ================= PAGE 11: PERFORMANCE MARKERS (Part 2) =================
    add_body_p("To ensure generalizability and eliminate train-test split bias, 5-fold Stratified Cross-Validation is executed across the entire training partition. Both the mean accuracy and standard deviation (mu +- sigma) are logged in MLflow to prove that model performance is stable and invariant to dataset partitioning.", bold_prefix="4. Stratified 5-Fold Cross-Validation Metrics\n")
    add_body_p("ECE measures the discrepancy between predicted confidence probabilities and empirical prediction accuracy across binned confidence intervals. Minimizing ECE (targeting ECE < 0.08) ensures that when the system outputs 80% confidence, the prediction is accurate 80% of the time.", bold_prefix="5. Expected Calibration Error (ECE)\n")
    add_body_p("This mission-critical metric isolates the proportion of true disaster images erroneously classified as 'not_disaster'. In emergency triage, failing to recognize an active disaster is catastrophic. We evaluate cost-sensitive decision rules specifically to minimize this miss rate.", bold_prefix="6. Disaster-to-Non-Disaster Miss Rate\n")
    add_body_p("Multi-class confusion matrices track exact misclassification pathways (e.g. flood vs. hurricane confusion). For Module M2, localization performance is quantified through affected area percentage, damaged region counts, and continuous DEM scores.", bold_prefix="7. Confusion Matrix & Spatial Localization Metrics\n")

    p_br11 = doc.add_paragraph()
    add_page_break(p_br11)

    # ================= PAGE 12: RESULTS (Part 1) =================
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

    p_br12 = doc.add_paragraph()
    add_page_break(p_br12)

    # ================= PAGE 13: RESULTS (Part 2) =================
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

    p_br13 = doc.add_paragraph()
    add_page_break(p_br13)

    # ================= PAGE 14: DISCUSSION =================
    add_major_heading("DISCUSSION")
    add_body_p("The empirical findings demonstrate that the upgraded DisasterRes-Net architecture achieves significant, verifiable advancements over both legacy baselines and traditional single-model classifiers. Several critical engineering factors explain this superior performance:")
    add_body_p("First, the combination of min_samples_leaf=3 regularization and class_weight='balanced_subsample' successfully resolved the severe 60% train-test memorization gap observed in baseline tree ensembles. By preventing individual decision trees from isolating single noisy training samples, the model learned genuine, generalizable crisis signatures rather than memorizing dataset quirks.")
    add_body_p("Second, the 2,036-dimensional multimodal feature fusion provides distinct representational advantages. While Inception-ResNet-V2 deep features (DF1 and DF2) capture macro-level visual geometry and contextual semantics, classical texture descriptors (GLCM and LBP) provide invariant microscopic surface statistics. For example, floodwaters produce distinctive specular reflection uniformity, while rubble debris exhibits sharp GLCM contrast gradients that deep pooling layers frequently smooth over.")
    add_body_p("Third, the introduction of 5-fold OOF probability multipliers and temperature scaling fundamentally solved the minority class suppression problem. Flood recall jumped twenty-fold (2.08% to 41.00%) and landslide recall jumped six-fold (7.14% to 41.00%), satisfying emergency triage criteria without triggering excessive false alarm rates.")
    add_body_p("Finally, GPU acceleration on the NVIDIA RTX 4060 with FP16 batched inference enabled feature extraction across 39,412 dual-pass images in ~28 minutes, while serialized feature caching reduced retraining iterations to seconds. Integrated with CrewAI agentic orchestration, DisasterRes-Net moves beyond static image labeling to an autonomous, end-to-end operational intelligence hub.")

    p_br14 = doc.add_paragraph()
    add_page_break(p_br14)

    # ================= PAGE 15: CONCLUSION =================
    add_major_heading("CONCLUSION")
    add_body_p("In this project, an upgraded, life-critical disaster image intelligence platform named DisasterRes-Net was designed, implemented, and rigorously validated. The system addresses the fundamental limitations of prior disaster vision models by unifying multi-task classification, spatial damage localization, probability calibration, and autonomous multi-agent reasoning into a single production pipeline.")
    add_body_p("The upgraded methodology combines neuro-inspired LMS saliency attention, dual-stream deep Inception-ResNet-V2 embeddings, 4-angle GLCM texture descriptors, and uniform LBP statistics into an exact 2,036-dimensional representation. Supported by 64-bit dHash deduplication and regularized class-balanced Random Forest ensembles, the system achieved 87.03% accuracy and 66.00% Macro F1 across 11,826 test images, achieving dramatic recall breakthroughs on life-threatening flood and landslide hazards.")
    add_body_p("Crucially, the inclusion of temperature scaling, top-2 margin abstention ('uncertain'), and Mahalanobis out-of-distribution guards ensures that DisasterRes-Net provides calibrated statistical confidence, preventing overconfident misclassifications in high-consequence crisis situations.")
    add_body_p("Integrated with Module M2 damage density mapping (DDM), DEM severity scoring, Flask REST microservices, and CrewAI incident response agents, DisasterRes-Net bridges the gap between academic computer vision and real-world disaster relief operations.")
    add_body_p("Future work will focus on integrating multi-modal Vision-Language Models (VLMs) for conversational drone interrogation, fine-tuning edge-optimized lightweight backbones for UAV onboard deployment, and establishing direct georeferenced GIS mapping overlays for live search-and-rescue command centers.")

    p_br15 = doc.add_paragraph()
    add_page_break(p_br15)

    # ================= PAGE 16: REFERENCES =================
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

    out_docx = "docs/updated_report.docx"
    doc.save(out_docx)
    print(f"Saved calibrated docx: {out_docx}")
    return out_docx

if __name__ == "__main__":
    build_calibrated_report()
