# QuCNet Research Explorer

**An Interactive Architecture, Reproducibility and Evaluation-Audit Workbench for Quantum-Inspired Thyroid Ultrasound Classification**

QuCNet Research Explorer is a standalone **mini-project** that makes an existing, completed thyroid-ultrasound reproduction study understandable and auditable through an interactive Streamlit application. It presents verified **aggregate research evidence**, source-linked model architectures, interactive educational computations, and methodological limitations. The application **does not train models, classify uploaded medical images, provide diagnoses, or demonstrate quantum advantage**.

## Quick start

Use Python 3.13 (the locally verified development environment). From this repository's root:

```powershell
py -3.13 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Open the local address displayed in the terminal (usually `http://localhost:8501`). A separately activated virtual environment is optional; the explicit executable ensures isolation.

**macOS/Linux:** create a virtual environment with `python3 -m venv .venv`, install via `.venv/bin/python -m pip install -r requirements.txt`, and launch via `.venv/bin/python -m streamlit run app.py`.

**Validation:**

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
& ".\.venv\Scripts\python.exe" -m pip check
```

The app reads only its bundled `data/` snapshots, `assets/` visuals and `provenance.json` at runtime. **A local checkout of the original research repository is not required to run the app.** Evidence-import scripts under `src/evidence/` are separate developer utilities and may reference a read-only local research checkout; they are **not executed by the public app**.

## Software contribution

The engineering contribution is an evidence-driven research-exploration interface, not a new classifier or an improvement to the reported experimental results.

| Module | What can be examined |
|---|---|
| **Research Overview** | Research question, saved historical results, protocol distinctions and provenance |
| **Interactive Architecture Lab** | Five distinct model/feature pathways; layer shapes; verified source paths; four-qubit synthetic feature experiment |
| **Dataset & Integrity Explorer** | Label missingness, TI-RADS distribution, ROI/case-ID partitions and duplicate audits |
| **Experimental Results Lab** | Confusion matrices, saved ROC/PR figures, classifier comparisons, fold-seed stability and labelled hypothetical metrics |
| **Reliability & Accuracy-Gap Analysis** | Paper/implementation differences, class failures, preprocessing, BatchNorm and fusion findings, with observations separated from hypotheses |
| **Evidence & Interpretation** | Source-backed walkthrough, explanations, discussion questions and an exported research briefing |

The educational circuit demonstration computes outputs for **synthetic 2 × 2 pixel patches**, not medical-image predictions. The confusion-matrix what-if controls perform **hypothetical arithmetic**, not inference or model retraining. Saved comparisons are historical evidence, not newly executed experiments.

## Research origin and provenance

- **Frozen read-only source:** [QCNN Thyroid Paper Reproduction](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction)
- **Source local Git commit recorded in this mini-project:** `e848013b9ff943d816a8c90840255233a5ae7b8d`
- **Machine-readable provenance:** [`provenance.json`](provenance.json), which includes relative source artifact paths and SHA-256 checksums for the initial evidence snapshot.
- **Saved application evidence:** `data/research_snapshot.json`, `data/extension_snapshot.json` and `data/dataset_integrity_snapshot.json`.
- **Original source README:** [README at the recorded commit](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/blob/e848013b9ff943d816a8c90840255233a5ae7b8d/README.md).

The original research code, data, figures, trained models, experiments, splits and conclusions are **not modified** by this repository. The source commit and snapshot hashes preserve the relationship between the two repositories, even if the upstream default branch changes.

### Source-to-feature map

Paths below are **relative to the frozen research repository**, not to this application:

| Evidence / feature | Primary research path |
|---|---|
| Fixed four-qubit mapping | `05_Source_Code/utilities/quantum_utils.py`; `05_Source_Code/phase5_quantum_filter.py` |
| Binary QuCNet | `05_Source_Code/phase6_qucnet_model.py`; `07_Results/phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json` |
| Six-class quantum CNN | `05_Source_Code/phase9_paper_tirads_model.py`; `07_Results/phase9_paper_tirads_class_weighted/test_metrics.json` |
| Six-class grayscale CNN | `05_Source_Code/phase10_classical_tirads_model.py`; `07_Results/phase10_classical_tirads_class_weighted/test_metrics.json` |
| ROI CNN | `05_Source_Code/phase14_train_roi_categorical_baseline.py`; `07_Results/phase14_roi_categorical_baseline/evaluation_metrics.json` |
| Case/ROI split audit | `07_Results/phase12_case_level_split/split_distribution.json`; `07_Results/phase13_roi_dataset/roi_dataset_distribution.json` |
| Matched classifiers and stability | `07_Results/exp_models_001/metrics.json`; `07_Results/exp_stability_001/metrics.json` |
| Fusion, preprocessing and BatchNorm | `07_Results/exp_fusion_001/metrics.json`; `07_Results/exp_preproc_001/metrics.json`; `07_Results/exp_bn_001/metrics.json` |

For individual source files, use the evidence links shown within the application. The mini-project's `src/evidence/build_*.py` scripts are audit utilities, not a new data-processing or training pipeline.

## Verified research findings — preserve evaluation context

The research findings below are **from different experiments and must not be treated as a common test set**.

| Historical evaluation | Reported result | Evaluation context |
|---|---:|---|
| Binary quantum-feature QuCNet | **54.29%** (38/70) | 70-image **validation-selected historical holdout**; TI-RADS-derived suspicion proxy |
| Binary classical CNN | **57.14%** | Historical binary classical baseline; validation-selected holdout |
| Six-class quantum QuCNet | **18.57%** (13/70) | Historical image-level comparison |
| Six-class grayscale CNN | **24.29%** (17/70) | Historical image-level comparison |
| Six-class majority reference | **27.14%** | Historical 70-image class distribution |
| Phase 14 ROI CNN | **5.63%** (4/71) | Separate 71-ROI evaluation |
| Phase 14 case aggregation | **4.55%** (2/44) | Separate 44-recorded-case-ID evaluation |

For the binary QuCNet, the confusion matrix is **TN=6, FP=25, FN=7, TP=32**, with **56.14% precision** (not accuracy), **82.05% recall**, **19.35% specificity**, **66.67% F1**, **ROC-AUC 0.6634** and **average precision 0.7487**. The **55.71% (39/70)** same-holdout majority accuracy is a **calculated reference**, not an originally reported model result.

In a distinct **development-only**, fivefold case-ID-grouped out-of-fold comparison on **203 recorded case IDs / 311 training ROIs**, classical and fixed-quantum 64-dimensional feature representations were compared across logistic regression, RBF-SVM, random forest and 7-NN. A fixed-quantum random forest achieved **31.03% (63/203)** in one seed-42 comparison; this does **not** establish quantum superiority. Five-seed stability comparisons were performed only for logistic regression and random forest.

Additional exploratory analyses observed image-only **45/203**, geometry-only **39/203**, and fused **46/203** correct cases (McNemar two-sided p=1.00 for image vs fusion); a separate BatchNorm diagnostic changed **historical validation** accuracy from **3.45% to 18.97%** without changing trainable weights, still below its **29.31%** majority reference. Neither result establishes a new test-set improvement.

## Architecture fidelity

These research pathways are **not interchangeable**:

1. **Phase 6, binary:** 128 × 128 grayscale → 2 × 2 non-overlapping patches → fixed four-qubit circuit → 64 × 64 × 4 feature tensor → Flatten → Dense 128/64 → two-class Softmax. RX input encoding, two fixed RY layers, ring CNOT gates and four Pauli-Z expectations; simulated with PennyLane `default.qubit` using a fixed seed.
2. **Phase 9, six-class:** 64 × 64 × 4 quantum features → Conv2D 16/pool → Conv2D 32/pool → Flatten → Dense 128/64 → six-class Softmax. Some hyperparameters were reconstruction choices because paper details were incomplete.
3. **Phase 10, classical six-class:** grayscale 128 × 128 resized to 64 × 64 → Conv2D 16/pool → Conv2D 32/pool → Flatten → Dense 128/64 → six-class Softmax.
4. **Phase 14 ROI CNN:** 64 × 64 grayscale ROI → training-only augmentation → convolution/BatchNorm/ReLU and pooling/dropout → global average pooling → Dense 64 → six-class Softmax.
5. **Later development extension:** classical 64-D versus fixed simulated quantum 64-D features with LR, RBF-SVM, RF and 7-NN. This is **not** the original QuCNet CNN.

## Data integrity, limitations and ethics

- The recovered inventory contained **480 JPG images**, **390 XML records**, **349 images with nonempty TI-RADS labels** and **131 without**. The strictly processed ROI study had **440 ROIs from 291 recorded case IDs**, partitioned into train 311/203, validation 58/44 and historical test 71/44 (ROIs/case IDs).
- The strict partitions were separated by **recorded case ID**, which does **not** establish patient independence. Earlier historical image-level partitions had recorded-case overlap.
- The binary target is a **TI-RADS-derived suspicion proxy**, **not pathology-confirmed diagnosis**. There was no proven independent compatible six-class external-validation cohort.
- The published paper and recovered reconstruction differ in available labels, sample selection, normalization, backend and some implementation details. The substantial accuracy gap **cannot be attributed to one proven cause**.
- Accuracy, classwise failure patterns and probability diagnostics do **not** demonstrate clinical utility, calibration for patient decisions or quantum advantage.
- Do not redistribute original medical JPG/XML data, individual case identifiers, case-level predictions/probabilities, raw arrays or model checkpoints. Heuristic XML privacy screening is **not** a privacy certificate; source-data redistribution rights remain unresolved.
- Figures must be reviewed individually before public publication to ensure they contain **only aggregate visual evidence** and no embedded raw ultrasound frames or identifying annotations.

## Reproducibility and test scope

The `tests/` suite checks synthetic quantum-gate calculations, research metric arithmetic, dataset consistency, page-navigation structure and evidence-flow logic. Tests do **not** validate clinical effectiveness, patient independence, exact paper reproduction or the existence of an independent external test cohort.

The UI is implemented in **Streamlit** with **Plotly**, **Pandas**, **NumPy**, **JSON** and source-linked evidence. TensorFlow, PennyLane, Qiskit, GPUs, checkpoints and raw DDTI images are **not required** to run the public app.

## Deployment

For **Streamlit Community Cloud**, deploy this separate repository, set the main file to **`app.py`**, and use its `requirements.txt`. No Dockerfile is needed on Community Cloud. Do not set any credentials or attach the original frozen repository. Complete the local tests and manual figure/privacy audit before public deployment.

An **optional** `Dockerfile` is included for a self-hosted demonstration or an appropriate Docker environment. It is not necessary for the free Streamlit Community Cloud option. The `.dockerignore` protects local environments, archives and backups from the build context. Building a Docker image is not required for the university demo.

## Presentation and interpretation

A suitable demonstration sequence is **research question → Phase 6 architecture → synthetic four-qubit calculation → dataset integrity → archived confusion matrices and classifiers → reliability/accuracy-gap analysis → evidence interpretation**. Explain that the interactive computations are educational and separate from historical trained-model results. The new contribution is the **traceable workbench**, not a new QuCNet accuracy result.

## Research reference

Original QuCNet research: **IEEE Access (2024)**, DOI **[10.1109/ACCESS.2024.3367806](https://doi.org/10.1109/ACCESS.2024.3367806)**. The published figures are descriptive of the original authors' protocol and are **not directly comparable** with this reconstruction's recovered dataset and evaluation designs.

---

**Status:** Local six-module implementation has been demonstrated in VS Code/Streamlit; automated tests have passed on the local project. Public GitHub publication, independent cloud deployment and full manual UI/privacy checks should not be marked complete until verified separately.
