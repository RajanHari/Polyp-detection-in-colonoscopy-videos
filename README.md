# An End-to-End Pipeline for Automatic Polyp Detection in Colonoscopy

> Snapshot README for a project report with code currently not a organized state. This file makes the repo understandable and runnable at a basic level while I refactor.

---

## TL;DR

Colorectal cancer screening via colonoscopy benefits from automated assistance. This project implements a **three-stage pipeline**:

1. **Frame Informativeness Classification** – filter out non-informative/blurry frames.
2. **Polyp Presence Classification** – binary classification (polyp / no-polyp) on informative frames.
3. **Polyp Segmentation** – pixel-level localization with a UNet-style model (MobileNetV2 encoder).

Full write‑up is in **`reports/Rajan_Final_Report.pdf`**.

---

## Repository Status

This repository is currently a **code dump** from coursework. It runs, but structure and interfaces are still being standardized. Expect duplicate utilities and ad‑hoc scripts. The near‑term roadmap below explains how I’ll tidy this up.

---

## Project Structure (current / evolving)

```text
project-root/
├── reports/
│   └── Rajan_Final_Report.pdf
├── data/                    # Place datasets here (see Data Setup)
├── notebooks/               # (optional) exploration notebooks
├── scripts/                 # training / eval scripts (may be mixed)
├── models/                  # saved weights / checkpoints (git-ignored)
├── requirements.txt         # if present; else see Dependencies
└── README.md (this file)
```

> **Note:** The code is not fully modularized yet (e.g., multiple entry points, relative imports). Run scripts **from the repo root** to avoid import errors.

---

## Quickstart


### 1) Install Dependencies


```bash
pip install -r requirements.txt
```


### 2) Running (current state)

Because scripts are still being consolidated, look in `scripts/` for files such as:

* `Unet_mobilenetv2_backbone.py`
* `Atention_unet.ipynb`



## Results (from the report)

Representative outcomes achieved in the report (test splits):

| Stage | Task                    | Representative Model                              | Metric(s)          | Result                                                  |
| ----: | ----------------------- | ------------------------------------------------- | ------------------ | ------------------------------------------------------- |
|     1 | Frame informativeness   | EfficientNet‑B0 / ResNet‑50 (ImageNet‑pretrained) | Accuracy / ROC‑AUC | 97.29% acc; AUC ≈ 0.998 (EfficientNet‑B0)               |
|     2 | Polyp presence (binary) | ResNet‑50 (fine‑tuned)                            | Accuracy / ROC‑AUC | 97.30% acc; AUC ≈ 0.998                                 |
|     3 | Polyp segmentation      | UNet (MobileNetV2 encoder)                        | Dice (test)        | ≈ 0.77 Dice; latency \~0.081s/img on the measured setup |

See **report** for dataset specifics, confusion matrices, and ROC curves.

---

## Data & Provenance

* **QA‑Polyp 2015** – images labeled as blurry/clear.
* **ASU–Mayo** – colonoscopy videos; frames extracted and labeled polyp/no‑polyp; segmentation GT used for Part 3.
* **CVC ClinicDB** – additional segmentation dataset for diversity.

> Dataset distribution and licenses vary; acquire from official sources and follow their terms. This repo does **not** redistribute datasets.

---

## Reproducibility (current scripts)

Until the refactor lands, use the ad‑hoc scripts in `scripts/`:

* **Stage 1:** train & evaluate informativeness classifier; output ROC curve, AUC, confusion matrix.
* **Stage 2:** train & evaluate polyp vs no‑polyp classifier.
* **Stage 3:** train UNet; report Dice/Precision/Recall; save example overlays and inference latency.


---

## Roadmap (near‑term)

* [ ] Restructure into `src/` with clear modules: `data/`, `models/`, `train/`, `eval/`, `viz/`.
* [ ] Single CLI (`python -m polyp_pipeline ...`) with subcommands for all stages.
* [ ] Colab notebook for demo on a few sample images.

---



## Report

The full technical report, figures, and detailed methodology are in:

**`reports/Rajan_Final_Report.pdf`**

Please cite the report in academic contexts when referencing these results.

---


## Acknowledgments

* ImageNet‑pretrained backbones (ResNet‑50, EfficientNet‑B0).
* UNet with MobileNetV2 encoder for segmentation.
* QA‑Polyp 2015, ASU–Mayo, and CVC ClinicDB datasets (used per their terms).

---