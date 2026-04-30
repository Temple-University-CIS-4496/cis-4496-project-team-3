# Narrative Homogenization in Hollywood Films
### CIS-4496 Data Science Capstone · Team 3 · Temple University · 2026

> **Are Hollywood films telling the same story?**  
> A computational analysis of narrative convergence across 2,017 film scripts spanning 1920–2026.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Research Question](#research-question)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Metrics & Methods](#metrics--methods)
- [Models](#models)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Team](#team)

---

## Project Overview

This project investigates whether Hollywood films have become measurably more similar in narrative structure, emotional arc, and thematic vocabulary over the past four decades. Using computational text analysis on 2,017 film scripts, we apply five independent metrics to test the hypothesis that modern films (2010–present) have converged toward a shared narrative template compared to the baseline era (1980–1995).

The analysis spans the full data science lifecycle: data acquisition and cleaning, feature engineering, unsupervised and supervised modeling, statistical validation, and results documentation.

---

## Research Question

> *Are films from the modern era (2010–present) measurably more similar in narrative structure, emotional arc, and thematic vocabulary compared to films from the baseline era (1980–1995)?*

**Hypothesis:** Films from the modern era show significantly higher narrative arc similarity, vocabulary convergence, and structural predictability compared to the pre-franchise baseline era.

---

## Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Arc similarity increase | **+43.7%** | Modern vs baseline — exceeds 20% charter threshold by 2.2× |
| Mann-Whitney U p-value | **0.0000** | Statistically significant at α = 0.001 |
| BERT KMeans ARI | **0.0073** | Unsupervised clustering cannot find era groups |
| BERT cross-era p-value | **0.0000** | Modern films significantly closer to each other than to baseline |
| Rolling window decline | **6.8%** | Pairwise distance 0.9895 → 0.9220 from 1987–2016 |
| RF weighted F1 (fixed) | **0.753 ± 0.006** | Post data-leakage fix |
| RF macro F1 | **0.52** | Near chance (0.50) — honest imbalance-adjusted metric |
| Transition entropy | **Increasing** | 0.695 → 0.848 bits — more active beats, not more formulaic |

**Bootstrap validation** (1,000 iterations, n=215 equal samples): mean change 43.2%, direction positive in 94.6% of runs.

---

## Dataset

- **Total scripts:** 2,017 TXT files
- **Year range:** 1920–2026
- **Source:** Pre-cleaned scripts curated from ScriptHive (original raw dataset: 12,322 PDFs, 32GB)

| Era | Scripts | Share |
|-----|---------|-------|
| Pre-1980 | 130 | 6.4% |
| 1980–1995 (Baseline) | 216 | 10.7% |
| 1996–2009 | 635 | 31.5% |
| 2010–Present | 1,036 | 51.4% |

**Class imbalance:** 4.8:1 (Modern vs Baseline) — addressed with BorderlineSMOTE during model training.

**Data location (Google Drive):**
```
data sci capstone files/
├── processed_scripts/cleaned_scripts/sorted_by_duplicates_2/unique/  ← TXT input
└── pipeline_outputs/                                                   ← all outputs
```

---

## Repository Structure

```
cis-4496-project-team-3/
├── Code/
│   ├── Data_Acquisition_and_Understanding/
│   │   ├── scraper.py              # ScriptHive PDF scraper
│   │   ├── filter.py               # Script filtering and deduplication
│   │   ├── dataPrep.py             # Data preparation utilities
│   │   ├── cell3_txt_loader.py     # TXT file loader with year parsing
│   │   └── datapipeline.json       # Pipeline configuration
│   │
│   ├── Modeling/
│   │   ├── full_pipeline.py        # Main end-to-end pipeline (Steps 1–11)
│   │   ├── full_pipeline.ipynb     # Colab notebook version
│   │   ├── pipeline_fixes.py       # Expert-recommended fixes (JSD leakage,
│   │   │                           #   cross-era distances, entropy viz)
│   │   ├── advanced_clustering.py  # Rolling windows + pairwise distance analysis
│   │   ├── feature_engineering_upgrade.py  # Rare-word arc + JSD features
│   │   └── methodology_explainer.py        # Plain-language visualizations
│   │
│   └── Deployment/
│       └── operationalization.py   # Model operationalization
│
├── Docs/
│   ├── Project/
│   │   └── Charter.md
│   ├── Data_Report/
│   │   └── data_report.md
│   └── Model_Report/
│       ├── Final_Model_Performance_Report.docx   ← primary report
│       ├── results_documentation_v2.docx
│       └── baseline_model_report.md
│
└── README.md
```

---

## Pipeline

`full_pipeline.py` runs the entire analysis in 11 sequential steps:

```
Step 1  → Load TXT scripts (year parsing, era labeling)
Step 2  → Preprocessing (remove scene headings, character cues, formatting)
Step 3  → Feature engineering (standard arc, rare-word arc, JSD, entropy)
Step 4  → EDA visualizations
Step 5  → BERT embeddings + UMAP (all-MiniLM-L6-v2)
Step 6  → Baseline model: BERT + KMeans
Step 7  → Advanced clustering (rolling windows, cross-era distances)
Step 8  → Improved model: Random Forest + BorderlineSMOTE
Step 9  → Similarity metrics
Step 10 → Statistical tests (Mann-Whitney U)
Step 11 → Summary report
```

**Post-pipeline fixes** (`pipeline_fixes.py`):
- Fix 1: Transition entropy per-position visualization
- Fix 2: Cross-era distance histograms (Modern vs each era)
- Fix 3: JSD computed after train/test split (data leakage eliminated)

---

## Metrics & Methods

### Sentiment Arc Similarity
Each script is divided into 10 equal segments. VADER scores compound sentiment per segment, producing a 10-dimensional arc vector. Cosine similarity measures how similar two films' emotional trajectories are.

```
arc(sᵢ) = mean(VADER_compound(w) for w in sᵢ)
sim(A,B) = (A·B) / (||A|| × ||B||)  ∈ [0, 1]
```

### Rare-Word Sentiment Arc
Same arc computed only on words appearing < 0.01% in the Brown corpus (threshold: `f_Brown(w) < 0.0001`). Captures distinctive creative vocabulary rather than common filler words.

### Jensen-Shannon Divergence
Vocabulary distribution distance between a film and the 1980–1995 baseline era.

```
JSD(P,Q) = (KL(P||M) + KL(Q||M)) / 2  where M = (P+Q)/2
```
⚠️ **Important:** JSD is computed **after** train/test split. Vectorizer fit on training data only to prevent data leakage.

### Transition Entropy
Shannon entropy of the UP/FLAT/DOWN transition sequence between arc segments.

```
H = -Σ pₖ log₂(pₖ)  where k ∈ {UP, FLAT, DOWN}  ∈ [0, 1.585 bits]
```

### Pairwise Cosine Distance
Mean cosine distance between all film pairs within each era (within-era) and between eras (cross-era). Cross-era distances use `scipy.spatial.distance.cdist`.

---

## Models

### Baseline: BERT + KMeans (Unsupervised)
- **Model:** `all-MiniLM-L6-v2` sentence transformer → 384-dim embeddings
- **Input:** 8,000-char middle sample per script
- **Clustering:** KMeans k=4, n_init=10
- **Evaluation:** Silhouette score (0.0099), ARI (0.0073)

### Improved: Random Forest (Supervised)
- **Features:** 23 total — arc(10) + rare_arc(10) + entropy(2) + JSD(1)
- **Estimators:** 200 trees, `class_weight` = 1.5× imbalance ratio
- **Oversampling:** BorderlineSMOTE (targets boundary cases)
- **Split:** 80/20 stratified, 5-fold cross-validation
- **Results:** Weighted F1 = 0.753 ± 0.006 | Macro F1 = 0.52

---

## Results Summary

### Arc Similarity by Era
| Era | Mean similarity | Change vs baseline |
|-----|-----------------|--------------------|
| Pre-1980 | ~0.71 | Reference |
| 1980–1995 (Baseline) | 0.72 | — |
| 1996–2009 | ~0.89 | +23.6% |
| 2010–Present | ~1.03 | **+43.7%** |

### Transition Entropy by Era
| Era | FLAT % | UP % | DOWN % | Entropy |
|-----|--------|------|--------|---------|
| Pre-1980 | 79.83% | 9.40% | 10.77% | 0.695 bits |
| 1980–1995 | 79.42% | 9.41% | 11.16% | 0.714 bits |
| 1996–2009 | 77.06% | 10.81% | 12.13% | 0.804 bits |
| 2010–Present | 75.18% | 11.75% | 13.06% | 0.848 bits |

### BERT Cross-Era Distances
| Comparison | Mean distance |
|------------|---------------|
| Modern vs Modern (within) | 0.7148 |
| Modern vs Mid-era | 0.7163 |
| Modern vs Baseline | 0.7235 |
| Modern vs Pre-1980 | 0.7253 |

Total range across all comparisons: **0.011** — era boundaries are dissolving.

---

## How to Run

### 1. Mount Google Drive (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Set paths in `full_pipeline.py`
```python
TXT_DIR    = "/content/drive/MyDrive/data sci capstone files/processed_scripts/cleaned_scripts/sorted_by_duplicates_2/unique"
OUTPUT_DIR = "/content/drive/MyDrive/data sci capstone files/pipeline_outputs"
```

### 3. Run main pipeline
```python
# In Colab — paste full_pipeline.py into a single cell and run
# Runtime: ~45-60 min for BERT encoding on GPU
```

### 4. Run expert fixes (after main pipeline)
```python
# Paste pipeline_fixes.py into a new cell
# Requires df, arc_matrix, rare_arc_matrix, embeddings in memory
```

### 5. Run bootstrap validation (optional)
```python
# Requires arc_matrix and df in memory
# ~5 min for 1,000 iterations
```

---

## Dependencies

```
python >= 3.10
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn          # BorderlineSMOTE
sentence-transformers     # all-MiniLM-L6-v2
umap-learn
nltk                      # VADER, Brown corpus
scipy
tqdm
```

Install all:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn \
            sentence-transformers umap-learn nltk scipy tqdm
```

NLTK downloads (run once):
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')
```

---

## Team

| Name | Role |
|------|------|
| Micah | Data acquisition, scraping, filtering |
| Dimitri | Feature engineering, modeling |
| Ankur | Pipeline integration, statistical validation, documentation |
| Vishwa | EDA, clustering analysis |
| Vishrut | Advanced clustering, deployment |

**Course:** CIS-4496 Data Science Capstone  
**Institution:** Temple University  
**Year:** 2026  
**Instructor:** [Instructor name]

---

## Prior Research

| Study | Finding | Our extension |
|-------|---------|---------------|
| Reagan et al. (2016) | 6 dominant emotional arc shapes in literature | Distribution over arc shapes is narrowing in film |
| Follows et al. (2014) | Genre diversity declined 1980s→2010s | Computational confirmation at arc level (+43.7%) |
| Verhoeven et al. (2019) | Narrative standardization with franchise dominance | Arc similarity and distance patterns mirror this structurally |

---

## Limitations

- **4.8:1 class imbalance** — modern era dominates; addressed with BorderlineSMOTE and bootstrap validation
- **VADER on screenplays** — validated for social media; terse action lines may score neutral
- **Year parsing gaps** — 2,061 of 4,225 raw files excluded (no parseable year in filename)
- **English-language bias** — international cinema underrepresented; findings are Hollywood-specific
- **One zero-norm baseline arc** — one script removed from bootstrap (all segments scored 0.0 by VADER)

---

*Hollywood is telling more similar stories. Not identical — but measurably, significantly, and increasingly alike.*
