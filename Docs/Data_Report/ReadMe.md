# DataReport Folder

This document summarizes the current state of our data collection, preprocessing, and exploratory analysis for the narrative similarity project.

## Project Data Status

Our project is focused on analyzing movie scripts to study narrative similarity and variation across films. At this stage, our work is centered on building a usable script corpus and validating a modeling pipeline using text embeddings and clustering.

We currently have two script collections:

- a small working dataset of about 500 scripts
- a large script collection of over 10,000 scripts

The smaller dataset is currently more usable and is serving as our main experimental dataset. The larger dataset offers more scale, but it is more difficult to clean and organize and it is not ready for full modeling.

---

## Current Data Quality Observations

The script data varies significantly in quality. Some files contain selectable text and are somewhat clean, while others are image-based, require OCR, or contain formatting noise. In addition, not all scripts can currently be matched to metadata or popularity weights.

Main issues observed so far include:
- inconsistent PDF formatting
- OCR-related text errors
- incomplete or non-standard file naming
- missing release year or metadata
- scripts that cannot yet be joined to external weights

Based on this, the 500-script dataset is currently the best dataset for testing and validating our pipeline.

---

# Current Modeling Pipeline

Our team is currently building and testing a pipeline based on:

**BERT embeddings → UMAP → HDBSCAN or K-Means clustering**

# Purpose of the pipeline
- convert script text into semantic embeddings
- reduce dimensionality for visualization and clustering
- group scripts into clusters based on narrative similarity
- explore whether scripts form meaningful thematic or structural groupings

# Why this approach
This pipeline gives us a more flexible and semantically useful approach than simpler word-count methods. BERT embeddings capture richer contextual information, while UMAP helps reduce the embedding space into a lower-dimensional structure that can be clustered more effectively.

---

# Current Use of the Two Datasets

# Small dataset (~500 scripts)
This dataset is currently being used for:
- testing text extraction and cleaning
- validating the embedding pipeline
- running early clustering experiments
- checking whether cluster outputs appear interpretable

### Large dataset (10,000+ scripts)
This dataset has strong long-term value but is currently more difficult to use because:
- cleaning requirements are much larger
- file quality is less consistent
- compute and storage demands are higher

For now, this larger set is better viewed as a scaling target rather than the main experimental dataset.

---

## Preliminary Conclusions

At this stage, the main conclusion is that data readiness is one of the biggest challenges in the project. The modeling pipeline is promising, but meaningful results depend heavily on script quality, text extraction consistency, and metadata linkage.

The smaller dataset gives us a realistic path for early testing and proof of concept. Once the pipeline is validated, we can work towards expanding to the larger dataset.

---

## Next Steps

Our next steps are to:

1. improve cleaning and standardization of the 500-script dataset  
2. finalize embedding and clustering experiments on the smaller set  
3. evaluate cluster quality and interpretability  
4. improve metadata matching and joinability
5. work on the baseline model
6. gradually scale the pipeline to the larger 10,000+ script collection

---

## Summary

The project has made strong progress in data collection, but the data is still in progress in terms of quality and completeness. The 500-script dataset is currently the most practical dataset for modeling, while the 10,000+ script dataset represents a future expansion opportunity once preprocessing and linkage issues are better resolved.
