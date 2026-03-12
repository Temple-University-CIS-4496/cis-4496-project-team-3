# Data Dictionaries

This document describes the datasets being used in our project on narrative similarity analysis in film scripts.

# Overview

Our project currently uses two main script datasets: 
*Small working dataset
*Large script collection

The small dataset is our main working dataset for early experimentation, testing, and pipeline validation. The large dataset is a broader collection intended for future use once cleaning, matching, and preprocessing issues are resolved.

---

# Dataset 1: Small Working Script Dataset

# Description
This dataset contains approximately 500 movie scripts  collected from scripthive. These scripts vary in quality and format. Some contain selectable text, while others require OCR or additional cleaning. Not all scripts are currently joinable to external metadata or popularity weights.

# Purpose
This is the primary dataset currently being used for:
- testing preprocessing pipelines
- generating embeddings
- dimensionality reduction with UMAP
- clustering with HDBSCAN or K-Means
- validating whether our overall pipeline works on a manageable sample

# Current Characteristics
- Approximate size: 500 scripts
- File types: primarily PDF scripts
- Text quality: mixed
- Joinability: partial
- Source type: publicly collected scripts

# Current Organization
Scripts in this dataset may be grouped into categories such as:
- selectable
- OCR
- joinable
- non-joinable
- clear OCR
- obscured OCR

These categories help us track script quality and determine which files are suitable for analysis.

# Information Available
Depending on the script, we may currently have:
- script title
- file name
- extracted text
- script quality category
- joinability status
- OCR quality label
- year (for some scripts)
- possible metadata match status

# Limitations
- Not all scripts have clean text
- Not all scripts can be matched to metadata or weights
- OCR-based scripts may contain extraction errors
- File naming may be inconsistent across sources


## Dataset 2: Large Script Collection

# Description
This dataset contains over 10,000 scripts collected from scripthive. This is our larger long-term dataset, but it is currently harder to use because of formatting inconsistencies and preprocessing complexity.

# Purpose
This dataset is intended for:
- scaling our narrative similarity analysis
- improving coverage across genres and time periods
- supporting future large-scale clustering and trend analysis

# Current Characteristics
- Approximate size: 10,000+ scripts
- Text quality: highly variable
- Joinability: currently limited / incomplete
  
### Current Challenges
- inconsistent formatting
- possibly duplicate pr partial scripts
- more cleaning and computing requirements
- higher risk of OCR and text extraction noise

### Planned Use
We plan to use this dataset after:
- improving script cleaning
- improving title/year matching
- standardizing file structure
- confirming that our modeling pipeline performs well on the smaller dataset first

---

## Supporting Metadata / Weights

Some scripts may later be linked to external metadata, variables such as:
- release year
- genre
- ratings
- box office or other popularity measures

At this stage, metadata linkage is still incomplete and is stronger for the smaller working dataset than for the large script collection.

---

## Notes

Because this project is still in active development, the data dictionary will continue to be updated as datasets are cleaned, standardized, and linked to additional metadata.
