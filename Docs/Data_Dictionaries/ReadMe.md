# Data Dictionaries

This document describes the datasets being used in our project on narrative similarity analysis in film scripts.

# Overview

Our project uses a large-scale movie script dataset collected from ScriptHive. The dataset originally contained over 10,000 scripts, but after preprocessing, filtering, and deduplication, it has been refined into a cleaner and more usable dataset.

Through this process:
- Approximately 7,000 scripts were identified as having usable, selectable text  
- Duplicate and near-duplicate scripts were removed  
- The dataset was reduced to approximately 4,226 cleaned and unique scripts

This refined dataset is now the primary dataset used for all modeling and analysis.

---

# Dataset: Cleaned Script Collection

## Description

This dataset consists of movie scripts collected from ScriptHive and processed through a preprocessing pipeline. The scripts span multiple decades, enabling analysis of narrative trends over time.

Each script has undergone:
- text extraction from PDF files  
- cleaning and normalization  
- filtering based on quality and formatting  
- deduplication to remove exact and near-duplicate files  

## Purpose

This dataset is used for:
- generating sentiment-based narrative arcs  
- creating BERT embeddings for semantic analysis  
- performing dimensionality reduction (UMAP)  
- clustering scripts based on narrative similarity  
- analyzing trends in storytelling across time

## Current Characteristics

- Total scripts (after cleaning and deduplication): ~4,226  
- Original dataset size: 10,000+ scripts  
- Usable scripts before deduplication: ~7,000  
- File type: PDF  
- Text quality: mixed but filtered for usability  
- Time range: approximately 1980–2024  
- Metadata availability: partial  

## Data Processing Pipeline

The dataset was processed through several stages:

### 1. Text Extraction
- Scripts are read from PDF files using PyMuPDF  
- Raw text is extracted page by page  

### 2. Filtering
Scripts are removed if they:
- contain fewer than a minimum number of words  
- do not include a valid year in the filename  
- fall outside the target year range  
- lack script formatting indicators (e.g., INT., EXT., CUT TO)  
- have poor OCR quality (low alphabetic character ratio)  

### 3. Cleaning and Normalization
- removal of formatting artifacts (page numbers, spacing issues)  
- lowercasing all text  
- standardizing whitespace and structure  

### 4. Deduplication (Key Update)

A deduplication process was applied to improve dataset quality:

- exact duplicates identified using hashing (SHA256)  
- near duplicates identified using similarity measures (token overlap and simhash)  
- scripts grouped into duplicate clusters  
- one representative script retained per group  

This step reduced redundancy and resulted in a final dataset of approximately **4,226 unique scripts**.

## Benefits of Deduplication

- prevents duplicate scripts from biasing results  
- improves clustering accuracy  
- reduces computational cost  
- ensures more reliable statistical analysis


## Information Available

For each script, the dataset may include:
- title (parsed from filename)  
- writer (when available)  
- year  
- full extracted text  
- word count  
- era classification (Pre-1980, 1980–1995, etc.)  
- source folder  


## Limitations

- metadata is incomplete (e.g., missing genres, ratings)  
- some scripts still contain OCR noise  
- file naming inconsistencies affect metadata extraction  
- not all scripts are perfectly cleaned  
- genre information is not yet fully integrated  


## Supporting Metadata

Future improvements include linking scripts to:
- genre  
- ratings  
- box office performance  
- additional external metadata sources  

This will enable deeper analysis beyond structural similarity.

## Notes

The dataset is still evolving as preprocessing improves. Additional cleaning, metadata enrichment, and validation steps will continue to refine the dataset and improve the reliability of downstream modeling.
