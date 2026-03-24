# DataReport Folder

This document summarizes the current state of our data collection, preprocessing, and exploratory analysis for the narrative similarity project.

## Project Data Status

Our project focuses on analyzing movie scripts to study narrative similarity and variation across films. At this stage, our work is centered on building a clean and scalable dataset and validating a modeling pipeline using text embeddings and clustering.

The dataset originally contained over 10,000 scripts collected from ScriptHive. After filtering for usability and removing low-quality files, approximately 7,000 scripts with selectable text remained. A deduplication process was then applied to remove exact and near-duplicate scripts, resulting in a final working dataset of approximately **4,226 unique scripts**.

This dataset is now the primary dataset used for all analysis and modeling.

## Current Data Quality Observations

The dataset has improved significantly after preprocessing and deduplication, but some challenges remain.

Main observations:

- text quality still varies across scripts  
- some scripts contain OCR noise or formatting inconsistencies  
- metadata such as genre and ratings is incomplete  
- file naming inconsistencies affect metadata extraction  

Key improvement:

- duplicate and near-duplicate scripts have been removed, reducing bias and improving overall dataset quality  

## Current Modeling Pipeline

Our team is currently building and testing a pipeline based on:

**BERT embeddings → UMAP → HDBSCAN / K-Means clustering**

### Purpose of the pipeline

- convert script text into semantic embeddings  
- reduce dimensionality for visualization and clustering  
- group scripts based on narrative similarity  
- analyze trends across time  

## Why this approach

This approach allows us to move beyond simple text features and capture deeper narrative meaning.

- BERT embeddings capture semantic content of scripts  
- UMAP reduces high-dimensional data into a structure that can be visualized  
- clustering identifies natural groupings of similar narratives  

## Data Processing Summary

The dataset has gone through the following pipeline:

1. **Text Extraction**
   - scripts extracted from PDFs using PyMuPDF  

2. **Filtering**
   - removed scripts with low word count  
   - removed scripts without valid year  
   - removed scripts outside target range  
   - removed scripts without screenplay formatting indicators  
   - filtered out poor OCR text  

3. **Cleaning and Normalization**
   - removed formatting artifacts  
   - standardized spacing  
   - converted all text to lowercase  

4. **Deduplication (Major Update)**
   - exact duplicates removed using hashing  
   - near duplicates identified using similarity measures  
   - scripts grouped into clusters  
   - one representative script retained per group  

## Current Dataset Statistics

- original dataset: 10,000+ scripts  
- usable scripts: ~7,000  
- final dataset after deduplication: ~4,226 scripts  

This reflects a major improvement in dataset quality and usability.

## Preliminary Conclusions

At this stage, the dataset is significantly more reliable after cleaning and deduplication. The pipeline is functioning end-to-end, allowing us to extract features, generate embeddings, and begin clustering analysis.

However, results are still sensitive to:
- remaining noise in the data  
- outliers affecting clustering  
- incomplete metadata  

## Next Steps

- further refine dataset by removing outliers  
- improve metadata (especially genre and year consistency)  
- evaluate clustering quality and interpretability  
- generate additional visualizations  
- finalize modeling and statistical analysis  

## Summary

The project has made strong progress in transforming a large, unstructured dataset into a cleaned and usable collection of scripts. Deduplication was a key milestone, reducing redundancy and improving the reliability of analysis. The current dataset provides a solid foundation for narrative similarity modeling and further exploration.
