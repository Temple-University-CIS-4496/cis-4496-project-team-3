# Project Charter

## Business background

* Who is the client, what business domain the client is in.
	- Client would be Academic Research Team
	- Domain would be Cultural Analytics

## Problem Description

There is an ongoing debate among film critics, scholars, and audiences that modern films have become increasingly formulaic. Many films follow predictable narrative structures rather than expressing the diversity seen in earlier cinema. While concerns about narrative homogenization in film are frequently discussed in critical and popular discourse, they are typically framed in qualitative or anecdotal terms. Film scholars and critics often rely on selective case studies, personal interpretation, or genre-based intuition when arguing that contemporary cinema has become increasingly formulaic. This makes it difficult to assess whether such claims reflect a measurable trend or are influenced by cultural perception and recency bias.

From a data science perspective, this presents an opportunity to reframe an abstract cultural debate as a quantitative problem. By analyzing large collections of movie scripts across decades and genres, this project aims to determine whether narrative structures are converging over time in a statistically detectable way. Addressing this question requires computational methods that can operate at scale and capture patterns that would be infeasible to identify through manual analysis alone. Specifically, we focus on two key aspects:

Emotional progression (sentiment arcs) — how the tone of a story changes from beginning to end

Semantic content (BERT embeddings) — the underlying meaning and themes present in the script

By combining these approaches, we aim to detect whether modern films show stronger patterns of similarity compared to earlier eras. If films are becoming more homogeneous, we expect to observe:

Higher similarity in narrative arcs

Lower variability in narrative transitions

Stronger clustering of scripts in feature space

Ultimately, the goal is to provide quantitative evidence that either supports or challenges the idea of narrative homogenization in contemporary cinema.

## Project Scope

This project develops a complete end-to-end narrative analysis pipeline that transforms raw movie scripts into structured features, applies machine learning models, and produces interpretable insights.

The system includes the following components:

Data Ingestion

Collection of movie script PDFs from Script hive

Metadata integration (title, year, era classification)

Validation of files and filtering for usability

Preprocessing Pipeline

Extraction of text from PDFs using PyMuPDF

Removal of structural noise such as:

Page numbers

Formatting artifacts

Character name headers

Stage directions and boilerplate text

Normalization of text:

Lowercasing

Standardized spacing

Clean formatting across all scripts

The preprocessing stage begins with a large collection of approximately 12,000 movie scripts, of which around 7,000 contain usable, selectable text. Because the raw data varies significantly in formatting and quality, additional filtering is applied to keep only scripts with sufficient length and valid metadata. The scripts span from the 1940s to the present, enabling analysis across multiple cinematic eras. During preprocessing, text is extracted from PDF files and cleaned by removing noise such as page numbers, formatting artifacts, and non-narrative elements. The text is then normalized through consistent lowercasing and spacing to ensure that all scripts follow a standardized structure, making them suitable for feature extraction and modeling.

Feature Engineering

In this project the focus is two primary types of features:

Sentiment Arcs

Each script is divided into equal narrative segments

Sentiment scores are computed per segment

Result: a vector representing emotional progression across the story

BERT Embeddings

A pretrained transformer model converts text into dense numerical vectors

Captures deeper semantic meaning beyond surface-level text

Modeling

We apply both unsupervised and supervised approaches:

Clustering (KMeans / Spectral)

Groups scripts based on similarity

Tests whether natural clusters align with time periods

Classification (Random Forest, Logistic Regression)

Predicts era based on narrative features

Helps identify which features differentiate older vs modern films

Visualization

PCA / UMAP for dimensionality reduction

Cluster visualization and trend analysis

Sentiment arc comparisons across eras

Evaluation & Statistical Testing

Model performance metrics

Narrative similarity metrics

Statistical validation of results

## Metrics

We evaluate the project using a combination of machine learning metrics and domain-specific narrative metrics:

Model Performance Metrics

Silhouette Score → cluster separation quality

Adjusted Rand Index (ARI) → alignment with true labels

F1 Score (Random Forest / Logistic Regression) → classification performance

Narrative Metrics

Narrative Arc Similarity
Measures how similar emotional progressions are across films

Transition Entropy
Captures how predictable or varied narrative transitions are
(lower entropy = more formulaic storytelling)

Genre/Cluster Convergence Ratio (GCR)
Compares similarity within groups vs across groups

Statistical Validation

Mann-Whitney U Test

Tests whether modern films show significantly higher similarity than earlier films

Provides statistical grounding for conclusions

Success Criteria

The project is successful if:

We detect measurable and statistically significant differences in narrative similarity across eras

Patterns are interpretable and consistent across multiple results

Results provide meaningful insight into narrative trends

## Architecture

The system is designed as a modular pipeline, allowing each stage to be developed, tested, and improved independently.

Pipeline Flow

Raw Data

PDF scripts + metadata

Preprocessing

Text extraction

Cleaning and normalization

Processed Data

Structured and standardized scripts

Feature Extraction

Sentiment arc vectors

BERT embedding vectors

Modeling

Clustering and classification models

Evaluation

Metrics + statistical tests

Visualization & Outputs

Plots, embeddings, summaries

This modular design allows us to:

Debug individual components

Improve data quality without affecting models

Easily scale or expand the system

## Plan

The project has progressed significantly beyond the initial planning stage.

Completed

Data collection and validation

Preprocessing pipeline implementation

Script cleaning and normalization

Feature extraction (sentiment + embeddings)

Initial clustering and PCA visualization

Current Work

Identifying and removing outliers affecting clustering

Investigating cluster quality and interpretability

Comparing scripts within clusters to understand differences

Next Steps

Refine dataset by removing corrupted or low-quality scripts

Re-run clustering and evaluate improvements

Finalize models and evaluation metrics

Complete statistical validation

Prepare reports and visualizations

This reflects a transition from building the pipeline to refining results and improving insights.

## Key Findings

Preliminary analysis has revealed several important insights:

Clustering results show potential groupings, but outliers heavily impact visualization and model performance

Some clusters appear to represent corrupted or poorly processed scripts rather than meaningful narrative differences

PCA plots indicate that most scripts are tightly grouped, suggesting possible similarity, but results are not yet reliable due to noise

This has led to a shift in focus toward:

Improving data quality

Removing outliers

Re-evaluating clustering results

## Personnel

The project is a collaborative effort with shared responsibilities across all stages of the pipeline:

Data collection and preprocessing

Feature engineering and modeling

Evaluation and visualization

Documentation and reporting

Team Members:

Micah

Dimitri

Ankur

Vishwa

Vishrut

## Communication

To ensure efficient collaboration:

GitHub → version control and pipeline tracking

Discord → quick communication and updates

Zoom/In-person meetings → discussions and presentations

Regular communication ensures alignment across technical and analytical work.
