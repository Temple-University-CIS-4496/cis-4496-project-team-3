# Baseline Model Report

---

## Analytic Approach

### Target Definition
The goal is to determine whether films have become narratively more similar over time. Rather than predicting a label (which assumes the answer), the baseline model uses **unsupervised clustering** to ask: *do film scripts naturally group by era with no prior knowledge of release year?*

Strong cluster-to-era alignment → films are era-distinct → no convergence  
Weak alignment → films are mixed across eras → convergence detected

### Inputs (Features)
Each script is encoded into a **384-dimensional BERT embedding** using the `all-MiniLM-L6-v2` sentence transformer. The embedding is generated from an 8,000-character sample of the script's cleaned middle section (avoiding credits and title pages).

**Preprocessing applied before embedding:**
- Removed scene headings (INT./EXT.)
- Removed character cue lines (ALL CAPS, <40 chars)
- Removed parentheticals, page numbers, transition directives (FADE IN/OUT, CUT TO)
- Removed copyright/watermark text
- Normalized whitespace

### Model Type
**BERT + KMeans Clustering (Unsupervised)**

BERT encodes each script's semantic meaning into a dense vector. KMeans then groups similar vectors into 4 clusters (matching the 4 eras). No era labels are used during training.

---

## Model Description

### Model and Parameters

**Sentence Transformer (BERT Encoder)**
```
Model     : all-MiniLM-L6-v2
Output dim: 384
Input     : 8,000-char sample per script (middle section)
Batch size: 16
```

**KMeans Clustering**
```
k (clusters) : 4  (matching number of eras)
n_init       : 10 (multiple restarts for stability)
random_state : 42
```

### Data Flow
```
430 film scripts (usable_scripts.csv)
        ↓
Text preprocessing (remove noise)
        ↓
BERT encoding → 384-dim vector per script
        ↓
KMeans clustering (k=4, no era labels)
        ↓
Cluster assignment per script
        ↓
Evaluation: Silhouette Score + Adjusted Rand Index
Compare clusters to true era labels
```

### Why This Is the Right Baseline
The research question is inherently unsupervised — we are not trying to predict a pre-defined category, we are discovering whether natural groupings exist. KMeans is the appropriate baseline because:
- It makes no assumptions about era structure
- The Adjusted Rand Index provides a principled measure of how well discovered clusters align with true eras
- Low ARI is itself evidence of convergence — the algorithm cannot find era-separated clusters

---

## Results (Model Performance)

### Evaluation Metrics

| Metric | Description | Interpretation |
|---|---|---|
| Silhouette Score | How tight and separated clusters are (-1 to 1) | Higher = more distinct clusters |
| Adjusted Rand Index (ARI) | How well clusters align with true eras (0 to 1) | Higher = clusters match eras |

*(Actual values will be populated after pipeline run)*

### Cluster × Era Distribution
See `baseline_cluster_crosstab.csv` in `pipeline_outputs/` for full breakdown of which films fall in which clusters by era.

### UMAP Visualization
See `02_baseline_kmeans.png` — side-by-side comparison of discovered clusters vs true era labels in 2D BERT embedding space. Visual overlap between eras in the right panel indicates semantic convergence.

---

## Model Understanding

### What the ARI Tells Us
- **ARI near 1.0** → KMeans independently rediscovered the era boundaries → films are strongly era-distinct
- **ARI near 0.0** → Clusters are unrelated to eras → films have converged semantically across time
- **ARI between 0.2-0.5** → Partial alignment → some era-specific patterns remain but convergence is occurring

### What the Silhouette Score Tells Us
A declining silhouette score in more recent eras (compared to earlier ones) would indicate that modern films cluster less tightly — consistent with within-era homogenization making it harder to form distinct groups.

### Insight
The BERT embedding captures **semantic meaning and topic content** rather than just surface-level sentiment. If clustering fails to separate eras, it suggests films have converged not just in emotional arc but in the deeper thematic vocabulary of their narratives.

---

## Conclusion and Discussions for Next Steps

### Feasibility Assessment
BERT + KMeans is a strong and honest baseline for this research question. It requires no labeled training data, directly tests the unsupervised structure of the data, and produces interpretable metrics (ARI, Silhouette) that translate directly into evidence for or against homogenization.

### Discussion on Overfitting
KMeans does not overfit in the traditional sense — it is a distance-based algorithm with no learned parameters that generalize to new data. The main risk is sensitivity to initialization, mitigated here by `n_init=10`.

### Improved Model
The improved model (Random Forest Era Classifier) adds supervised learning on top of the sentiment arc features. It measures how well a trained model can distinguish eras — complementing the unsupervised baseline by asking "even with labeled supervision, can we tell these eras apart?"

### Additional Features for Next Iteration
- Segment-level BERT embeddings (encode each narrative act separately)
- LDA topic distributions per script
- Structural features (scene count, dialogue ratio, vocabulary richness)
- TF-IDF vectors for thematic vocabulary comparison

### Other Relevant Data Sources
- IMDb genre/budget metadata for genre-stratified analysis
- Additional pre-1995 scripts to balance era representation
- Screenwriting award data (WGA, Oscars) for quality-stratified analysis
