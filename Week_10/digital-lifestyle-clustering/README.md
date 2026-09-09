# Digital Lifestyle Pattern Analysis Using Clustering

A complete, beginner-friendly Machine Learning project demonstrating **unsupervised clustering algorithms** and **Scikit-learn Pipelines** to analyze daily habits and categorize users into behavioral clusters.

---

## 📌 Project Objective

The goal of this project is to analyze digital lifestyle patterns and automatically group users into meaningful behavioral clusters based on daily device usage, productivity, study/work hours, app interaction frequency, and sleep habits.

---

## 📊 Dataset & Features

The dataset (`data/digital_lifestyle.csv`) contains synthetic digital usage records collected across **150 users** with 6 numerical features:

| Feature Name | Unit | Description |
| :--- | :--- | :--- |
| `Screen_Time_Hours` | Hours/day | Total active device screen time |
| `Social_Media_Hours` | Hours/day | Hours spent on social media applications |
| `Gaming_Hours` | Hours/day | Hours spent playing mobile/PC games |
| `Work_Study_Hours` | Hours/day | Productive study or professional work time |
| `Sleep_Hours` | Hours/day | Nightly sleep duration |
| `Daily_App_Opens` | Count/day | Total number of application unlocks/opens per day |

---

## 🏗️ Project Architecture & File Structure

```text
digital-lifestyle-clustering/
│
├── data/
│   └── digital_lifestyle.csv           # Runnable local CSV dataset (~150 records)
│
├── outputs/                             # Visualizations & saved charts
│   ├── elbow_curve.png                 # K-Means Elbow & Silhouette curve
│   ├── dendrogram.png                  # Hierarchical Clustering Dendrogram
│   ├── pca_clusters_comparison.png     # 2D PCA cluster overlays
│   └── silhouette_comparison.png       # Silhouette score bar chart
│
├── src/
│   └── main.py                         # Complete end-to-end Python pipeline script
│
├── requirements.txt                    # Project dependencies
└── README.md                           # Comprehensive documentation & Viva prep
```

---

## ⚙️ Workflow & Core ML Concepts

### 1. Scikit-learn Pipeline
A **Pipeline** chains preprocessing steps sequentially to prevent data leakage and keep code modular:
```python
preprocessing_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])
```

### 2. Feature Scaling (`StandardScaler`)
Distance-based algorithms (K-Means, Hierarchical, DBSCAN) rely on Euclidean distance:
$$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$
Without scaling, features with large numeric scales (e.g., `Daily_App_Opens` ranging 30–250) completely dominate features with small numeric ranges (e.g., `Gaming_Hours` ranging 0–6). `StandardScaler` standardizes features to **mean = 0** and **variance = 1**.

---

## 🤖 Clustering Algorithms Implemented

### 1. K-Means Clustering
* **How it works**: Centroid-based partitioning algorithm. Minimizes the Within-Cluster Sum of Squares (WCSS / Inertia) by iteratively assigning data points to the nearest cluster mean and recomputing cluster centroids.
* **Optimal K Selection**: Evaluated using the **Elbow Method** (inertia inflection point) and **Silhouette Score** across $K \in [2, 8]$.

### 2. Agglomerative (Hierarchical) Clustering
* **How it works**: Bottom-up hierarchical approach. Starts with each point as its own cluster and iteratively merges the closest pair of clusters using **Ward's minimum variance linkage**.
* **Visualization**: Rendered via a **Dendrogram** to inspect distance thresholds.

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* **How it works**: Groups dense regions of data points separated by sparse areas using two parameters:
  * `eps`: Maximum distance between two samples to be considered neighbors.
  * `min_samples`: Minimum number of points required to form a dense core region.
* **Advantage**: Automatically detects arbitrary cluster shapes and identifies noise points (labeled as `-1`).

---

## 📈 Evaluation & 2D PCA Projections

### Evaluation Metric: Silhouette Score
The **Silhouette Score** measures how well-separated and cohesive clusters are:
$$s = \frac{b - a}{\max(a, b)}$$
* $a$: Mean intra-cluster distance (cohesion).
* $b$: Mean nearest-cluster distance (separation).
* Score ranges from **-1 to +1**, where values closer to **+1** indicate distinct, well-separated clusters.

### PCA Disclaimer
> ⚠️ **Note**: **Principal Component Analysis (PCA)** is used **STRICTLY for 2D visualization** to project 6-dimensional features onto two principal axes (PC1 and PC2) for scatter plotting. PCA is **NOT** used as a dimensionality reduction preprocessing step before clustering.

---

## 💡 Cluster Interpretation & Behavioral Profiles

The optimal K-Means model partitions users into **3 distinct digital lifestyle archetypes**:

1. **Cluster 0: Studious / Academic Professionals**
   * **Characteristics**: High study/work time (~7.5 hrs/day), low gaming (< 0.5 hrs), low social media (~1.5 hrs), healthy sleep (~7.5 hrs), low app opens (~60/day).
   * **Behavior**: Highly focused, productive device users.

2. **Cluster 1: Gaming & Social Media Enthusiasts**
   * **Characteristics**: Extremely high total screen time (~11 hrs/day), high gaming (~4.0 hrs), high social media (~4.5 hrs), low work/study (~2.0 hrs), reduced sleep (~5.5 hrs), high app opens (~160/day).
   * **Behavior**: Heavy digital media consumers experiencing sleep deficit.

3. **Cluster 2: Balanced Digital Users**
   * **Characteristics**: Moderate screen time (~6.5 hrs/day), balanced work/study (~4.5 hrs), moderate social media (~2.5 hrs), healthy sleep (~7.0 hrs), moderate app opens (~95/day).
   * **Behavior**: Moderately active users with healthy digital discipline.

---

## 🎓 College Viva Preparation Guide

Here are concise answers to common viva questions regarding this project:

### Q1: What is clustering?
**Answer**: Clustering is an unsupervised machine learning task of partitioning an unlabeled dataset into groups (clusters) such that data points within the same group are more similar to each other than to those in other groups.

### Q2: What is unsupervised learning?
**Answer**: Unsupervised learning is a paradigm of machine learning where models find hidden patterns, structures, or groupings in input data without predefined target labels or ground-truth supervision.

### Q3: What is a Pipeline in Scikit-learn?
**Answer**: A Pipeline is a wrapper object in Scikit-learn that encapsulates sequential data preprocessing transformers (e.g., imputer, scaler) and an estimator into a single unified object. It simplifies workflows and prevents data leakage.

### Q4: Why do we scale the data?
**Answer**: Distance-based clustering algorithms calculate Euclidean distances between points. Scaling prevents features with larger numerical ranges (like `Daily_App_Opens` ~200) from dominating features with smaller scales (like `Gaming_Hours` ~2).

### Q5: How does K-Means work?
**Answer**: K-Means randomly initializes $K$ centroids, assigns each point to its nearest centroid using Euclidean distance, updates the centroids to the mean of assigned points, and repeats until convergence.

### Q6: How does Hierarchical Clustering work?
**Answer**: Agglomerative Hierarchical Clustering starts with every point in its own cluster and repeatedly merges the pair of clusters with the smallest inter-cluster distance (e.g., using Ward's linkage) until a single tree structure (dendrogram) is formed.

### Q7: How does DBSCAN work?
**Answer**: DBSCAN groups points based on density. It identifies core points that have at least `min_samples` within distance `eps`, expands clusters by adding reachable points, and marks sparse isolated points as noise (`-1`).

### Q8: What is the main difference between K-Means and DBSCAN?
**Answer**: 
* K-Means requires specifying $K$ in advance, assumes spherical clusters, and assigns every point to a cluster.
* DBSCAN automatically determines the number of clusters, finds arbitrary non-spherical shapes, and detects noise/outliers.

### Q9: What is Silhouette Score?
**Answer**: Silhouette Score is a validation metric measuring cluster cohesion and separation. It ranges from -1 to +1, where values close to +1 indicate dense, well-separated clusters.

### Q10: Why might the three algorithms produce different clusters?
**Answer**: Because each algorithm operates under a different mathematical assumption:
* K-Means optimizes global variance around centroids (spherical assumption).
* Hierarchical clustering builds a hierarchical tree based on linkage distance.
* DBSCAN groups by local density threshold (`eps`, `min_samples`), ignoring distance to centroids.

---

## 🚀 How to Run

1. Clone or navigate to the repository folder:
   ```bash
   cd digital-lifestyle-clustering
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```
4. View generated charts in `outputs/`.
