"""
main.py - Digital Lifestyle Pattern Analysis Using Clustering
=============================================================
This script demonstrates an end-to-end unsupervised learning workflow using Scikit-learn:
1. Data Loading & Inspection
2. Missing Value Imputation & Feature Scaling via Scikit-learn Pipeline
3. Clustering Algorithms:
   - K-Means Clustering (with Elbow Method & Silhouette Score)
   - Hierarchical (Agglomerative) Clustering (with Dendrogram)
   - DBSCAN Clustering
4. Model Evaluation & Silhouette Score Comparison
5. 2D Visualization using PCA (PCA used strictly for 2D visual projection)
6. Cluster Interpretation & Profiling
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to save PNG figures cleanly
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def main():
    print("=" * 70)
    print("   DIGITAL LIFESTYLE PATTERN ANALYSIS USING CLUSTERING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Setup Paths
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, "data", "digital_lifestyle.csv")
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Data Loading & Inspection
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Dataset...")
    df = pd.read_csv(data_path)

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- First 5 Rows ---")
    print(df.head())

    feature_cols = [
        "Screen_Time_Hours",
        "Social_Media_Hours",
        "Gaming_Hours",
        "Work_Study_Hours",
        "Sleep_Hours",
        "Daily_App_Opens",
    ]

    print("\n--- Data Summary Statistics ---")
    print(df[feature_cols].describe().round(2))

    print("\n--- Missing Value Count ---")
    missing = df[feature_cols].isnull().sum()
    print(missing)

    X_raw = df[feature_cols]

    # -------------------------------------------------------------------------
    # 2. Scikit-learn Pipeline (Imputation + Scaling)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Constructing Preprocessing Pipeline...")
    print("-> Step 1: SimpleImputer (strategy='mean')")
    print("-> Step 2: StandardScaler (mean=0, std=1)")

    preprocessing_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # Fit and transform dataset
    X_scaled = preprocessing_pipeline.fit_transform(X_raw)
    print(f"Scaled Feature Matrix Shape: {X_scaled.shape}")

    # -------------------------------------------------------------------------
    # 3. K-Means Clustering & Optimal K Selection
    # -------------------------------------------------------------------------
    print("\n[Step 3] K-Means Clustering & Optimal K Search...")
    k_range = range(2, 9)
    inertias = []
    kmeans_silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(X_scaled, labels)
        kmeans_silhouette_scores.append(score)
        print(f"  K = {k}: Inertia = {km.inertia_:.2f}, Silhouette Score = {score:.4f}")

    # Plot Elbow Curve & Silhouette Scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_title("Elbow Method for Optimal K", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(list(k_range), kmeans_silhouette_scores, "ro-", linewidth=2, markersize=8)
    ax2.set_title("Silhouette Score for Different K", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    elbow_plot_path = os.path.join(output_dir, "elbow_curve.png")
    fig.savefig(elbow_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved Elbow & Silhouette curves to: {elbow_plot_path}")

    # Select optimal K = 3
    optimal_k = 3
    print(f"\nFitting Final K-Means with Optimal K = {optimal_k}...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(X_scaled)
    score_kmeans = silhouette_score(X_scaled, kmeans_labels)
    print(f"K-Means (K={optimal_k}) Silhouette Score: {score_kmeans:.4f}")

    # -------------------------------------------------------------------------
    # 4. Hierarchical (Agglomerative) Clustering
    # -------------------------------------------------------------------------
    print("\n[Step 4] Hierarchical (Agglomerative) Clustering...")

    # Plot Dendrogram using SciPy
    fig, ax = plt.subplots(figsize=(10, 5))
    linked = linkage(X_scaled, method="ward")
    dendrogram(linked, orientation="top", distance_sort="descending", show_leaf_counts=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Euclidean Distance")

    plt.tight_layout()
    dendrogram_path = os.path.join(output_dir, "dendrogram.png")
    fig.savefig(dendrogram_path, dpi=150)
    plt.close(fig)
    print(f"Saved Dendrogram plot to: {dendrogram_path}")

    # Fit Agglomerative Clustering
    agg_model = AgglomerativeClustering(n_clusters=optimal_k, metric="euclidean", linkage="ward")
    agg_labels = agg_model.fit_predict(X_scaled)
    score_agg = silhouette_score(X_scaled, agg_labels)
    print(f"Hierarchical Clustering (n_clusters={optimal_k}) Silhouette Score: {score_agg:.4f}")

    # -------------------------------------------------------------------------
    # 5. DBSCAN Clustering
    # -------------------------------------------------------------------------
    print("\n[Step 5] DBSCAN Clustering...")
    # Parameters eps=0.8, min_samples=4 suit scaled data well
    dbscan_model = DBSCAN(eps=0.8, min_samples=4)
    dbscan_labels = dbscan_model.fit_predict(X_scaled)

    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"DBSCAN detected {n_dbscan_clusters} clusters and {n_noise} noise points (-1).")

    # Evaluate DBSCAN (only if more than 1 cluster is found and non-noise points exist)
    if n_dbscan_clusters > 1:
        # Calculate silhouette score considering non-noise points or full dataset
        valid_mask = dbscan_labels != -1
        if len(set(dbscan_labels[valid_mask])) > 1:
            score_dbscan = silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask])
        else:
            score_dbscan = silhouette_score(X_scaled, dbscan_labels)
    else:
        score_dbscan = 0.0
    print(f"DBSCAN Silhouette Score (excluding noise): {score_dbscan:.4f}")

    # -------------------------------------------------------------------------
    # 6. Evaluation & Comparison of Silhouette Scores
    # -------------------------------------------------------------------------
    print("\n[Step 6] Silhouette Score Evaluation & Model Comparison")
    print("-" * 55)
    print(f" {'Algorithm':<30} | {'Silhouette Score':<15}")
    print("-" * 55)
    print(f" {'K-Means (K=3)':<30} | {score_kmeans:.4f}")
    print(f" {'Hierarchical (Ward, K=3)':<30} | {score_agg:.4f}")
    print(f" {'DBSCAN (eps=0.8, min_samples=4)':<30} | {score_dbscan:.4f}")
    print("-" * 55)

    # Plot Silhouette Score Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    algo_names = ["K-Means", "Hierarchical", "DBSCAN"]
    scores = [score_kmeans, score_agg, score_dbscan]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bars = ax.bar(algo_names, scores, color=colors, edgecolor="black", width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Clustering Algorithm Comparison (Silhouette Score)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{s:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    score_plot_path = os.path.join(output_dir, "silhouette_comparison.png")
    fig.savefig(score_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved Silhouette Score comparison chart to: {score_plot_path}")

    # -------------------------------------------------------------------------
    # 7. 2D Visualization using PCA
    # Note: PCA is used STRICTLY to reduce 6 features down to 2 dimensions
    #       for visualization purposes, NOT as the primary clustering method.
    # -------------------------------------------------------------------------
    print("\n[Step 7] 2D Projection using PCA for Cluster Visualizations...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    print(f"PCA Variance Explained: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # K-Means Plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="tab10", ax=axes[0], s=60, edgecolor="k")
    axes[0].set_title(f"K-Means Clustering (K={optimal_k})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")

    # Hierarchical Plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg_labels, palette="Set2", ax=axes[1], s=60, edgecolor="k")
    axes[1].set_title(f"Hierarchical Clustering (K={optimal_k})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")

    # DBSCAN Plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette="Set1", ax=axes[2], s=60, edgecolor="k")
    axes[2].set_title("DBSCAN Clustering", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")

    plt.suptitle("2D PCA Visualization of Clusters (PCA used for visualization only)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, "pca_clusters_comparison.png")
    fig.savefig(pca_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved 2D PCA cluster comparison plot to: {pca_plot_path}")

    # -------------------------------------------------------------------------
    # 8. Cluster Interpretation & Behavioral Profiling (K-Means)
    # -------------------------------------------------------------------------
    print("\n[Step 8] Cluster Interpretation & Behavioral Profiling")
    df_imputed = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(df[feature_cols]), columns=feature_cols)
    df_imputed["Cluster"] = kmeans_labels

    cluster_profiles = df_imputed.groupby("Cluster").mean().round(2)
    cluster_profiles["User_Count"] = df_imputed.groupby("Cluster").size()

    # Map human-readable names based on mean features
    archetype_names = {
        0: "Cluster 0: Studious / Academic Professionals",
        1: "Cluster 1: Gaming & Social Media Enthusiasts",
        2: "Cluster 2: Balanced Digital Users",
    }
    # Dynamically sort by Work_Study_Hours to assign clear names
    work_sorted = cluster_profiles["Work_Study_Hours"].sort_values(ascending=False).index.tolist()
    screen_sorted = cluster_profiles["Screen_Time_Hours"].sort_values(ascending=False).index.tolist()

    naming_map = {}
    naming_map[work_sorted[0]] = "Studious / Academic Professionals"
    naming_map[screen_sorted[0]] = "Gaming & Social Media Enthusiasts"
    for c in range(optimal_k):
        if c not in naming_map:
            naming_map[c] = "Balanced Digital Users"

    cluster_profiles["Archetype"] = [naming_map[c] for c in cluster_profiles.index]

    print("\n--- Mean Feature Values per Cluster ---")
    print(cluster_profiles[["Archetype", "User_Count"] + feature_cols])

    print("\n--- Key Behavioral Insights ---")
    for c, row in cluster_profiles.iterrows():
        print(f"\n* {row['Archetype']} (n={int(row['User_Count'])} users):")
        print(f"   - Screen Time   : {row['Screen_Time_Hours']} hrs/day")
        print(f"   - Social Media  : {row['Social_Media_Hours']} hrs/day")
        print(f"   - Gaming Time   : {row['Gaming_Hours']} hrs/day")
        print(f"   - Work/Study    : {row['Work_Study_Hours']} hrs/day")
        print(f"   - Sleep Hours   : {row['Sleep_Hours']} hrs/day")
        print(f"   - Daily App Opens: {int(row['Daily_App_Opens'])} opens/day")

    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE! All outputs saved in 'outputs/' directory.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
