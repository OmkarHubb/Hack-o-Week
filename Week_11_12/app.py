import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="PCA and t-SNE Case Study", layout="wide")

# Title
st.title("Uncovering Hidden Student Learning Patterns Using PCA and t-SNE")

# Load Dataset
@st.cache_data
def load_data():
    data_path = os.path.join("data", "student_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error(f"Dataset not found at {data_path}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Settings")

# 1. Dataset Info
st.sidebar.subheader("Dataset")
st.sidebar.text(f"Loaded: student_data.csv\nRows: {df.shape[0]}, Cols: {df.shape[1]}")

# Define numerical features
features = [col for col in df.columns if col != 'performance_category']

# 2. PCA Settings
st.sidebar.subheader("PCA Settings")
n_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=len(features), value=3)

# 3. t-SNE Settings
st.sidebar.subheader("t-SNE Settings")
perplexity = st.sidebar.slider("t-SNE Perplexity", min_value=5, max_value=50, value=30)
st.sidebar.text("t-SNE random_state = 42 (Fixed)")

# 4. Coloring Toggle
color_by_category = st.sidebar.checkbox("Color points by Performance Category", value=True)

# --- STANDARDIZATION ---
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- COMPUTE PCA ---
pca = PCA(n_components=n_components, random_state=42)
pca_result = pca.fit_transform(X_scaled)

# --- COMPUTE t-SNE ---
# Use the standardized feature matrix
@st.cache_data
def compute_tsne(X_scaled_data, perp):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    return tsne.fit_transform(X_scaled_data)

tsne_result = compute_tsne(X_scaled, perplexity)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "PCA", "t-SNE", "PCA vs t-SNE", "Findings"])

with tab1:
    st.header("Dataset Overview")
    st.write("A synthetic dataset of student academic performance and behavior.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Students", df.shape[0])
    col2.metric("Number of Features", len(features))
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Features")
    st.write(", ".join(features))
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

with tab2:
    st.header("Principal Component Analysis (PCA)")
    st.info("PCA finds new directions that capture as much variation in the data as possible.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scree Plot
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance,
            name='Individual Explained Variance'
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative Explained Variance'
        ))
        fig_scree.update_layout(title="Explained Variance by Principal Components",
                                xaxis_title="Principal Components",
                                yaxis_title="Variance Ratio")
        st.plotly_chart(fig_scree, use_container_width=True)

    with col2:
        # 2D PCA Scatter
        pca_df = pd.DataFrame(data=pca_result[:, :2], columns=['PC1', 'PC2'])
        if color_by_category:
            pca_df['Category'] = df['performance_category']
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Category',
                                 title="PCA: PC1 vs PC2",
                                 color_discrete_sequence=px.colors.qualitative.Set1)
        else:
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', title="PCA: PC1 vs PC2")
        
        st.plotly_chart(fig_pca, use_container_width=True)
        
    st.subheader("PCA Feature Loadings (PC1 & PC2)")
    # Show contribution of each original feature to PC1 and PC2
    loadings = pd.DataFrame(
        pca.components_.T[:, :2], 
        columns=['PC1', 'PC2'], 
        index=features
    )
    # Sort by absolute value to see top contributors easily
    loadings['Max_Abs'] = loadings.abs().max(axis=1)
    loadings = loadings.sort_values(by='Max_Abs', ascending=False).drop('Max_Abs', axis=1)
    
    st.dataframe(loadings.style.background_gradient(cmap='RdBu', vmin=-0.5, vmax=0.5))

with tab3:
    st.header("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    st.info("t-SNE places similar high-dimensional observations near one another in a 2D map, making local patterns easier to visualize.")
    
    tsne_df = pd.DataFrame(data=tsne_result, columns=['Dim 1', 'Dim 2'])
    
    if color_by_category:
        tsne_df['Category'] = df['performance_category']
        fig_tsne = px.scatter(tsne_df, x='Dim 1', y='Dim 2', color='Category',
                              title=f"t-SNE Visualization (Perplexity: {perplexity})",
                              color_discrete_sequence=px.colors.qualitative.Set1)
    else:
        fig_tsne = px.scatter(tsne_df, x='Dim 1', y='Dim 2', 
                              title=f"t-SNE Visualization (Perplexity: {perplexity})")
                              
    st.plotly_chart(fig_tsne, use_container_width=True)

with tab4:
    st.header("PCA vs t-SNE")
    
    comparison_data = {
        "Feature": ["Method", "Type", "Main Purpose", "What it preserves", "Interpretability", "Best use in this case study"],
        "PCA": [
            "Linear", 
            "Deterministic (mostly)",
            "Dimensionality reduction, variance maximization",
            "Global structure (large pairwise distances)",
            "High (Axes/Loadings mean something)",
            "Finding main drivers of performance (e.g., overall score vs participation)"
        ],
        "t-SNE": [
            "Non-linear",
            "Stochastic (depends on random seed)",
            "Visualization of high-dimensional data",
            "Local structure (small pairwise distances)",
            "Low (Axes have no distinct meaning)",
            "Revealing local clusters of students with similar profiles"
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.table(comp_df.set_index("Feature"))

with tab5:
    st.header("Key Findings")
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc1_pc2 = np.sum(pca.explained_variance_ratio_[:2]) * 100
    
    # Top features for PC1 and PC2 based on absolute loadings
    top_pc1 = np.abs(pca.components_[0]).argsort()[::-1][:3]
    top_pc2 = np.abs(pca.components_[1]).argsort()[::-1][:3]
    
    top_pc1_features = [features[i] for i in top_pc1]
    top_pc2_features = [features[i] for i in top_pc2]

    st.markdown(f"**1. Variance Explained:**")
    st.markdown(f"- The first principal component (PC1) explains **{var_pc1:.1f}%** of the variance.")
    st.markdown(f"- Together, PC1 and PC2 explain **{var_pc1_pc2:.1f}%** of the variance.")
    
    st.markdown(f"**2. Principal Components Meaning:**")
    st.markdown(f"- The top features contributing to PC1 are **{', '.join(top_pc1_features)}**.")
    st.markdown(f"- The top features contributing to PC2 are **{', '.join(top_pc2_features)}**.")
    
    st.markdown(f"**3. Visualization Observations:**")
    st.markdown("- **PCA:** The visualization suggests that performance categories (Excellent vs. Poor) are mostly separated along the first principal component, aligning with the variance explained by core academic scores.")
    st.markdown("- **t-SNE:** The embedding shows local groupings of students with similar multidimensional profiles. The distinct islands suggest that students share specific sub-patterns in their study and behavior habits, which correspond closely to their overall performance category.")
