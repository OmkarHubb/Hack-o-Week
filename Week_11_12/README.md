# Uncovering Hidden Student Learning Patterns Using PCA and t-SNE

## Project Objective
This is a minimal, educational case-study demonstrating how dimensionality reduction techniques—Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE)—can be used to visualize and understand patterns in a high-dimensional dataset of student academic performance.

## Dataset Description
The dataset (`data/student_data.csv`) is a synthetic, realistic dataset comprising 500 students and 15 numerical features. The features represent various aspects of student engagement and academic outcomes, such as `attendance`, `study_hours`, `assignment_score`, `midterm_score`, `sleep_hours`, `projects_completed`, and more. A derived `performance_category` (Poor, Average, Good, Excellent) is also included based on overall calculated scores.

## How PCA is Used
PCA is applied after standardizing the numerical features to find new orthogonal axes (Principal Components) that capture the maximum variance in the data. In the Streamlit app, we calculate the explained variance ratio, plot a scree plot, visualize the first two PCs in a 2D scatter plot, and examine the feature loadings to interpret what PC1 and PC2 represent.

## How t-SNE is Used
t-SNE is applied to the standardized feature matrix to embed the high-dimensional data into a 2D space. By emphasizing the preservation of local distances, t-SNE helps in mapping out local similarities and groups among students with similar academic and behavioral profiles. The perplexity parameter can be adjusted interactively to observe its effect on the mapping.

## PCA vs t-SNE
The project includes a direct comparison of the two techniques:
- **PCA** is linear, deterministic, preserves global structure, and is highly interpretable (via loadings).
- **t-SNE** is non-linear, stochastic, preserves local structure, and is primarily a tool for visualization rather than interpretability.

## Installation
Ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

## Run Command
Start the Streamlit application by running:

```bash
streamlit run app.py
```

## Limitations
- The dataset is synthetic; actual student data might contain noise, nonlinear relationships, or distinct distributions that aren't perfectly modeled here.
- t-SNE axes lack intrinsic meaning, so distances between distant clusters in the t-SNE plot shouldn't be over-interpreted.
- t-SNE results are sensitive to hyperparameters like perplexity, which must be tuned carefully.
