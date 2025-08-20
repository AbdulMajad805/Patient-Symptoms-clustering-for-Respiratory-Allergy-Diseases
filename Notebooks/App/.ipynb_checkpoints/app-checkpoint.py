import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# --------------------------
# Load trained KMeans model & scaler
# --------------------------
kmeans = joblib.load("kmeans_symptom_model.pkl")   # trained KMeans
scaler = joblib.load("scaler.pkl")                 # saved StandardScaler/MinMaxScaler

# Load your dataset (replace with correct path)
data = pd.read_csv("symptom_data.csv")   # <-- must be the SAME dataset used for training

# --------------------------
# Feature Selection (must match training features)
# --------------------------
# If your dataset has extra columns (like ID, target), drop them here:
symptoms = [
    "FEVER","COUGH","SORE_THROAT","SNEEZING","ITCHY_NOSE","ITCHY_EYES",
    "ITCHY_INNER_EAR","PINK_EYE","NAUSEA","VOMITING","DIARRHEA",
    "DIFFICULTY_BREATHING","SHORTNESS_OF_BREATH","LOSS_OF_TASTE","MUSCLE_ACHES",
    "ITCHY_MOUTH","LOSS_OF_SMELL","RUNNY_NOSE","STUFFY_NOSE","TIREDNESS"
]

X = data[scaler.feature_names_in_] 
X_scaled = scaler.transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Cluster names
cluster_names = {
    0: "🌿 Allergy Related Symptoms",
    1: "🤒 COVID-like Symptoms",
    2: "🌸 Allergy Variant Symptoms",
    3: "🤢 Flu-like / Gastro Symptoms"
}

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Symptom Clustering App", page_icon="🩺", layout="wide")

st.title("🩺 Symptom Clustering Prediction App")
st.markdown("This app predicts **which cluster of symptoms you belong to** using KMeans clustering.")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Adjust app parameters")

show_plot = st.sidebar.checkbox("Show PCA Cluster Plot", True)

# --------------------------
# Symptom Selection
# --------------------------
st.subheader("✅ Select Your Symptoms")

# Organize symptoms in 3 columns
cols = st.columns(3)
user_input = []
for i, sym in enumerate(symptoms):
    with cols[i % 3]:
        val = st.checkbox(sym)
        user_input.append(1 if val else 0)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict Cluster"):
    user_input = np.array(user_input).reshape(1, -1)
    user_input_df = pd.DataFrame(user_input, columns=symptoms)

    # Reorder columns to match training
    user_input_df = user_input_df[scaler.feature_names_in_]

    # Scale input
    user_input_scaled = scaler.transform(user_input_df)

    # Predict cluster
    cluster = kmeans.predict(user_input_scaled)[0]

    # Display styled result card
    st.markdown(f"""
    <div style="padding:20px; border-radius:15px; background-color:#e6f7ff; border:1px solid #90caf9;">
    <h3 style="color:#1565c0;">🩺 Predicted Cluster: {cluster_names[cluster]}</h3>
    <p style="color:#1565c0;"><b>Cluster ID:</b> {cluster}</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # PCA Visualization
    # --------------------------
    if show_plot:
        user_data_pca = pca.transform(user_input_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        labels = kmeans.predict(X_scaled)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.4)
        ax.scatter(user_data_pca[0, 0], user_data_pca[0, 1], color="red", s=200, marker="X", label="You")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Projection with Your Position")
        ax.legend()
        st.pyplot(fig)

# --------------------------
# Cluster Info Section
# --------------------------
with st.expander("📊 See Cluster Details"):
    df_clusters = pd.DataFrame.from_dict(cluster_names, orient="index", columns=["Cluster Meaning"])
    st.table(df_clusters)
