import os
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image

from preprocess import preprocess_signature
from model import SiameseEmbeddingNet

# Page configuration
st.set_page_config(
    page_title="Offline Signature Verification",
    page_icon="✍️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(weights_path="models/siamese_signature.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseEmbeddingNet(embedding_dim=128)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        status = "Loaded trained model weights."
    else:
        status = "No trained model weights found (using initialized model for preview)."
    model.to(device)
    model.eval()
    return model, device, status

st.title("✍️ Offline Signature Identification & Verification System")
st.markdown(
    "Final Year Computer Engineering Capstone | **Deep Metric Learning with Siamese CNN & Contrastive Loss**"
)

model, device, status_msg = load_model()
st.sidebar.info(f"Model Status: {status_msg}")
st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")

st.sidebar.header("⚙️ Verification Settings")
threshold = st.sidebar.slider(
    "Decision Distance Threshold (τ)",
    min_value=0.20,
    max_value=1.50,
    value=0.65,
    step=0.05,
    help="Pairwise Euclidean distance below this value is classified as GENUINE; above or equal is classified as FORGED / IMPOSTOR."
)

# Quick Demo Preset Selector
st.sidebar.header("⚡ 1-Click Demo Presets")
demo_options = {
    "None (Manual Upload)": None,
    "Genuine Match (Student 1 vs Student 1)": ("demo_samples/anchor_student1.png", "demo_samples/genuine_student1.png"),
    "Skilled Forgery (Student 1 vs Forgery)": ("demo_samples/anchor_student1.png", "demo_samples/forged_student1.png"),
    "Impostor / Random (Student 1 vs Student 2)": ("demo_samples/anchor_student1.png", "demo_samples/impostor_student2.png"),
}
selected_preset = st.sidebar.selectbox("Load Sample Pair:", list(demo_options.keys()))

col1, col2 = st.columns(2)

ref_image = None
query_image = None

# Check preset selection
if demo_options[selected_preset] is not None:
    ref_path, query_path = demo_options[selected_preset]
    if os.path.exists(ref_path) and os.path.exists(query_path):
        ref_image = Image.open(ref_path)
        query_image = Image.open(query_path)

with col1:
    st.subheader("1. Reference / Anchor Signature")
    ref_file = st.file_uploader(
        "Upload Genuine Signature from Student Record",
        type=["png", "jpg", "jpeg"],
        key="ref_upload"
    )
    if ref_file:
        ref_image = Image.open(ref_file)
    if ref_image:
        st.image(ref_image, caption="Anchor Signature", use_container_width=True)

with col2:
    st.subheader("2. Query / Questioned Signature")
    query_file = st.file_uploader(
        "Upload Test / Questioned Signature to Verify",
        type=["png", "jpg", "jpeg"],
        key="query_upload"
    )
    if query_file:
        query_image = Image.open(query_file)
    if query_image:
        st.image(query_image, caption="Test / Questioned Signature", use_container_width=True)

st.divider()

if ref_image and query_image:
    if st.button("🚀 Run Biometric Verification", type="primary", use_container_width=True):
        with st.spinner("Executing image preprocessing & deep feature extraction..."):
            # 1. Preprocess
            ref_proc = preprocess_signature(ref_image)
            query_proc = preprocess_signature(query_image)

            # 2. Show preprocessed outputs
            st.write("#### 🔬 Preprocessed Inputs (Binarized, Bounding-Box Cropped, 128x128)")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(ref_proc, caption="Anchor Processed", clamp=True, width=160)
            with p_col2:
                st.image(query_proc, caption="Query Processed", clamp=True, width=160)

            # 3. Model inference
            t_ref = torch.from_numpy(ref_proc).unsqueeze(0).unsqueeze(0).float().to(device)
            t_query = torch.from_numpy(query_proc).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                emb_ref = model.forward_once(t_ref)
                emb_query = model.forward_once(t_query)
                dist = F.pairwise_distance(emb_ref, emb_query).item()
                similarity_pct = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100.0))

            # 4. Display verdict
            st.divider()
            st.write("### 📊 Verification Result")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric("Euclidean Distance", f"{dist:.4f}", help="Lower distance means higher structural similarity")
                st.metric("Similarity Score", f"{similarity_pct:.2f}%")

            with res_col2:
                is_match = dist < threshold
                if is_match:
                    st.success("### ✅ VERIFIED: GENUINE MATCH")
                    st.write(f"The signatures match the student profile. Distance **{dist:.4f}** is below threshold **{threshold:.2f}**.")
                else:
                    st.error("### ❌ REJECTED: FORGERY / MISMATCH")
                    st.write(f"Significant structural deviation detected. Distance **{dist:.4f}** exceeds threshold **{threshold:.2f}**.")
else:
    st.info("👆 Please upload signatures or choose a **1-Click Demo Preset** from the sidebar.")
