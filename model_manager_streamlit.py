# model_manager_streamlit.py
import streamlit as st
import os
import torch

# --- Save Model using Streamlit text input ---
def save_model_streamlit(model, stock_name):
    st.markdown("### 💾 Save Trained Model")
    save_model = st.radio("Do you want to save the model?", ("No", "Yes"))
    if save_model == "Yes":
        save_path = st.text_input("Enter path to save the model (folder will be created if it doesn't exist):", value="saved_models")
        if st.button("Save Now"):
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, f"{stock_name}.pt")
            torch.save(model.state_dict(), full_path)
            st.success(f"✅ Model saved to `{full_path}`")

# --- Load Model using Streamlit file uploader ---
def load_model_streamlit(model):
    st.markdown("### 📂 Load Saved Model")
    use_saved = st.radio("Use saved model instead of training?", ("No", "Yes"))
    if use_saved == "Yes":
        uploaded_file = st.file_uploader("Upload the saved model (.pt) for this stock:", type=["pt"])
        if uploaded_file is not None:
            try:
                buffer = uploaded_file.read()
                with open("temp_model.pt", "wb") as f:
                    f.write(buffer)
                model.load_state_dict(torch.load("temp_model.pt"))
                st.success("✅ Model loaded successfully.")
                st.warning("⚠️ This model is trained up to its last saved date. Accuracy may degrade with time.")
                return model
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
    return None
