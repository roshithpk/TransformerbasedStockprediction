# model_manager.py

import os
import torch
import streamlit as st
from tkinter import filedialog, Tk

# -- Save model to user-specified folder --
def save_model_dialog(model, stock_name, last_date):
    st.info("📦 Saving model...")

    # Open file dialog using Tkinter
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select folder to save model")
    root.destroy()

    if folder_selected:
        file_path = os.path.join(folder_selected, f"{stock_name.upper()}_model.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "last_trained_date": str(last_date)
        }, file_path)
        st.success(f"✅ Model saved at:\n`{file_path}`")
    else:
        st.warning("⚠️ Save cancelled. Model not saved.")

# -- Load model from user-specified file --
def load_model_dialog(model_class, input_size):
    st.info("📂 Loading saved model...")

    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")], title="Select saved model")
    root.destroy()

    if file_path and os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        last_trained_date = checkpoint.get("last_trained_date", "Unknown")
        st.success(f"✅ Model loaded from `{os.path.basename(file_path)}`")
        st.warning(f"⚠️ This model was trained only up to **{last_trained_date}**. Forecast accuracy may be impacted.")

        return model
    else:
        st.warning("⚠️ Load cancelled or invalid file.")
        return None
