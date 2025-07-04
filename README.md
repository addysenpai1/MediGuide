markdown
Copy
Edit
# 🧠 AI Medical Portal

An advanced **multi-module AI medical portal** built with **Streamlit**, enabling early disease prediction from user inputs such as **text, images, and audio**. The platform supports detection for multiple conditions:

### 🩺 Supported Diseases

- ❤️ Heart Disease  
- 🧬 Breast Cancer  
- 🧠 Alzheimer’s Disease  
- 🎧 Parkinson’s Disease  
- ⚡ Migraine  
- 🌞 Skin Disease (via image uploads)

---

## 🌟 Features

- **Streamlit Interface**: Simple, intuitive web interface for end-users.

- **Multi-Modal Inputs**:
  - 📝 Structured data (e.g., patient info, symptoms)
  - 🖼️ Medical images (for skin disease and Alzheimer’s)
  - 🔊 Audio samples (for Parkinson’s)

- **Machine Learning Models**:
  - 📊 XGBoost, Random Forest for tabular data
  - 🧠 CNN-based Transfer Learning for image-based predictions
  - 🎵 Signal processing + ML for audio-based diseases

- **Real-time Predictions** with optional **SHAP-based model explainability**

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, NumPy, Pandas, OpenCV, Librosa  
- **ML/DL**: XGBoost, Scikit-learn, TensorFlow/Keras (for Transfer Learning)

---

## 🛠️ Installation

Follow the steps below to set up the project locally in a Python 3.11 environment.

1. Download or clone all files from the repository into a Python 3.11 environment, preferably using Anaconda.  
2. Create and activate a virtual environment using Python 3.11.  
3. Install the required libraries including streamlit, numpy, pandas, scikit-learn, xgboost, tensorflow, keras, opencv-python, and librosa.  
4. Train or obtain pretrained model files for each disease (heart, cancer, Parkinson’s, etc.) and save them as `.pkl` files in the appropriate folders.  
5. Ensure that all model file paths in the code are correctly pointing to the local files.  
6. (Optional) Install the LLaMA 3 (3.2B) model using frameworks like LLaMA.cpp, Ollama, or Hugging Face Transformers, and update the API link or local endpoint in the code (e.g., http://localhost:11434/api).  
7. Launch the application by running the main script with Streamlit.
