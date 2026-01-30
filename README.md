# 🩺 🫀 Heart Disease Predictor

📌 Overview

The Heart Disease Predictor is a machine learning project designed to predict heart diseases based on patient health data. The model is trained using classification techniques to analyze symptoms/medical attributes and provide predictions that can assist in healthcare diagnostics.

This project is implemented in Python with Jupyter Notebook using PyCharm IDE and leverages machine learning libraries for training and evaluation.

🚀 Features

Data preprocessing and cleaning for health-related datasets

Training ML models for disease prediction

Model evaluation with accuracy and metrics

Exporting trained model for reuse

Easy-to-use interface via Jupyter Notebook

Deployment as a web app for user-friendly interaction

🛠️ Technologies Used

Python 3.13.9

Kaggle → Dataset input

NumPy, Pandas → Data handling

Scikit-learn → Machine learning algorithms

Matplotlib, Seaborn → Visualization

Joblib → Model persistence

Streamlit → Model deployment

📂 Project Structure

    │── Disease_Predictor.ipynb # Main Jupyter Notebook
    
    │── requirements.txt # List of dependencies
    
    │── README.md # Project documentation
    
    │── heart_rf_model.pkl # Saved ML model

    │── heart_scaler.pkl # Saved ML model scaler
    
    │── data/ # Dataset

    │── Heart_user_template.csv # User template for data input

    │── heart_app.py # Web deployment app

⚙️ Installation

Clone the repository:

    git clone https://github.com/sayantan-de10/Heart-Disease-Predictor.git
    
    cd Heart-Disease-Predictor

Install Python if not already present on your system.

Install dependencies:

    pip install -r requirements.txt

Log in to Kaggle and generate an API key for uploading as kaggle.json

▶️ Usage

Open PyCharm.

Run the Disease Predictor.ipynb file step by step.

Train the model and generate predictions.

For deployment, open the terminal and run:

    streamlit run heart_app.py

👨‍💻 Author

Sayantan De

💼 LinkedIn - https://www.linkedin.com/in/sayantan-de-pulsar16

📧 Email: sayantande612@gmail.com
