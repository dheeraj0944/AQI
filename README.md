# 🌆 AQI – Air Quality Index Monitoring System

A Flask and Deep Learning–based web application that predicts and visualizes the Air Quality Index (AQI) using historical air pollution data. This project demonstrates an end-to-end pipeline from model training to deployment through a web interface.

---

## 🚀 Project Overview

Air pollution is a major environmental concern affecting public health. The Air Quality Index (AQI) helps quantify pollution levels and associated health risks. This system uses deep learning models to predict AQI values and presents results through an interactive Flask web application.

---

## 🧠 Features

- AQI prediction using deep learning models (CNN/LSTM with attention)
- Flask-based backend for model inference
- Interactive web interface for predictions
- Visualization of training performance (loss and MAE curves)
- User authentication and login system
- SQLite database integration

---

## 🛠 Tech Stack

- **Backend:** Python, Flask  
- **Machine Learning:** TensorFlow / Keras  
- **Frontend:** HTML, CSS, JavaScript, Jinja2  
- **Database:** SQLite  

---

## 📁 Project Structure

AQI/
├── BACKEND/
│ ├── Train.py
│ ├── test_model.py
│ └── plots/
├── static/
├── templates/
├── app.py
├── aqi_cnn_lstm_attention_model.keras
├── aqi_scaler.save
├── user_data.db
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

1. Clone the repository
   git clone https://github.com/dheeraj0944/AQI.git
   cd AQI

2. Install dependencies
   pip install -r requirements.txt

3. Run the application
   python app.py

Open your browser and visit:
http://localhost:5000

---

## 📊 Model & Visualizations

The repository includes trained deep learning models and performance plots such as:
- Training loss curve
- Mean Absolute Error (MAE) curve

These files are provided for direct evaluation without retraining.

---

## 📌 Usage

1. Start the Flask application
2. Register or log in as a user
3. Enter the required input parameters
4. View the predicted AQI results and visual outputs

---

## 📄 Notes

Trained models are included in the repository for demonstration and ease of evaluation.

---

## 📜 License

This project is developed for academic and educational purposes.

---
