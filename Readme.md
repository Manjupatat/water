# Water Quality Monitoring System (Streamlit & Deep Learning)

This project is a **real-time water quality monitoring** system using a **Convolutional Neural Network (CNN)** for predicting the **Water Quality Index (WQI)**. Built with **Streamlit**, it provides an interactive dashboard for **data visualization, WQI prediction, and live monitoring**.

## 📌 Features
- **Data Visualization**: View real-time water quality data and trends.
- **WQI Prediction**: Uses a trained CNN model to predict water quality status.
- **Live Monitoring**: Continuously updates with real-time data and generates reports.
- **Downloadable Reports**: Generate PDF reports for monitoring insights.
- **User-friendly Interface**: Built with Streamlit for an intuitive experience.

## 📁 Project Structure
```
WaterQualityMonitoring/
│
├── model/                     # Folder containing trained model & scaler
│   ├── trained_wqi_model.h5    # Trained CNN model
│   ├── scaler.pkl              # MinMaxScaler object for normalization
│
├── data/                       # Dataset folder
│   ├── Ponds.csv               # Dataset file
│   ├── Predicted.csv           # Stores real-time predictions
│
├── streamlit_app.py            # Main Streamlit app script
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies file
```

## 🛠️ Installation & Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/water-quality-monitoring.git
   cd water-quality-monitoring
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\ctivate     # On Windows
   ```

3. **Install required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run
1. **Start the Streamlit app**  
   ```bash
   streamlit run streamlit_app.py
   ```
2. **Navigate through the menu**  
   - **Data**: View dataset preview.  
   - **Visualization**: Select a month to visualize water quality trends.  
   - **Predict**: Manually input water parameters for WQI prediction.  
   - **Live Monitor**: Real-time monitoring with visualization and report download.  

## 🖥️ Model Details
- Uses a **CNN with SeparableConv1D layers** for WQI prediction.
- Normalized inputs using **MinMaxScaler**.
- Trained with TensorFlow and evaluated using Mean Absolute Error (MAE).

## 🔧 Troubleshooting
- **ModuleNotFoundError?** Install missing dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

### 5️⃣ Run the App
```sh
streamlit run main.py
```

  
- **Dataset Not Found?** Ensure `testing.csv` is inside the `data/` folder.
- **Streamlit App Not Running?** Verify Python version (`python --version`).

## 📜 License
This project is for educational and research purposes.

---
👨‍💻 Developed for real-time water quality monitoring using deep learning and Streamlit.

---

## 💡 Contributing
Feel free to submit PRs, open issues, and contribute to making this bot even better! 😊

---

## 📞 Contact +91 - 8971812177
For any queries or suggestions, reach out to:  
📧 **Email**: manjupatat80@gmail.com  
🐙 **GitHub**: [Manjunath L Patat](https://github.com/Manjupatat)

**Contributor**: [@Revanth Sharma M](https://github.com/RevanthSharmaM)  contributed to this.
