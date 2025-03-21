import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import io
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import datetime
import time
import os


st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load model and scaler
try:
    model = load_model('model/trained_wqi_model.h5')
    scaler = joblib.load('model/scaler.pkl')
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = f"Error: Unable to connect to the server. Details: {str(e)}"

# Load dataset
@st.cache_data
def load_dataset():
    try:
        dataset = pd.read_csv('data/testing.csv')
        return dataset
    except Exception as e:
        return None

# Function for user input
def get_user_input():
    st.write("\n### Enter values for the water quality parameters:")
    nitrate = st.slider("Nitrate (PPM) [0-140]:", 0.0, 140.0, 23.0, 0.1)
    ph = st.slider("pH [0-10]:", 0.0, 10.0, 7.0, 0.1)
    ammonia = st.slider("Ammonia (mg/L) [0-2]:", 0.0, 2.0, 1.0, 0.1)
    temp = st.slider("Temperature (°C) [14-42]:", 14.0, 42.0, 30.0, 0.1)
    do = st.slider("Dissolved Oxygen (mg/L) [6.5-12]:", 6.5, 12.0, 8.0, 0.1)
    turbidity = st.slider("Turbidity [0-400]:", 0.0, 400.0, 50.0, 0.1)
    manganese = st.slider("Manganese (mg/L) [0.5-3.6]:", 0.5, 3.6, 1.0, 0.1)

    user_data = pd.DataFrame([[nitrate, ph, ammonia, temp, do, turbidity, manganese]],
                             columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY',
                                      'MANGANESE(mg/l)'])
    return user_data

# Prediction function
def predict_wqi(input_data):
    input_data_normalized = scaler.transform(input_data)
    input_data_normalized = input_data_normalized.reshape(1, input_data_normalized.shape[1], 1)
    predicted_wqi = model.predict(input_data_normalized)[0][0]
    return predicted_wqi


def live_monitoring(dataset, model_loaded):
    st.title("Live Monitoring with Visualization and Report Download")

    # Check if the dataset is empty
    if dataset.empty:
        st.error("Dataset is empty. Please provide valid data.")
        return

    if model_loaded:
        # Define the required features used during training
        required_features = ['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)']

        # Filter the dataset to include only the required features
        dataset = dataset[required_features]

        # Containers for visualization and live updates
        placeholder_table = st.empty()
        placeholder_chart = st.empty()
        placeholder_status = st.empty()
        placeholder_report = st.empty()

        # Store live data for visualization and report generation
        live_data_records = pd.DataFrame(columns=required_features + ['Predicted_WQI', 'Status'])
        if "report_downloaded" not in st.session_state:
            st.session_state.report_downloaded = False

        # Create a CSV file if it doesn't exist yet
        csv_file_path = 'data/Predicted.csv'
        if not os.path.exists(csv_file_path):
            live_data_records.to_csv(csv_file_path, index=False, mode='w')  # Create the CSV with headers
        else:
            live_data_records.to_csv(csv_file_path, index=False, mode='a', header=False)  # Append to the CSV

        for index, row in dataset.iterrows():
            live_data = row.to_frame().T  # Reshape row into DataFrame

            # Predict WQI
            predicted_wqi = predict_wqi(live_data)
            if predicted_wqi < 0.3:
                status = "Unsafe"
            elif 0.3 <= predicted_wqi < 0.7:
                status = "Good"
            else:
                status = "Safe"

            # Add prediction to live data
            live_data['Predicted_WQI'] = predicted_wqi
            live_data['Status'] = status
            live_data_records = pd.concat([live_data_records, live_data], ignore_index=True)

            # Update table
            with placeholder_table.container():
                st.write("### Current Water Quality Data:")
                st.write(live_data)

            # Display status
            with placeholder_status.container():
                st.write(f"### Water Quality Status: **{status}**")
                st.write(f"### Predicted WQI: **{predicted_wqi:.4f}**")

            # Update chart
            with placeholder_chart.container():
                st.write("### Live Monitoring Visualization")
                  # Visualize predicted WQI over time
                st.line_chart(live_data_records[['Predicted_WQI']])  # Visualize predicted WQI over time

                # Generate report option (display only once)
                # if index == len(dataset) - 1:

            # Generate report option
            with placeholder_report.container():
                st.write("### Download Prediction Report")
                if st.button(f"Generate Report for Record {index + 1}"):
                    # Generate PDF report
                    pdf = generate_pdf_report(
                        live_data[required_features],
                        predicted_wqi,
                        status,
                        f"Record {index + 1}: Real-time Monitoring Report"
                    )

                    # Define report filename
                    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    file_name = f"live_monitoring_report_{current_time}.pdf"

                    # Provide download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf,
                        file_name=file_name,
                        mime="application/pdf"
                    )

                    #set session state to prevent rerunning and dowloading multiple times
                    st.session_state.report_downloaded=True
            time.sleep(10)  # Simulate a 2-second interval for live updates

        # st.experimental_rerun()
    else:
        st.error(error_message)


def generate_pdf_report(input_data, predicted_wqi, status, title="Water Quality Report"):
    """
    Generate a PDF report for water quality prediction.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, title)

    # Input Parameters Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Input Parameters:")
    c.setFont("Helvetica", 10)
    y = height - 120
    for column, value in zip(input_data.columns, input_data.values[0]):
        c.drawString(70, y, f"{column}: {value}")
        y -= 20

    # Prediction Results Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Prediction Result:")
    c.setFont("Helvetica", 10)
    c.drawString(70, y - 30, f"Predicted WQI: {predicted_wqi:.4f}")
    c.drawString(70, y - 50, f"Water Quality Status: {status}")

    # Causing Parameters and Solutions (if status is Unsafe)
    if status == "Unsafe":
        c.drawString(50, y - 70, "Causing Parameters and Solutions:")
        y -= 90
        thresholds = {
            "NITRATE(PPM)": 50.0,
            "PH": (6.5, 8.5),
            "AMMONIA(mg/l)": 0.5,
            "TEMP": 35.0,
            "DO": 6.5,
            "TURBIDITY": 5.0,
            "MANGANESE(mg/l)": 0.1
        }
        issues = {}
        for column in input_data.columns:
            value = input_data[column].values[0]
            if isinstance(thresholds[column], tuple):  # Range thresholds
                if not (thresholds[column][0] <= value <= thresholds[column][1]):
                    issues[column] = value
            else:  # Single value threshold
                if value > thresholds[column]:
                    issues[column] = value

        solutions_dict = {
            "NITRATE(PPM)": "Reduce agricultural runoff and improve wastewater treatment.",
            "PH": "Add alkaline or acidic substances to adjust pH levels.",
            "AMMONIA(mg/l)": "Improve wastewater treatment or reduce agricultural runoff.",
            "TEMP": "Introduce cooling systems to reduce water temperature.",
            "DO": "Increase aeration or reduce organic pollution.",
            "TURBIDITY": "Install filtration systems or reduce erosion.",
            "MANGANESE(mg/l)": "Improve filtration and water treatment processes."
        }

        for param, value in issues.items():
            c.drawString(70, y, f"{param}: {value}")
            c.drawString(70, y - 20, f"Solution: {solutions_dict.get(param, 'No solution available.')}")
            y -= 40

    # Save the PDF
    c.save()
    buffer.seek(0)
    return buffer


def categorize_wqi(predicted_wqi):
    if predicted_wqi < 0.3:
        return "Poor", "WQI < 0.3"
    elif 0.3 <= predicted_wqi < 0.7:
        return "Good", "0.3 ≤ WQI < 0.7"
    else:
        return "Safe", "WQI ≥ 0.7"

# Dashboard layout
def main():
    menu = ["Data", "Visualization", "Predict", "Live Monitor"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Data":
        st.title("Dataset")
        dataset = load_dataset()
        if dataset is not None:
            st.write("### Dataset Preview (First 100 Rows)")
            st.dataframe(dataset.head(100))
        else:
            st.error("Error loading dataset.")

    elif choice == "Visualization":
        st.title("Data Visualization")
        dataset = load_dataset()
        if dataset is not None:
            if 'Date' in dataset.columns:
                dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
                dataset = dataset.dropna(subset=['Date'])
                dataset['Month'] = dataset['Date'].dt.month

                st.write("### Select Month to Visualize")
                selected_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: pd.to_datetime(f"2024-{x:02d}-01").strftime('%B'))
                month_data = dataset[dataset['Month'] == selected_month]
                month_data.set_index('Date', inplace=True)
                attributes = st.multiselect("Select Attributes to Visualize", month_data.columns.drop(['Station', 'Month']), default=['NITRATE(PPM)', 'PH', 'TEMP'])
                if attributes:
                    st.line_chart(month_data[attributes])
                else:
                    st.warning("Select attributes to visualize.")
            else:
                st.error("No 'Date' column in dataset.")
        else:
            st.error("Error loading dataset.")

    elif choice == "Predict":
        st.title("Water Quality Prediction")
        user_input = get_user_input()
        if st.button("Predict"):
            if model_loaded:
                predicted_wqi = predict_wqi(user_input)
                status, range_desc = categorize_wqi(predicted_wqi)
                st.success(f"Predicted WQI: **{predicted_wqi:.4f}**")
                st.info(f"Water Quality Status: **{status}** ({range_desc})")
            else:
                st.error(error_message)

    elif choice == "Live Monitor":
        dataset = load_dataset()
        if dataset is not None:
            live_monitoring(dataset, model_loaded)
        else:
            st.error("Error loading dataset.")

if __name__ == "__main__":
    main()
