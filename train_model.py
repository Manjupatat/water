import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split


# Load and preprocess the dataset
# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert only numeric columns to numeric, and exclude non-numeric columns like 'Station' and 'Date'
    numeric_data = data.select_dtypes(include=[np.number])

    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

    # Define synthetic target (Water Quality Index, WQI)
    data_scaled['WQI'] = (
        0.3 * data_scaled['NITRATE(PPM)'] +
        0.2 * data_scaled['PH'] +
        0.15 * data_scaled['AMMONIA(mg/l)'] +
        0.1 * data_scaled['TEMP'] +
        0.1 * data_scaled['DO'] +
        0.1 * data_scaled['TURBIDITY'] +
        0.05 * data_scaled['MANGANESE(mg/l)']
    )

    # Split into features (X) and target (y)
    X = data_scaled.drop(columns=['WQI']).values
    y = data_scaled['WQI'].values

    # Reshape X for CNN input: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler


# Build the CNN model
def build_model(input_shape):
    model = Sequential([
        SeparableConv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        SeparableConv1D(filters=32, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')  # Linear activation for regression (WQI)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['mae'])
    return model

# Train and save the model
def train_and_save_model(file_path='data/training.csv', model_save_path='model/trained_wqi_model.h5', scaler_save_path='model/scaler.pkl'):
    X, y, scaler = load_and_preprocess_data(file_path)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(input_shape=(X.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, validation_split=0.2, epochs=5000, batch_size=32, verbose=1)

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the scaler
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    # joblib.dump(model,scaler_save_path)

    print(f"Scaler saved to {scaler_save_path}")

# Call the training and saving function
train_and_save_model()
