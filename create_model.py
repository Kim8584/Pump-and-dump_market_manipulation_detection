import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_single_model(df, identifier):
    """
    Trains a Random Forest model on the given dataframe and saves it.
    Args:
        df (pd.DataFrame): The input dataframe (e.g., df_5s or df_25s).
        identifier (str): A string to identify the model (e.g., '5s', '25s').
    """
    MODEL_FILE = f'random_forest_{identifier}_model.joblib'
    SCALER_FILE = f'scaler_{identifier}.joblib' # Save scaler as well for prediction

    print(f"Training model for {identifier} data...")
    
    # 1. Prepare the data
    X = df.drop(columns=['date', 'symbol', 'gt'])
    y = df['gt']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Scale test data too, though not used in training here

    # Save the scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler for {identifier} data saved to {SCALER_FILE}")
    
    # Train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # 5. Save the trained model to a file
    joblib.dump(rf, MODEL_FILE)
    print(f"Model for {identifier} data trained and saved to {MODEL_FILE}")

if __name__ == "__main__":
    # Define file paths for the raw data
    data_5s_path = 'labaled_features/features_5S.csv.gz'
    data_25s_path = 'labaled_features/features_25S.csv.gz'

    # Define model and scaler file names
    model_5s_file = 'random_forest_5s_model.joblib'
    scaler_5s_file = 'scaler_5s.joblib'
    model_25s_file = 'random_forest_25s_model.joblib'
    scaler_25s_file = 'scaler_25s.joblib'

    # Check if both models already exist
    if os.path.exists(model_5s_file) and os.path.exists(model_25s_file) and \
       os.path.exists(scaler_5s_file) and os.path.exists(scaler_25s_file):
        print("Both 5s and 25s model files and scalers already exist. Skipping training.")
    else:
        print("One or both model files/scalers are missing. Starting training for both.")
        
        # Load datasets
        df_5s = pd.read_csv(data_5s_path, compression='gzip')
        df_25s = pd.read_csv(data_25s_path, compression='gzip')
        
        # Train and save model for 5s data
        train_and_save_single_model(df_5s, '5s')
        
        # Train and save model for 25s data
        train_and_save_single_model(df_25s, '25s')

    print("\nDemonstrating loading and prediction for 5s model (example)...")
    # Example of how to load and use the 5s model
    if os.path.exists(model_5s_file) and os.path.exists(scaler_5s_file):
        loaded_rf_5s_model = joblib.load(model_5s_file)
        loaded_scaler_5s = joblib.load(scaler_5s_file)

        df_5s_original = pd.read_csv(data_5s_path, compression='gzip')
        X_5s_original = df_5s_original.drop(columns=['date', 'symbol', 'gt'])
        
        # Get a sample for prediction
        sample_data_5s = X_5s_original.head(1)
        sample_data_5s_scaled = loaded_scaler_5s.transform(sample_data_5s)

        prediction_5s = loaded_rf_5s_model.predict(sample_data_5s_scaled)
        print(f"5s Model Prediction for sample data: {'Pump-and-Dump' if prediction_5s[0] == 1 else 'Normal Event'}")
    else:
        print("5s model or scaler not found, cannot demonstrate prediction.")
