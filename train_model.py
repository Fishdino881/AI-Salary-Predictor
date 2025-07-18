# train_model.py

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# ---------- Configuration ----------
CONFIG = {
    "data_path": "data/salary_data.csv", # Ensure this CSV exists and contains 'Management' and 'PerformanceRating' columns
    "test_size": 0.2,
    "random_state": 42,
    "model_params": {
        "n_estimators": [100, 200], # Reduced for faster tuning
        "max_depth": [10, 20],      # Reduced for faster tuning
        "min_samples_split": [2, 5] # Reduced for faster tuning
    },
    "categorical_cols": [
        "EducationLevel", 
        "JobRole", 
        "Location", 
        "CompanyType", 
        "SkillLevel", 
        "RemoteWork"
    ],
    "numerical_cols": [
        "YearsExperience",
        "NumCertifications"
    ],
    "additional_features": [ # These MUST be included and handled
        "Management",
        "PerformanceRating"
    ],
    "target_col": "Salary",
    "artifact_dir": "artifacts"
}

# ---------- Setup Logging ----------
def setup_logging():
    """Configure logging for the training process"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------- Helper Functions ----------
def create_artifact_dir():
    """Create directory for saving model artifacts"""
    os.makedirs(CONFIG["artifact_dir"], exist_ok=True)
    logger.info(f"Created artifact directory at {CONFIG['artifact_dir']}")

def save_metadata(metrics, best_params, X_shape):
    """Save training metadata and metrics"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestRegressor",
        "metrics": metrics,
        "best_params": best_params,
        "features": {
            "categorical": CONFIG["categorical_cols"],
            "numerical": CONFIG["numerical_cols"],
            "additional": CONFIG["additional_features"] # Include additional features in metadata
        },
        "data_shape_trained": X_shape # Store the shape of X used for training
    }
    
    # Create a timestamped directory for this run's artifacts
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_artifact_path = os.path.join(CONFIG["artifact_dir"], f"salary_prediction_{timestamp_dir}")
    os.makedirs(current_artifact_path, exist_ok=True)

    with open(f"{current_artifact_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved training metadata to {current_artifact_path}/metadata.json")

# ---------- Data Loading & Validation ----------
def load_and_validate_data():
    """Load and validate the input dataset"""
    logger.info(f"Loading data from {CONFIG['data_path']}")
    
    try:
        df = pd.read_csv(CONFIG["data_path"])
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)
    
    # Validate columns - ensure all defined features and target are present
    all_features = CONFIG["categorical_cols"] + CONFIG["numerical_cols"] + CONFIG["additional_features"]
    required_cols = all_features + [CONFIG["target_col"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns in '{CONFIG['data_path']}': {missing_cols}")
        sys.exit(1)
    
    # Check for null values and handle them
    if df.isnull().sum().sum() > 0:
        logger.warning("Dataset contains null values. Handling them...")
        # Fill categorical NaNs with 'Missing'
        for col in CONFIG["categorical_cols"]:
            if col in df.columns:
                df[col] = df[col].fillna("Missing")
        # Fill numerical NaNs with 0 or mean, depending on context. Using 0 for simplicity.
        for col in CONFIG["numerical_cols"] + CONFIG["additional_features"]:
            if col in df.columns:
                df[col] = df[col].fillna(0) # Fill numerical NaNs with 0
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

# ---------- Feature Engineering ----------
def preprocess_features(df):
    """Preprocess and transform features"""
    logger.info("Preprocessing features...")
    
    encoders = {}
    # Encode standard categorical columns
    for col in CONFIG["categorical_cols"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
        # Save encoder
        # We'll save encoders to a timestamped directory later, not here directly
    
    # Handle PerformanceRating separately as it's ordinal and has a fixed mapping
    # This mapping must be consistent with app.py
    performance_mapping = {
        "Below Average": 0,
        "Average": 1,
        "Above Average": 2,
        "Exceptional": 3,
        "Top Performer": 4
    }
    if 'PerformanceRating' in df.columns:
        # Ensure all values in 'PerformanceRating' are in the mapping
        unknown_performance_values = set(df['PerformanceRating'].unique()) - set(performance_mapping.keys())
        if unknown_performance_values:
            logger.error(f"Found unknown values in 'PerformanceRating' column: {unknown_performance_values}. Please check your data.")
            sys.exit(1)
        df['PerformanceRating'] = df['PerformanceRating'].map(performance_mapping)
    else:
        logger.warning("'PerformanceRating' column not found after loading. This should not happen if data validation passes.")

    # Ensure 'Management' is treated as numerical (0 or 1)
    if 'Management' in df.columns:
        df['Management'] = df['Management'].astype(int) # Convert boolean or other types to 0/1
    else:
        logger.warning("'Management' column not found after loading. This should not happen if data validation passes.")
            
    logger.info("Categorical and additional features preprocessed.")
    return df, encoders

# ---------- Model Training ----------
def train_model(X_train, y_train):
    """Train model with hyperparameter tuning"""
    logger.info("Starting model training...")
    
    # Define which columns are numerical and which are categorical (after LabelEncoding)
    # The categorical columns are already numerical after preprocess_features
    # So, all features for the regressor will be numerical.
    # The StandardScaler should only apply to the original numerical columns.
    
    # Separate the original numerical and the now-numerical-categorical columns
    # The ColumnTransformer needs to know which columns to scale vs. pass through.
    
    # The features passed to the pipeline are all 10 features
    # (6 encoded categorical + 2 original numerical + 2 additional numerical)
    
    # The StandardScaler should only apply to the original numerical columns + additional numerical columns
    # because categorical columns are already encoded to numbers (0 to N-1) and don't need scaling in this context
    # for a tree-based model like RandomForest.
    
    numerical_features_for_scaling = CONFIG["numerical_cols"] + CONFIG["additional_features"]
    categorical_features_for_passthrough = CONFIG["categorical_cols"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_for_scaling),
            ('cat', 'passthrough', categorical_features_for_passthrough)
        ],
        remainder='drop' # Drop any columns not specified
    )
    
    # Create model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=CONFIG["random_state"]))
    ])
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline,
        param_grid={
            'regressor__n_estimators': CONFIG["model_params"]["n_estimators"],
            'regressor__max_depth': CONFIG["model_params"]["max_depth"],
            'regressor__min_samples_split': CONFIG["model_params"]["min_samples_split"]
        },
        cv=3, # Reduced CV for faster execution
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return grid_search.best_estimator_, grid_search.best_params_

# ---------- Model Evaluation ----------
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model...")
    
    predictions = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions)
    }
    
    logger.info(f"Model Metrics:\n"
                f"MAE: {metrics['mae']:.2f}\n"
                f"RMSE: {metrics['rmse']:.2f}\n"
                f"R2: {metrics['r2']:.2f}")
    
    return metrics

# ---------- Main Execution ----------
if __name__ == "__main__":
    start_time = time.time()
    logger.info("🚀 Starting model training pipeline")
    
    try:
        # Step 1: Setup
        create_artifact_dir()
        
        # Step 2: Data preparation
        df = load_and_validate_data()
        df, encoders = preprocess_features(df)
        
        # Step 3: Train-test split
        # Define the exact order of features that will be used for training
        # This order MUST match the order expected by app.py
        all_feature_cols = CONFIG["categorical_cols"] + CONFIG["numerical_cols"] + CONFIG["additional_features"]
        
        X = df[all_feature_cols] # Ensure X contains all 10 features in the correct order
        y = df[CONFIG["target_col"]]
        
        logger.info(f"Features (X) shape before split: {X.shape}")
        logger.info(f"Features used for training (order matters!): {X.columns.tolist()}")
        logger.info(f"Number of features used for training: {X.shape[1]}") 

        # Critical check: Ensure X has exactly 10 features
        if X.shape[1] != 10:
            logger.error(f"Expected 10 features for training, but found {X.shape[1]}. Please check your data and CONFIG.")
            sys.exit(1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=CONFIG["test_size"], 
            random_state=CONFIG["random_state"]
        )
        
        # Step 4: Model training
        model, best_params = train_model(X_train, y_train)
        
        # Step 5: Model evaluation
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 6: Save artifacts
        # Create a timestamped directory for this run's artifacts
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_artifact_path = os.path.join(CONFIG["artifact_dir"], f"salary_prediction_{timestamp_dir}")
        os.makedirs(current_artifact_path, exist_ok=True)

        with open(f"{current_artifact_path}/salary_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {current_artifact_path}/salary_model.pkl")
        
        # Save encoders to the timestamped directory
        for col, encoder in encoders.items():
            with open(f"{current_artifact_path}/{col}_encoder.pkl", "wb") as f:
                pickle.dump(encoder, f)
            logger.info(f"Encoder for {col} saved to {current_artifact_path}/{col}_encoder.pkl")

        # Dummy files for app.py to load - these would ideally be generated by other parts of the pipeline
        # For demonstration, we'll create simple placeholder files if they don't exist
        skills_impact_data = {
            "Python": {"impact": 5, "demand": "High"},
            "SQL": {"impact": 3, "demand": "High"},
            "Cloud Computing": {"impact": 7, "demand": "High"},
            "Machine Learning": {"impact": 8, "demand": "High"},
            "Data Visualization": {"impact": 4, "demand": "Medium"},
            "Project Management": {"impact": 6, "demand": "Medium"},
            "Communication": {"impact": 2, "demand": "Medium"},
            "Leadership": {"impact": 7, "demand": "High"},
            "Negotiation": {"impact": 3, "demand": "Low"}
        }
        with open(f"{current_artifact_path}/skills_impact.json", "w") as f:
            json.dump(skills_impact_data, f, indent=2)
        logger.info(f"skills_impact.json saved to {current_artifact_path}/skills_impact.json")

        # Dummy salary trend model (or a simple DataFrame saved as pickle)
        dummy_trend_data = pd.DataFrame({
            "Year": [2020, 2021, 2022, 2023, 2024],
            "Tech Industry": [8_50_000, 9_20_000, 10_50_000, 11_80_000, 12_50_000],
            "Finance": [9_00_000, 9_30_000, 9_80_000, 10_20_000, 10_60_000],
            "Healthcare": [7_20_000, 7_50_000, 8_00_000, 8_40_000, 8_90_000]
        })
        with open(f"{current_artifact_path}/salary_trend_model.pkl", "wb") as f:
            pickle.dump(dummy_trend_data, f)
        logger.info(f"salary_trend_model.pkl saved to {current_artifact_path}/salary_trend_model.pkl")


        save_metadata(metrics, best_params, X.shape) # Pass X.shape to save_metadata
        
        # Step 7: Final summary
        elapsed = time.time() - start_time
        logger.info(f"✅ Pipeline completed successfully in {elapsed:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"All artifacts saved to: {current_artifact_path}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)