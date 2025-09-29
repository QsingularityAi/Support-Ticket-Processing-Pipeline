import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Any
import logging
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for support tickets
    """
    
    def __init__(self):
        self.text_vectorizers = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = {}
        
        # Define column types
        self.text_columns = ['subject', 'description', 'error_logs', 'feedback_text']
        self.categorical_columns = [
            'product', 'product_version', 'product_module', 'channel', 
            'customer_tier', 'agent_specialization', 'environment', 
            'business_impact', 'language', 'region'
        ]
        self.numerical_columns = [
            'previous_tickets', 'agent_experience_months', 'account_age_days',
            'account_monthly_value', 'similar_issues_last_30_days', 
            'product_version_age_days', 'affected_users', 'response_count',
            'attachments_count', 'ticket_text_length'
        ]
        
    def extract_text_features(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract TF-IDF features from text columns
        """
        text_features = {}
        
        for col in text_columns:
            if col in df.columns:
                logger.info(f"Extracting TF-IDF features from {col}")
                
                # Fill NaN values with empty string
                text_data = df[col].fillna('').astype(str)
                
                # Initialize or load existing vectorizer
                if col not in self.text_vectorizers:
                    # Adaptive parameters based on dataset size
                    dataset_size = len(text_data)
                    min_df = 1 if dataset_size < 100 else 2
                    max_features = min(500, dataset_size * 10) if dataset_size < 100 else 1000
                    
                    # For single documents, use very permissive settings
                    if dataset_size == 1:
                        min_df = 1
                        max_df = 1.0
                    else:
                        max_df = 0.95
                    
                    self.text_vectorizers[col] = TfidfVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=min_df,
                        max_df=max_df
                    )
                
                # Fit and transform or just transform
                if col not in self.feature_names:
                    # First time fitting
                    tfidf_matrix = self.text_vectorizers[col].fit_transform(text_data)
                    self.feature_names[col] = self.text_vectorizers[col].get_feature_names_out()
                else:
                    # Already fitted, just transform
                    tfidf_matrix = self.text_vectorizers[col].transform(text_data)
                
                text_features[col] = tfidf_matrix.toarray()
                logger.info(f"TF-IDF features extracted for {col}. Shape: {text_features[col].shape}")
        
        return text_features
    
    def load_fitted_transformers(self):
        """
        Load pre-fitted transformers from training
        """
        import joblib
        
        # Try to load saved transformers
        try:
            # Load text vectorizers
            if os.path.exists("artifacts/feature_transformers/text_vectorizers.pkl"):
                self.text_vectorizers = joblib.load("artifacts/feature_transformers/text_vectorizers.pkl")
                logger.info("Loaded pre-fitted text vectorizers")
            
            # Load label encoders (categorical encoders)
            if os.path.exists("artifacts/feature_transformers/label_encoders.pkl"):
                self.label_encoders = joblib.load("artifacts/feature_transformers/label_encoders.pkl")
                logger.info("Loaded pre-fitted label encoders")
            
            # Load numerical scalers
            if os.path.exists("artifacts/feature_transformers/scalers.pkl"):
                self.scalers = joblib.load("artifacts/feature_transformers/scalers.pkl")
                logger.info("Loaded pre-fitted numerical scalers")
                
        except Exception as e:
            logger.warning(f"Could not load pre-fitted transformers: {e}")
    
    def save_fitted_transformers(self):
        """
        Save fitted transformers for inference
        """
        import joblib
        
        os.makedirs("artifacts/feature_transformers", exist_ok=True)
        
        # Save text vectorizers
        if self.text_vectorizers:
            joblib.dump(self.text_vectorizers, "artifacts/feature_transformers/text_vectorizers.pkl")
        
        # Save label encoders (categorical encoders)
        if self.label_encoders:
            joblib.dump(self.label_encoders, "artifacts/feature_transformers/label_encoders.pkl")
        
        # Save numerical scalers
        if self.scalers:
            joblib.dump(self.scalers, "artifacts/feature_transformers/scalers.pkl")
        
        logger.info("Saved fitted transformers for inference")
    
    def transform_for_inference(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Transform data for inference using pre-fitted transformers
        """
        results = {}
        
        logger.info(f"Available text vectorizers: {list(self.text_vectorizers.keys())}")
        logger.info(f"Available label encoders: {list(self.label_encoders.keys())}")
        logger.info(f"Available scalers: {list(self.scalers.keys())}")
        
        for target in targets:
            logger.info(f"Processing features for target: {target}")
            
            # Extract text features using pre-fitted vectorizers
            text_features = {}
            for col in self.text_columns:
                if col in df.columns and col in self.text_vectorizers:
                    text_data = df[col].fillna('').astype(str)
                    tfidf_matrix = self.text_vectorizers[col].transform(text_data)
                    text_features[col] = tfidf_matrix.toarray()
                    logger.info(f"Text feature {col} shape: {text_features[col].shape}")
                elif col in df.columns:
                    logger.warning(f"No pre-trained vectorizer found for {col}")
            
            # Extract categorical features using pre-fitted encoders
            categorical_features = {}
            for col in self.categorical_columns:
                if col in df.columns and col in self.label_encoders:
                    try:
                        encoded = self.label_encoders[col].transform(df[col].fillna('unknown'))
                        categorical_features[col] = encoded.reshape(-1, 1)
                        logger.info(f"Categorical feature {col} shape: {categorical_features[col].shape}")
                    except ValueError:
                        # Handle unseen categories
                        logger.warning(f"Unseen category in {col}, using default encoding")
                        categorical_features[col] = np.zeros((len(df), 1))
                elif col in df.columns:
                    logger.warning(f"No pre-trained encoder found for {col}")
            
            # Extract numerical features using pre-fitted scalers
            numerical_features = {}
            for col in self.numerical_columns:
                if col in df.columns and col in self.scalers:
                    scaled = self.scalers[col].transform(df[[col]].fillna(0))
                    numerical_features[col] = scaled
                    logger.info(f"Numerical feature {col} shape: {numerical_features[col].shape}")
                elif col in df.columns:
                    logger.warning(f"No pre-trained scaler found for {col}")
            
            # Combine all features
            all_features = []
            
            # Add text features
            for col in self.text_columns:
                if col in text_features:
                    all_features.append(text_features[col])
            
            # Add categorical features
            for col in self.categorical_columns:
                if col in categorical_features:
                    all_features.append(categorical_features[col])
            
            # Add numerical features
            for col in self.numerical_columns:
                if col in numerical_features:
                    all_features.append(numerical_features[col])
            
            # Combine into final feature matrix
            if all_features:
                combined_features = np.hstack(all_features)
            else:
                logger.error("No features found! Creating dummy feature matrix")
                combined_features = np.zeros((len(df), 1))
            
            logger.info(f"Combined feature matrix shape: {combined_features.shape}")
            
            results[target] = {
                'features': combined_features,
                'targets': df[target].values if target in df.columns else np.array(['unknown'] * len(df))
            }
        
        return results
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Encode categorical features using label encoding
        """
        categorical_features = {}
        
        for col in categorical_columns:
            if col in df.columns:
                logger.info(f"Encoding categorical feature {col}")
                
                # Fill NaN values
                cat_data = df[col].fillna('unknown').astype(str)
                
                # Initialize or load existing encoder
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Fit and transform or just transform
                if col not in self.feature_names:
                    # First time fitting
                    encoded_data = self.label_encoders[col].fit_transform(cat_data)
                    self.feature_names[col] = self.label_encoders[col].classes_
                else:
                    # Already fitted, just transform
                    encoded_data = self.label_encoders[col].transform(cat_data)
                
                categorical_features[col] = encoded_data.reshape(-1, 1)
                logger.info(f"Categorical features encoded for {col}. Shape: {categorical_features[col].shape}")
        
        return categorical_features
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Scale numerical features using standard scaling
        """
        numerical_features = {}
        
        for col in numerical_columns:
            if col in df.columns:
                logger.info(f"Scaling numerical feature {col}")
                
                # Fill NaN values with 0 (assuming preprocessing already handled this)
                num_data = df[col].fillna(0)
                
                # Initialize or load existing scaler
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                
                # Fit and transform or just transform
                if col not in self.feature_names:
                    # First time fitting
                    scaled_data = self.scalers[col].fit_transform(num_data.values.reshape(-1, 1))
                else:
                    # Already fitted, just transform
                    scaled_data = self.scalers[col].transform(num_data.values.reshape(-1, 1))
                
                numerical_features[col] = scaled_data
                logger.info(f"Numerical features scaled for {col}. Shape: {numerical_features[col].shape}")
        
        return numerical_features
    
    def combine_features(self, text_features: Dict[str, np.ndarray], 
                        categorical_features: Dict[str, np.ndarray], 
                        numerical_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine all features into a single feature matrix
        """
        logger.info("Combining all features into a single matrix")
        
        # Collect all feature arrays
        feature_arrays = []
        
        # Add text features
        for col, features in text_features.items():
            feature_arrays.append(features)
            
        # Add categorical features
        for col, features in categorical_features.items():
            feature_arrays.append(features)
            
        # Add numerical features
        for col, features in numerical_features.items():
            feature_arrays.append(features)
        
        # Combine all features horizontally
        if feature_arrays:
            combined_features = np.hstack(feature_arrays)
            logger.info(f"Combined feature matrix shape: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No features to combine")
            return np.array([])
    
    def process_features_for_targets(self, df: pd.DataFrame, target_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process features for all target variables
        """
        results = {}
        
        # Define feature columns for each target
        text_columns = ['subject', 'description', 'error_logs', 'feedback_text']
        categorical_columns = ['product', 'product_version', 'product_module', 'channel', 'customer_tier', 
                              'agent_specialization', 'environment', 'business_impact', 'language', 'region']
        numerical_columns = ['previous_tickets', 'agent_experience_months', 'account_age_days', 
                            'account_monthly_value', 'similar_issues_last_30_days', 'product_version_age_days',
                            'affected_users', 'response_count', 'attachments_count', 'ticket_text_length']
        
        # Extract text features
        text_features = self.extract_text_features(df, text_columns)
        
        # Extract categorical features
        categorical_features = self.encode_categorical_features(df, categorical_columns)
        
        # Extract numerical features
        numerical_features = self.scale_numerical_features(df, numerical_columns)
        
        # Combine features for each target
        for target in target_names:
            if target in df.columns:
                logger.info(f"Processing features for target: {target}")
                
                # Get target values
                target_values = df[target].fillna('unknown')
                
                # Combine all features
                combined_features = self.combine_features(text_features, categorical_features, numerical_features)
                
                results[target] = {
                    'features': combined_features,
                    'targets': target_values.values,
                    'feature_names': {
                        'text': list(text_features.keys()),
                        'categorical': list(categorical_features.keys()),
                        'numerical': list(numerical_features.keys())
                    }
                }
        
        return results
    
    def save_artifacts(self, save_dir: str = "artifacts/feature_store"):
        """
        Save feature engineering artifacts
        """
        logger.info(f"Saving feature engineering artifacts to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save text vectorizers
        for col, vectorizer in self.text_vectorizers.items():
            joblib.dump(vectorizer, os.path.join(save_dir, f"tfidf_{col}.pkl"))
            
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, os.path.join(save_dir, f"encoder_{col}.pkl"))
            
        # Save scalers
        for col, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(save_dir, f"scaler_{col}.pkl"))
            
        logger.info("Feature engineering artifacts saved successfully")

def main():
    """
    Main function to run the feature engineering pipeline
    """
    # Load data splits
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Define target names
    target_names = ['category', 'subcategory', 'priority', 'severity']
    
    # Process features for all targets
    train_features = pipeline.process_features_for_targets(train_df, target_names)
    val_features = pipeline.process_features_for_targets(val_df, target_names)
    test_features = pipeline.process_features_for_targets(test_df, target_names)
    
    # Save artifacts
    pipeline.save_artifacts()
    
    # Save processed features
    os.makedirs("data/features", exist_ok=True)
    
    for target in target_names:
        if target in train_features:
            # Save train features
            np.save(f"data/features/train_{target}_features.npy", train_features[target]['features'])
            np.save(f"data/features/train_{target}_targets.npy", train_features[target]['targets'])
            
            # Save validation features
            np.save(f"data/features/val_{target}_features.npy", val_features[target]['features'])
            np.save(f"data/features/val_{target}_targets.npy", val_features[target]['targets'])
            
            # Save test features
            np.save(f"data/features/test_{target}_features.npy", test_features[target]['features'])
            np.save(f"data/features/test_{target}_targets.npy", test_features[target]['targets'])
    
    logger.info("Feature engineering pipeline completed successfully")

if __name__ == "__main__":
    main()
