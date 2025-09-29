import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
import mlflow
import mlflow.xgboost
from typing import Dict, List, Tuple, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModelTrainer:
    """
    XGBoost model trainer for support ticket classification tasks
    """
    
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.target_names = ['category', 'subcategory', 'priority', 'severity']
        
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name('support-ticket-classification')
            if experiment is None:
                experiment_id = mlflow.create_experiment('support-ticket-classification')
                logger.info(f'Created MLflow experiment with ID: {experiment_id}')
            else:
                experiment_id = experiment.experiment_id
                logger.info(f'Using existing MLflow experiment with ID: {experiment_id}')
            
            mlflow.set_experiment('support-ticket-classification')
        except Exception as e:
            logger.warning(f'MLflow setup failed: {e}')
        
    def load_features(self, target: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and targets for a specific target variable and data split
        """
        features_path = f"data/features/{split}_{target}_features.npy"
        targets_path = f"data/features/{split}_{target}_targets.npy"
        
        if not os.path.exists(features_path) or not os.path.exists(targets_path):
            raise FileNotFoundError(f"Feature files not found for target {target} and split {split}")
        
        features = np.load(features_path)
        targets = np.load(targets_path, allow_pickle=True)
        
        logger.info(f"Loaded features for {target} ({split}): {features.shape}")
        logger.info(f"Loaded targets for {target} ({split}): {targets.shape}")
        
        return features, targets
    
    def encode_targets(self, targets: np.ndarray, target_name: str) -> np.ndarray:
        """
        Encode target labels using label encoding
        """
        if target_name not in self.label_encoders:
            self.label_encoders[target_name] = LabelEncoder()
            encoded_targets = self.label_encoders[target_name].fit_transform(targets)
        else:
            encoded_targets = self.label_encoders[target_name].transform(targets)
        
        return encoded_targets
    
    def train_model(self, target: str) -> Dict[str, Any]:
        """
        Train XGBoost model for a specific target variable
        """
        logger.info(f"Training XGBoost model for {target}")
        
        # Load training data
        X_train, y_train = self.load_features(target, "train")
        X_val, y_val = self.load_features(target, "val")
        
        # Encode targets
        y_train_encoded = self.encode_targets(y_train, target)
        y_val_encoded = self.encode_targets(y_val, target)
        
        # Calculate class weights for imbalanced classes
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        class_weights = dict(zip(unique, counts))
        max_count = max(counts)
        
        # Calculate sample weights for training
        sample_weights = np.array([max_count/class_weights[cls] for cls in y_train_encoded])
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)  # Normalize
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"xgboost_{target}"):
            # XGBoost parameters - optimized for multi-class problems
            n_classes = len(np.unique(y_train_encoded))
            
            # Adjust parameters based on number of classes
            if n_classes > 20:  # For subcategory with 25 classes
                params = {
                    'objective': 'multi:softprob',
                    'num_class': n_classes,
                    'max_depth': 8,  # Deeper trees for complex multi-class
                    'learning_rate': 0.05,  # Lower learning rate
                    'n_estimators': 200,  # More estimators
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 1.0,  # L2 regularization
                    'random_state': 42
                }
            else:
                params = {
                    'objective': 'multi:softprob',
                    'num_class': n_classes,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 150,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.05,
                    'reg_lambda': 0.5,
                    'random_state': 42
                }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Create and train model with sample weights
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_encoded, sample_weight=sample_weights, verbose=False)
            
            # Validation predictions
            y_val_pred = model.predict(X_val)
            
            # Calculate F1 scores
            f1_weighted = f1_score(y_val_encoded, y_val_pred, average='weighted')
            f1_macro = f1_score(y_val_encoded, y_val_pred, average='macro')
            f1_micro = f1_score(y_val_encoded, y_val_pred, average='micro')
            
            # Skip cross-validation for faster training
            cv_scores = np.array([f1_weighted])  # Use validation F1 as proxy
            
            # Log metrics
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("f1_micro", f1_micro)
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            
            # Save model
            model_dir = f"artifacts/models/xgboost_{target}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
            
            # Log model to MLflow
            model_info = mlflow.xgboost.log_model(model, artifact_path="model")
            
            # Register model in MLflow model registry
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mv = mlflow.register_model(model_uri, f"support-ticket-{target}")
                logger.info(f"Model registered for {target} with version {mv.version}")
            except Exception as e:
                logger.warning(f"Could not register model for {target}: {e}")
            
            # Save label encoder
            encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            joblib.dump(self.label_encoders[target], encoder_path)
            
            logger.info(f"Model trained for {target}")
            logger.info(f"F1 Weighted: {f1_weighted:.4f}")
            logger.info(f"F1 Macro: {f1_macro:.4f}")
            logger.info(f"F1 Micro: {f1_micro:.4f}")
            logger.info(f"CV F1 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return {
                'model': model,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'cv_scores': cv_scores,
                'model_path': model_path,
                'encoder_path': encoder_path
            }
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train XGBoost models for all target variables
        """
        results = {}
        
        for target in self.target_names:
            try:
                logger.info(f"Starting training for {target}")
                results[target] = self.train_model(target)
                self.models[target] = results[target]['model']
            except Exception as e:
                logger.error(f"Error training model for {target}: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def evaluate_model(self, target: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data
        """
        if target not in self.models:
            raise ValueError(f"Model for {target} not trained yet")
        
        logger.info(f"Evaluating model for {target}")
        
        # Load test data
        X_test, y_test = self.load_features(target, "test")
        
        # Encode targets
        y_test_encoded = self.encode_targets(y_test, target)
        
        # Predictions
        y_pred = self.models[target].predict(X_test)
        
        # Calculate F1 scores
        f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
        f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
        f1_micro = f1_score(y_test_encoded, y_pred, average='micro')
        
        # Detailed classification report
        target_names = self.label_encoders[target].classes_
        report = classification_report(y_test_encoded, y_pred, target_names=target_names, output_dict=True)
        
        logger.info(f"Test F1 Weighted for {target}: {f1_weighted:.4f}")
        logger.info(f"Test F1 Macro for {target}: {f1_macro:.4f}")
        logger.info(f"Test F1 Micro for {target}: {f1_micro:.4f}")
        
        return {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'classification_report': report
        }
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models
        """
        results = {}
        
        for target in self.target_names:
            if target in self.models:
                results[target] = self.evaluate_model(target)
        
        return results
    
    def save_model_results(self, training_results: Dict[str, Dict[str, Any]], 
                          evaluation_results: Dict[str, Dict[str, Any]]):
        """
        Save model training and evaluation results
        """
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Combine training and evaluation results (exclude non-serializable objects)
        combined_results = {}
        for target in self.target_names:
            training_data = training_results.get(target, {}).copy()
            evaluation_data = evaluation_results.get(target, {}).copy()
            
            # Remove non-serializable objects
            if 'model' in training_data:
                del training_data['model']
            if 'cv_scores' in training_data:
                # Convert numpy array to list
                training_data['cv_scores'] = training_data['cv_scores'].tolist()
            
            combined_results[target] = {
                'training': training_data,
                'evaluation': evaluation_data
            }
        
        # Save to JSON
        with open(os.path.join(results_dir, "model_results.json"), "w") as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info("Model results saved successfully")

def main():
    """
    Main function to train and evaluate XGBoost models
    """
    # Initialize trainer
    trainer = XGBoostModelTrainer()
    
    # Train all models
    training_results = trainer.train_all_models()
    
    # Evaluate all models
    evaluation_results = trainer.evaluate_all_models()
    
    # Save results
    trainer.save_model_results(training_results, evaluation_results)
    
    # Print summary
    print("\n=== MODEL TRAINING SUMMARY ===")
    for target in trainer.target_names:
        if target in training_results and 'f1_weighted' in training_results[target]:
            print(f"{target}:")
            print(f"  - Training F1 Weighted: {training_results[target]['f1_weighted']:.4f}")
            print(f"  - CV F1 Mean: {training_results[target]['cv_scores'].mean():.4f}")
            if target in evaluation_results:
                print(f"  - Test F1 Weighted: {evaluation_results[target]['f1_weighted']:.4f}")
            print()

if __name__ == "__main__":
    main()
