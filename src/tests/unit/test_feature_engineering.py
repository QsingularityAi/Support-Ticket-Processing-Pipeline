import pytest
import pandas as pd
import numpy as np
from src.feature_store.engineering import FeatureEngineeringPipeline

def test_feature_pipeline_initialization():
    """Test that FeatureEngineeringPipeline initializes correctly"""
    pipeline = FeatureEngineeringPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'tfidf_vectorizers')
    assert hasattr(pipeline, 'label_encoders')

def test_text_feature_extraction():
    """Test TF-IDF feature extraction for text fields"""
    pipeline = FeatureEngineeringPipeline()
    
    # Sample data
    df = pd.DataFrame({
        'subject': ['Product issue', 'Feature request', 'Bug report'],
        'description': [
            'Having problems with the product functionality',
            'Would like to request a new feature',
            'Found a bug in the software'
        ]
    })
    
    # Process text features
    subject_features = pipeline._process_text_features(df, 'subject')
    description_features = pipeline._process_text_features(df, 'description')
    
    # Assertions
    assert isinstance(subject_features, np.ndarray)
    assert isinstance(description_features, np.ndarray)
    assert subject_features.shape[0] == 3
    assert description_features.shape[0] == 3

def test_categorical_encoding():
    """Test categorical feature encoding"""
    pipeline = FeatureEngineeringPipeline()
    
    # Sample data
    df = pd.DataFrame({
        'category': ['Technical', 'Billing', 'Technical'],
        'priority': ['High', 'Low', 'Medium']
    })
    
    # Process categorical features
    category_features = pipeline._process_categorical_features(df, 'category')
    priority_features = pipeline._process_categorical_features(df, 'priority')
    
    # Assertions
    assert isinstance(category_features, np.ndarray)
    assert isinstance(priority_features, np.ndarray)
    assert category_features.shape[0] == 3
    assert priority_features.shape[0] == 3

def test_numerical_processing():
    """Test numerical feature processing"""
    pipeline = FeatureEngineeringPipeline()
    
    # Sample data
    df = pd.DataFrame({
        'previous_tickets': [5, 10, 2],
        'account_age_days': [365, 180, 730]
    })
    
    # Process numerical features
    numerical_features = pipeline._process_numerical_features(
        df, ['previous_tickets', 'account_age_days']
    )
    
    # Assertions
    assert isinstance(numerical_features, np.ndarray)
    assert numerical_features.shape[0] == 3
    assert numerical_features.shape[1] == 2
