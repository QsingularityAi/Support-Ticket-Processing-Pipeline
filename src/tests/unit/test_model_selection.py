import pytest
import pandas as pd
import numpy as np
from src.models.selection import ModelSelector

def test_model_selector_initialization():
    """Test that ModelSelector initializes correctly"""
    selector = ModelSelector()
    assert selector is not None
    assert hasattr(selector, 'models')
    assert hasattr(selector, 'label_encoders')

def test_model_evaluation():
    """Test model evaluation with F1 scores"""
    selector = ModelSelector()
    
    # Sample data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Evaluate model
    f1_scores = selector._evaluate_model(X, y, cv_folds=3)
    
    # Assertions
    assert isinstance(f1_scores, dict)
    assert 'mean' in f1_scores
    assert 'std' in f1_scores
    assert f1_scores['mean'] >= 0 and f1_scores['mean'] <= 1

def test_model_selection_criteria():
    """Test model selection based on F1 score criteria"""
    selector = ModelSelector()
    
    # Test with high F1 score
    high_f1 = {'mean': 0.90, 'std': 0.05}
    assert selector._meets_selection_criteria(high_f1) == True
    
    # Test with low F1 score
    low_f1 = {'mean': 0.70, 'std': 0.05}
    assert selector._meets_selection_criteria(low_f1) == False
    
    # Test with boundary F1 score
    boundary_f1 = {'mean': 0.85, 'std': 0.05}
    assert selector._meets_selection_criteria(boundary_f1) == True

def test_model_training():
    """Test model training process"""
    selector = ModelSelector()
    
    # Sample data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Train model
    model, label_encoder = selector._train_model(X, y)
    
    # Assertions
    assert model is not None
    assert label_encoder is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
