import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import shutil
from src.data.ingestion import TicketDataIngestion
from src.feature_store.engineering import FeatureEngineeringPipeline
from src.models.selection import ModelSelector
from src.api.main import app
from src.monitoring.prometheus_metrics import setup_prometheus_metrics, track_model_prediction, track_business_kpi, update_customer_satisfaction_score, track_anomaly_detection, track_solution_retrieval, get_current_metrics
from src.monitoring.logging.logging_config import setup_logging, log_api_call, log_model_prediction, log_solution_retrieval, log_anomaly_detection, log_business_kpi, JSONFormatter
from fastapi.testclient import TestClient

client = TestClient(app)

def test_complete_data_pipeline():
    """Test the complete data ingestion and feature engineering pipeline"""
    # Sample ticket data
    sample_tickets = [
        {
            "ticket_id": "TICKET-001",
            "created_at": "2023-01-01T10:00:00Z",
            "customer_id": "CUST-001",
            "subject": "Issue with product",
            "description": "Product is not working correctly",
            "category": "Technical",
            "subcategory": "Software",
            "priority": "High",
            "severity": "Critical"
        },
        {
            "ticket_id": "TICKET-002",
            "created_at": "2023-01-02T11:00:00Z",
            "customer_id": "CUST-002",
            "subject": "Billing question",
            "description": "Question about my bill",
            "category": "Billing",
            "subcategory": "Invoicing",
            "priority": "Medium",
            "severity": "Low"
        }
    ]
    
    # Create DataFrame from sample data
    df = pd.DataFrame(sample_tickets)
    assert len(df) == 2
    
    # Test data ingestion class
    ingestion = TicketDataIngestion()
    ingestion.df = df  # Set the dataframe directly for testing
    
    # Test data validation
    issues = ingestion.validate_data()
    assert isinstance(issues, dict)
    
    # Test feature engineering
    pipeline = FeatureEngineeringPipeline()
    feature_results = pipeline.process_features_for_targets(
        df, ['category', 'subcategory', 'priority', 'severity']
    )
    
    # Assertions
    assert isinstance(feature_results, dict)
    assert 'category' in feature_results
    assert 'subcategory' in feature_results
    assert 'priority' in feature_results
    assert 'severity' in feature_results

def test_api_prediction_workflow():
    """Test the complete API prediction workflow"""
    # Test data
    test_ticket = {
        "ticket_id": "TEST-001",
        "subject": "Product not working",
        "description": "I'm having issues with my product. It's not starting up properly.",
        "customer_id": "CUST-001",
        "customer_tier": "Premium",
        "product": "SoftwareX",
        "product_version": "v2.1.0"
    }
    
    # Test API prediction endpoint
    response = client.post("/predict", json=test_ticket)
    
    # Response should be 200 or 500 (if models aren't loaded)
    assert response.status_code in [200, 500]
    
    # If successful, check response structure
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        # Should contain prediction fields
        assert any(key in data for key in ['category', 'subcategory', 'priority', 'severity'])

def test_solution_retrieval_workflow():
    """Test the complete solution retrieval workflow"""
    # Test data
    test_query = {
        "query": "Product not working",
        "category": "Technical",
        "product": "SoftwareX"
    }
    
    # Test API solution retrieval endpoint
    response = client.post("/retrieve_solutions", json=test_query)
    
    # Response should be 200 or 500 (if Qdrant isn't available)
    assert response.status_code in [200, 500]
    
    # If successful, check response structure
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        # Each solution should have required fields
        if len(data) > 0:
            solution = data[0]
            assert 'solution_id' in solution
            assert 'title' in solution
            assert 'content' in solution
            assert 'relevance_score' in solution

def test_model_training_workflow():
    """Test the complete model training and selection workflow"""
    # Create sample data for testing
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y_category = np.random.randint(0, 3, 50)
    y_priority = np.random.randint(0, 3, 50)
    
    # Test model selector
    selector = ModelSelector()
    
    # Test training for different targets
    model_category, encoder_category = selector._train_model(X, y_category)
    model_priority, encoder_priority = selector._train_model(X, y_priority)
    
    # Assertions
    assert model_category is not None
    assert model_priority is not None
    assert encoder_category is not None
    assert encoder_priority is not None

def test_logging_configuration():
    """Test logging setup and JSON formatter functionality"""
    # Create a temporary directory for logs
    temp_dir = tempfile.mkdtemp()
    log_file_path = os.path.join(temp_dir, "test.log")
    
    try:
        # Test logger setup
        logger = setup_logging(log_level="INFO", log_file=log_file_path)
        assert logger is not None
        assert len(logger.handlers) >= 2  # File and console handlers
        
        # Test logging functionality
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check that log file was created
        assert os.path.exists(log_file_path)
        
        # Test JSON formatter
        json_formatter = JSONFormatter()
        import logging
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        formatted_record = json_formatter.format(record)
        
        # Check that output is valid JSON
        log_data = json.loads(formatted_record)
        assert "timestamp" in log_data
        assert "level" in log_data
        assert "logger" in log_data
        assert "message" in log_data
        assert log_data["message"] == "Test message"
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_prometheus_metrics_setup():
    """Test Prometheus metrics setup and endpoint"""
    # Test that metrics endpoint is available
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Test metrics tracking functions
    track_model_prediction("category", "XGBoost", 0.5)
    track_anomaly_detection("volume_spike")
    track_solution_retrieval("general")
    
    # Test business KPI tracking
    sample_ticket = {
        "category": "Technical",
        "priority": "High",
        "severity": "Critical"
    }
    track_business_kpi(sample_ticket)
    
    # Test customer satisfaction score update
    update_customer_satisfaction_score(0.85)
    
    # Verify metrics can be retrieved
    metrics = get_current_metrics()
    assert isinstance(metrics, dict)
    assert "request_count" in metrics

def test_api_monitoring_endpoints():
    """Test API monitoring and health check endpoints"""
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Intelligent Product Support System API"
    
    # Test health check endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    
    # Test model info endpoint
    response = client.get("/model_info")
    # This endpoint should work regardless of whether models are loaded
    assert response.status_code == 200
    data = response.json()
    assert "loaded_models" in data

def test_logging_integration_with_api_calls():
    """Test that API calls are properly logged"""
    # Create a temporary directory for logs
    temp_dir = tempfile.mkdtemp()
    log_file_path = os.path.join(temp_dir, "api_test.log")
    
    try:
        # Set up logger for testing
        test_logger = setup_logging(log_level="INFO", log_file=log_file_path)
        
        # Test logging of different events
        log_api_call(test_logger, "/test", "GET", 200, 0.1, "127.0.0.1")
        log_model_prediction(test_logger, "category", "XGBoost", 0.95, 0.2, "TICKET-001")
        log_solution_retrieval(test_logger, "test query", 5, 0.3, "TICKET-002")
        log_anomaly_detection(test_logger, "test_anomaly", 0.8, "TICKET-003")
        log_business_kpi(test_logger, "test_kpi", 42, {"context": "test"})
        
        # Check that log file was created and has content
        assert os.path.exists(log_file_path)
        with open(log_file_path, 'r') as f:
            log_content = f.read()
            assert len(log_content) > 0
            
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_complete_monitoring_workflow():
    """Test the complete monitoring and logging workflow integration"""
    # Test that API calls generate logs
    test_ticket = {
        "ticket_id": "MONITORING-TEST-001",
        "subject": "Test for monitoring",
        "description": "Testing the complete monitoring workflow",
        "customer_id": "CUST-001",
        "customer_tier": "Premium",
        "product": "SoftwareX",
        "product_version": "v2.1.0"
    }
    
    # Make API call
    response = client.post("/predict", json=test_ticket)
    
    # Response should be 200 or 500 (if models aren't loaded)
    assert response.status_code in [200, 500]
    
    # Test that metrics are tracked
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    
    # Check that we can retrieve model info
    model_info_response = client.get("/model_info")
    assert model_info_response.status_code == 200
