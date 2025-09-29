"""
Logging configuration for the support ticket system
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def log_api_call(logger: logging.Logger, endpoint: str, method: str, 
                status_code: int, duration: float, client_ip: Optional[str] = None):
    """
    Log API call information
    """
    logger.info(f"API Call - {method} {endpoint} - Status: {status_code} - Duration: {duration:.3f}s - IP: {client_ip}")

def log_model_prediction(logger: logging.Logger, target: str, model_type: str, 
                        confidence: float, duration: float, ticket_id: Optional[str] = None):
    """
    Log model prediction information
    """
    logger.info(f"Model Prediction - Target: {target} - Model: {model_type} - Confidence: {confidence:.3f} - Duration: {duration:.3f}s - Ticket: {ticket_id}")

def log_solution_retrieval(logger: logging.Logger, query: str, results_count: int, 
                          duration: float, ticket_id: Optional[str] = None):
    """
    Log solution retrieval information
    """
    logger.info(f"Solution Retrieval - Query: '{query[:50]}...' - Results: {results_count} - Duration: {duration:.3f}s - Ticket: {ticket_id}")

def log_anomaly_detection(logger: logging.Logger, anomaly_type: str, 
                         anomaly_score: float, ticket_id: Optional[str] = None):
    """
    Log anomaly detection information
    """
    logger.info(f"Anomaly Detection - Type: {anomaly_type} - Score: {anomaly_score:.3f} - Ticket: {ticket_id}")

def log_business_kpi(logger: logging.Logger, kpi_name: str, value: float, 
                    timestamp: Optional[datetime] = None):
    """
    Log business KPI information
    """
    if timestamp is None:
        timestamp = datetime.now()
    logger.info(f"Business KPI - {kpi_name}: {value} - Time: {timestamp}")