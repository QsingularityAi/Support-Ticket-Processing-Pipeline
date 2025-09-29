import time
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import choose_encoder
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from typing import Callable
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

REQUEST_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress'
)

MODEL_PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total model predictions made',
    ['target', 'model_type']
)

MODEL_PREDICTION_DURATION = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['target', 'model_type']
)

ANOMALY_DETECTION_COUNT = Counter(
    'anomalies_detected_total',
    'Total anomalies detected',
    ['anomaly_type']
)

SOLUTION_RETRIEVAL_COUNT = Counter(
    'solution_retrievals_total',
    'Total solution retrievals',
    ['query_type']
)

# Custom business KPI metrics
TICKET_VOLUME = Counter(
    'ticket_volume_total',
    'Total number of tickets processed'
)

TICKET_CATEGORY_COUNT = Counter(
    'ticket_categories_total',
    'Ticket count by category',
    ['category']
)

TICKET_PRIORITY_COUNT = Counter(
    'ticket_priorities_total',
    'Ticket count by priority',
    ['priority']
)

TICKET_SEVERITY_COUNT = Counter(
    'ticket_severities_total',
    'Ticket count by severity',
    ['severity']
)

CUSTOMER_SATISFACTION_SCORE = Gauge(
    'customer_satisfaction_score',
    'Average customer satisfaction score'
)

RESOLUTION_TIME = Histogram(
    'ticket_resolution_time_hours',
    'Ticket resolution time in hours'
)

def setup_prometheus_metrics(app: FastAPI):
    """
    Set up Prometheus metrics for the FastAPI application
    """
    
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable):
        """
        Middleware to track HTTP request metrics
        """
        REQUEST_IN_PROGRESS.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            return response
        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            raise e
        finally:
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)
            REQUEST_IN_PROGRESS.dec()
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """
        Endpoint to expose Prometheus metrics
        """
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    logger.info("Prometheus metrics setup completed")

def track_model_prediction(target: str, model_type: str, duration: float):
    """
    Track model prediction metrics
    """
    MODEL_PREDICTION_COUNT.labels(target=target, model_type=model_type).inc()
    MODEL_PREDICTION_DURATION.labels(target=target, model_type=model_type).observe(duration)

def track_anomaly_detection(anomaly_type: str):
    """
    Track anomaly detection metrics
    """
    ANOMALY_DETECTION_COUNT.labels(anomaly_type=anomaly_type).inc()

def track_solution_retrieval(query_type: str):
    """
    Track solution retrieval metrics
    """
    SOLUTION_RETRIEVAL_COUNT.labels(query_type=query_type).inc()

def track_business_kpi(ticket_data: dict):
    """
    Track business KPI metrics
    """
    TICKET_VOLUME.inc()
    
    # Track category
    category = ticket_data.get('category', 'unknown')
    TICKET_CATEGORY_COUNT.labels(category=category).inc()
    
    # Track priority
    priority = ticket_data.get('priority', 'unknown')
    TICKET_PRIORITY_COUNT.labels(priority=priority).inc()
    
    # Track severity
    severity = ticket_data.get('severity', 'unknown')
    TICKET_SEVERITY_COUNT.labels(severity=severity).inc()
    
    # Track resolution time if available
    resolution_time = ticket_data.get('resolution_time_hours', 0)
    if resolution_time > 0:
        RESOLUTION_TIME.observe(resolution_time)

def update_customer_satisfaction_score(score: float):
    """
    Update customer satisfaction score gauge
    """
    CUSTOMER_SATISFACTION_SCORE.set(score)

# Example usage functions
def get_current_metrics():
    """
    Get current metrics values for logging or monitoring
    """
    metrics_data = {
        'request_count': REQUEST_COUNT.describe(),
        'request_duration': REQUEST_DURATION.describe(),
        'requests_in_progress': REQUEST_IN_PROGRESS.describe(),
        'model_predictions': MODEL_PREDICTION_COUNT.describe(),
        'anomalies_detected': ANOMALY_DETECTION_COUNT.describe(),
        'solution_retrievals': SOLUTION_RETRIEVAL_COUNT.describe(),
        'ticket_volume': TICKET_VOLUME.describe(),
        'ticket_categories': TICKET_CATEGORY_COUNT.describe(),
        'ticket_priorities': TICKET_PRIORITY_COUNT.describe(),
        'ticket_severities': TICKET_SEVERITY_COUNT.describe(),
        'customer_satisfaction': CUSTOMER_SATISFACTION_SCORE.describe(),
        'resolution_times': RESOLUTION_TIME.describe()
    }
    return metrics_data
