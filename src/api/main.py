import time
import mlflow
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
import json

# Import our custom modules
from src.feature_store.engineering import FeatureEngineeringPipeline
from src.models.selection import ModelSelector
from src.api.retrieval_optimized import OptimizedSolutionRetriever
from src.api.anomaly_detection import AnomalyDetector
from src.monitoring.prometheus_metrics import setup_prometheus_metrics, track_model_prediction, track_business_kpi, update_customer_satisfaction_score, track_anomaly_detection, track_solution_retrieval
from src.monitoring.logging.logging_config import setup_logging, log_api_call, log_model_prediction, log_solution_retrieval, log_anomaly_detection, log_business_kpi

# Set up comprehensive logging
logger = setup_logging(log_level="INFO", log_file="logs/api.log")

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Product Support System",
    description="AI-powered support ticket classification and solution retrieval system",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "ticket-processing",
            "description": "Endpoints for processing support tickets and making predictions"
        },
        {
            "name": "solution-retrieval",
            "description": "Endpoints for retrieving relevant solutions for support tickets"
        },
        {
            "name": "monitoring",
            "description": "Endpoints for system monitoring and health checks"
        },
        {
            "name": "anomaly-detection",
            "description": "Endpoints for detecting anomalies in support tickets"
        }
    ]
)

# Set up Prometheus metrics before startup
setup_prometheus_metrics(app)

# Add middleware for logging API calls
@app.middleware("http")
async def api_call_logger(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Log the API call
    client_ip = request.client.host if request.client else None
    log_api_call(
        logger,
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=duration,
        client_ip=client_ip
    )
    
    return response

# Global variables for models and feature pipeline
models = {}
label_encoders = {}
feature_pipeline = None
model_selector = None
anomaly_detector = None

class TicketInput(BaseModel):
    """
    Input schema for support ticket processing
    """
    ticket_id: Optional[str] = None
    created_at: Optional[str] = None
    customer_id: Optional[str] = None
    customer_tier: Optional[str] = None
    organization_id: Optional[str] = None
    product: Optional[str] = None
    product_version: Optional[str] = None
    product_module: Optional[str] = None
    channel: Optional[str] = None
    subject: str
    description: str
    error_logs: Optional[str] = None
    stack_trace: Optional[str] = None
    previous_tickets: Optional[int] = 0
    agent_experience_months: Optional[int] = 0
    agent_specialization: Optional[str] = None
    environment: Optional[str] = None
    account_age_days: Optional[int] = 0
    account_monthly_value: Optional[float] = 0.0
    similar_issues_last_30_days: Optional[int] = 0
    product_version_age_days: Optional[int] = 0
    affected_users: Optional[int] = 0
    response_count: Optional[int] = 0
    attachments_count: Optional[int] = 0
    business_impact: Optional[str] = None
    language: Optional[str] = "en"
    region: Optional[str] = "NA"

class TicketPrediction(BaseModel):
    """
    Output schema for ticket predictions
    """
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    severity: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None

class SolutionRetrievalInput(BaseModel):
    """
    Input schema for solution retrieval
    """
    query: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    product: Optional[str] = None
    tags: Optional[List[str]] = None

class Solution(BaseModel):
    """
    Schema for retrieved solutions with optimized RAG scores
    """
    solution_id: str
    title: str
    content: str
    relevance_score: float
    success_rate: float
    category: Optional[str] = None
    subcategory: Optional[str] = None
    product: Optional[str] = None
    keyword_score: Optional[float] = None
    entity_score: Optional[float] = None
    category_score: Optional[float] = None

# Load models and feature pipeline on startup
@app.on_event("startup")
async def load_models():
    global models, label_encoders, feature_pipeline, model_selector, anomaly_detector
    
    logger.info("Loading models and feature pipeline")
    
    # Initialize feature pipeline
    feature_pipeline = FeatureEngineeringPipeline()
    
    # Load model selector
    model_selector = ModelSelector()
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_baseline_statistics()
    
    # Prometheus metrics already set up during app initialization
    
    # Load selection report
    try:
        with open("results/model_selection_report.json", "r") as f:
            selection_report = json.load(f)
        
        # Load selected models from MLflow
        for target, info in selection_report["selected_models"].items():
            if info["status"] == "selected":
                # Load model from MLflow
                model_uri = f"models:/support-ticket-{target}/latest"
                try:
                    models[target] = mlflow.xgboost.load_model(model_uri)
                    logger.info(f"Loaded model for {target} from MLflow")
                except Exception as e:
                    logger.warning(f"Could not load model for {target} from MLflow: {e}")
                    # Fallback to local model
                    model_path = f"artifacts/models/xgboost_{target}/model.pkl"
                    if os.path.exists(model_path):
                        models[target] = joblib.load(model_path)
                        logger.info(f"Loaded model for {target} from local file")
                
                # Load label encoder
                encoder_path = f"artifacts/models/xgboost_{target}/label_encoder.pkl"
                if os.path.exists(encoder_path):
                    label_encoders[target] = joblib.load(encoder_path)
                    logger.info(f"Loaded label encoder for {target}")
    except FileNotFoundError:
        logger.warning("Model selection report not found. Models will be loaded during first prediction.")

@app.get("/", tags=["monitoring"])
async def root():
    """
    Root endpoint - Welcome message for the API
    
    Returns:
        dict: A welcome message
        
    Example:
        Response:
        {
            "message": "Intelligent Product Support System API"
        }
    """
    return {"message": "Intelligent Product Support System API"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=TicketPrediction)
async def predict_ticket_category(ticket: TicketInput):
    """
    Predict category, subcategory, priority, and severity for a support ticket
    """
    global models, label_encoders, feature_pipeline
    start_time = time.time()
    
    try:
        # Convert ticket input to DataFrame
        ticket_dict = ticket.model_dump()
        
        # Add missing fields with default values to match training data structure
        default_fields = {
            'feedback_text': '',
            'product_version': 'unknown',
            'product_module': 'unknown', 
            'channel': 'email',
            'agent_specialization': 'unknown',
            'environment': 'production',
            'business_impact': 'medium',
            'language': 'en',
            'region': 'NA',
            'previous_tickets': 0,
            'agent_experience_months': 12,
            'account_age_days': 365,
            'account_monthly_value': 1000.0,
            'similar_issues_last_30_days': 0,
            'product_version_age_days': 100,
            'affected_users': 1,
            'response_count': 0,
            'attachments_count': 0,
            'ticket_text_length': len(ticket_dict.get('subject', '') + ticket_dict.get('description', ''))
        }
        
        # Add missing fields
        for field, default_value in default_fields.items():
            if field not in ticket_dict:
                ticket_dict[field] = default_value
        
        ticket_df = pd.DataFrame([ticket_dict])
        
        # Process features using pre-trained pipeline
        if feature_pipeline is None:
            feature_pipeline = FeatureEngineeringPipeline()
        
        # Always load transformers for inference to ensure we have the right ones
        feature_pipeline.load_fitted_transformers()
        
        # For inference, we need to transform using pre-fitted transformers
        feature_results = feature_pipeline.transform_for_inference(
            ticket_df, ['category', 'subcategory', 'priority', 'severity']
        )
        
        # Make predictions for each target
        predictions = {}
        confidence_scores = {}
        
        for target in ['category', 'subcategory', 'priority', 'severity']:
            if target in models:
                # Get features for this target
                X = feature_results[target]['features']
                
                # Make prediction
                y_pred = models[target].predict(X)
                y_proba = models[target].predict_proba(X)
                
                # Get confidence score (max probability)
                confidence = float(np.max(y_proba))
                
                # Track model prediction metrics
                track_model_prediction(target, "XGBoost", time.time() - start_time)
                
                # Log model prediction
                log_model_prediction(
                    logger,
                    target=target,
                    model_type="XGBoost",
                    confidence=confidence,
                    duration=time.time() - start_time,
                    ticket_id=ticket.ticket_id
                )
                
                # Decode prediction
                if target in label_encoders:
                    pred_label = label_encoders[target].inverse_transform(y_pred)[0]
                    predictions[target] = pred_label
                    confidence_scores[target] = confidence
                else:
                    predictions[target] = int(y_pred[0])
                    confidence_scores[target] = confidence
        
        # Track business KPIs
        track_business_kpi(ticket_dict)
        
        return TicketPrediction(
            category=predictions.get('category'),
            subcategory=predictions.get('subcategory'),
            priority=predictions.get('priority'),
            severity=predictions.get('severity'),
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error in predict_ticket_category: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrieve_solutions", response_model=List[Solution])
async def retrieve_solutions(input_data: SolutionRetrievalInput):
    """
    Retrieve relevant solutions based on query and context
    """
    global solution_retriever
    start_time = time.time()
    
    # Initialize optimized solution retriever if not already done
    if 'solution_retriever' not in globals():
        solution_retriever = OptimizedSolutionRetriever(use_gpu=True)
        solution_retriever.load_or_create_index()
    
    try:
        # Perform optimized search
        results = solution_retriever.search_solutions_fast(
            query=input_data.query,
            category=input_data.category,
            subcategory=input_data.subcategory,
            product=input_data.product,
            limit=10
        )
        
        # Track solution retrieval metrics
        query_type = "categorized" if input_data.category else "general"
        track_solution_retrieval(query_type)
        
        # Convert to Solution models with optimized scores
        solutions = []
        for result in results:
            solution = Solution(
                solution_id=result['solution_id'],
                title=result['title'],
                content=result['content'],
                relevance_score=result['relevance_score'],
                success_rate=result['success_rate'],
                category=result.get('category'),
                subcategory=result.get('subcategory'),
                product=result.get('product'),
                keyword_score=result.get('keyword_score'),
                entity_score=result.get('entity_score'),
                category_score=result.get('category_score')
            )
            solutions.append(solution)
        
        # Log solution retrieval
        log_solution_retrieval(
            logger,
            query=input_data.query,
            results_count=len(solutions),
            duration=time.time() - start_time,
            ticket_id=None  # We don't have ticket_id in this endpoint
        )
        
        return solutions
        
    except Exception as e:
        logger.error(f"Error in retrieve_solutions: {e}")
        raise HTTPException(status_code=500, detail=f"Solution retrieval error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """
    Get information about loaded models
    """
    model_info = {}
    for target, model in models.items():
        model_info[target] = {
            "type": type(model).__name__,
            "features_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
        }
    
    return {"loaded_models": model_info}

class Anomaly(BaseModel):
    """
    Schema for detected anomalies
    """
    type: str
    description: str
    anomaly_score: float

@app.post("/detect_anomalies", response_model=List[Anomaly])
async def detect_anomalies(tickets: List[TicketInput]):
    """
    Detect anomalies in a batch of tickets
    """
    global anomaly_detector
    
    if anomaly_detector is None:
        anomaly_detector = AnomalyDetector()
        anomaly_detector.load_baseline_statistics()
    
    try:
        # Update current statistics with tickets
        for ticket in tickets:
            ticket_dict = ticket.model_dump()
            anomaly_detector.update_current_stats(ticket_dict)
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies()
        
        # Track anomaly detection metrics
        for anomaly in anomalies:
            track_anomaly_detection(anomaly['type'])
        
        # Convert to Anomaly models
        anomaly_models = []
        for anomaly in anomalies:
            anomaly_model = Anomaly(
                type=anomaly['type'],
                description=anomaly['description'],
                anomaly_score=anomaly['anomaly_score']
            )
            anomaly_models.append(anomaly_model)
            
            # Log anomaly detection
            log_anomaly_detection(
                logger,
                anomaly_type=anomaly['type'],
                anomaly_score=anomaly['anomaly_score'],
                ticket_id=None  # We don't have specific ticket_id for each anomaly
            )
        
        # Reset current stats for next batch
        anomaly_detector.reset_current_stats()
        
        return anomaly_models
        
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
