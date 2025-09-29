#!/usr/bin/env python3
"""
Script to start MLflow tracking server
"""
import os
import sys
import logging
import time
import subprocess
from mlflow_config import start_mlflow_server, setup_mlflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to start MLflow tracking server
    """
    # Create directories if they don't exist
    os.makedirs("mlflow_artifacts", exist_ok=True)
    
    # Setup MLflow configuration
    experiment_id = setup_mlflow()
    logger.info(f"MLflow experiment ID: {experiment_id}")
    
    # Start MLflow server
    logger.info("Starting MLflow tracking server...")
    process = start_mlflow_server()
    
    # Save process information
    with open("mlflow.pid", "w") as f:
        f.write(str(process.pid))
    
    logger.info(f"MLflow server started with PID: {process.pid}")
    logger.info("Server is running in the background")
    logger.info("Access the UI at: http://localhost:5000")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping MLflow server...")
        process.terminate()
        process.wait()
        # Remove PID file
        if os.path.exists("mlflow.pid"):
            os.remove("mlflow.pid")
        logger.info("MLflow server stopped")

if __name__ == "__main__":
    main()
