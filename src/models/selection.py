"""
Model selection utilities for the support ticket system
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Model selection and management class
    """
    
    def __init__(self, selection_report_path: str = "results/model_selection_report.json"):
        self.selection_report_path = selection_report_path
        self.selected_models = {}
        self.load_selection_report()
    
    def load_selection_report(self):
        """
        Load model selection report
        """
        try:
            if os.path.exists(self.selection_report_path):
                with open(self.selection_report_path, 'r') as f:
                    report = json.load(f)
                self.selected_models = report.get('selected_models', {})
                logger.info(f"Loaded model selection report with {len(self.selected_models)} models")
            else:
                logger.warning(f"Model selection report not found at {self.selection_report_path}")
        except Exception as e:
            logger.error(f"Error loading model selection report: {e}")
    
    def get_selected_models(self) -> Dict[str, Any]:
        """
        Get dictionary of selected models
        """
        return {
            target: info for target, info in self.selected_models.items()
            if info.get('status') == 'selected'
        }
    
    def is_model_selected(self, target: str) -> bool:
        """
        Check if a model is selected for a given target
        """
        return (target in self.selected_models and 
                self.selected_models[target].get('status') == 'selected')
    
    def get_model_info(self, target: str) -> Optional[Dict[str, Any]]:
        """
        Get model information for a specific target
        """
        return self.selected_models.get(target)