import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly detection system for monitoring support ticket patterns
    """
    
    def __init__(self, history_window_days: int = 30):
        self.history_window_days = history_window_days
        self.baseline_stats = {}
        self.current_stats = defaultdict(lambda: defaultdict(int))
        self.current_stats['resolution_times'] = []
        self.current_stats['resolution_helpful'] = []
        
    def load_baseline_statistics(self, baseline_data_path: str = "data/splits/train.csv"):
        """
        Load baseline statistics from historical data
        """
        try:
            df = pd.read_csv(baseline_data_path)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Calculate baseline statistics
            cutoff_date = df['created_at'].max() - timedelta(days=self.history_window_days)
            baseline_df = df[df['created_at'] >= cutoff_date]
            
            self.baseline_stats = {
                'category_distribution': baseline_df['category'].value_counts().to_dict(),
                'priority_distribution': baseline_df['priority'].value_counts().to_dict(),
                'severity_distribution': baseline_df['severity'].value_counts().to_dict(),
                'sentiment_distribution': baseline_df['customer_sentiment'].value_counts().to_dict(),
                'product_distribution': baseline_df['product'].value_counts().to_dict(),
                'total_tickets': len(baseline_df),
                'avg_resolution_time': baseline_df['resolution_time_hours'].mean(),
                'success_rate': baseline_df['resolution_helpful'].mean()
            }
            
            logger.info(f"Baseline statistics loaded from {len(baseline_df)} tickets")
            return self.baseline_stats
            
        except FileNotFoundError:
            logger.warning("Baseline data not found. Initializing with empty statistics.")
            self.baseline_stats = {
                'category_distribution': {},
                'priority_distribution': {},
                'severity_distribution': {},
                'sentiment_distribution': {},
                'product_distribution': {},
                'total_tickets': 0,
                'avg_resolution_time': 0,
                'success_rate': 0
            }
            return self.baseline_stats
    
    def update_current_stats(self, ticket: Dict[str, Any]):
        """
        Update current statistics with a new ticket
        """
        # Update category counts
        category = ticket.get('category', 'unknown')
        self.current_stats['category'][category] += 1
        
        # Update priority counts
        priority = ticket.get('priority', 'unknown')
        self.current_stats['priority'][priority] += 1
        
        # Update severity counts
        severity = ticket.get('severity', 'unknown')
        self.current_stats['severity'][severity] += 1
        
        # Update sentiment counts
        sentiment = ticket.get('customer_sentiment', 'unknown')
        self.current_stats['sentiment'][sentiment] += 1
        
        # Update product counts
        product = ticket.get('product', 'unknown')
        self.current_stats['product'][product] += 1
        
        # Update resolution time tracking
        resolution_time = ticket.get('resolution_time_hours', 0)
        self.current_stats['resolution_times'].append(resolution_time)
        
        # Update success rate tracking
        resolution_helpful = ticket.get('resolution_helpful', False)
        self.current_stats['resolution_helpful'].append(resolution_helpful)
    
    def detect_anomalies(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the current ticket patterns
        """
        anomalies = []
        
        if not self.baseline_stats:
            logger.warning("No baseline statistics available for anomaly detection")
            return anomalies
        
        # Calculate current period statistics - handle both dict and list values
        current_total = 0
        for key, value in self.current_stats.items():
            if isinstance(value, dict):
                current_total += sum(value.values())
            elif isinstance(value, list):
                current_total += len(value)
        
        if current_total == 0:
            logger.info("No current tickets to analyze for anomalies")
            return anomalies
        
        # 1. Unusual ticket volume patterns per category
        category_anomalies = self._detect_category_anomalies()
        anomalies.extend(category_anomalies)
        
        # 2. Sentiment shifts in specific product areas
        sentiment_anomalies = self._detect_sentiment_anomalies()
        anomalies.extend(sentiment_anomalies)
        
        # 3. Resolution time anomalies
        resolution_time_anomalies = self._detect_resolution_time_anomalies()
        anomalies.extend(resolution_time_anomalies)
        
        # 4. Success rate anomalies
        success_rate_anomalies = self._detect_success_rate_anomalies()
        anomalies.extend(success_rate_anomalies)
        
        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def _detect_category_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect unusual ticket volume patterns per category
        """
        anomalies = []
        baseline_categories = self.baseline_stats['category_distribution']
        current_categories = self.current_stats['category']
        
        if not baseline_categories:
            return anomalies
        
        baseline_total = sum(baseline_categories.values())
        
        for category, current_count in current_categories.items():
            baseline_count = baseline_categories.get(category, 0)
            baseline_rate = baseline_count / baseline_total if baseline_total > 0 else 0
            current_rate = current_count / sum(current_categories.values()) if sum(current_categories.values()) > 0 else 0
            
            # Calculate anomaly score (simple approach)
            if baseline_rate > 0:
                ratio = current_rate / baseline_rate
                if ratio > threshold or ratio < (1/threshold):
                    anomalies.append({
                        'type': 'category_volume_anomaly',
                        'category': category,
                        'baseline_rate': baseline_rate,
                        'current_rate': current_rate,
                        'anomaly_score': ratio,
                        'description': f"Unusual volume for category '{category}': baseline rate {baseline_rate:.4f}, current rate {current_rate:.4f}"
                    })
        
        return anomalies
    
    def _detect_sentiment_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect sentiment shifts in specific product areas
        """
        anomalies = []
        baseline_sentiments = self.baseline_stats['sentiment_distribution']
        current_sentiments = self.current_stats['sentiment']
        
        if not baseline_sentiments:
            return anomalies
        
        baseline_total = sum(baseline_sentiments.values())
        
        for sentiment, current_count in current_sentiments.items():
            baseline_count = baseline_sentiments.get(sentiment, 0)
            baseline_rate = baseline_count / baseline_total if baseline_total > 0 else 0
            current_rate = current_count / sum(current_sentiments.values()) if sum(current_sentiments.values()) > 0 else 0
            
            # Calculate anomaly score
            if baseline_rate > 0:
                ratio = current_rate / baseline_rate
                if ratio > threshold or ratio < (1/threshold):
                    anomalies.append({
                        'type': 'sentiment_shift_anomaly',
                        'sentiment': sentiment,
                        'baseline_rate': baseline_rate,
                        'current_rate': current_rate,
                        'anomaly_score': ratio,
                        'description': f"Sentiment shift for '{sentiment}': baseline rate {baseline_rate:.4f}, current rate {current_rate:.4f}"
                    })
        
        return anomalies
    
    def _detect_resolution_time_anomalies(self, z_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in resolution times
        """
        anomalies = []
        resolution_times = self.current_stats['resolution_times']
        
        if len(resolution_times) < 10:  # Need sufficient data
            return anomalies
        
        current_avg_time = np.mean(resolution_times)
        baseline_avg_time = self.baseline_stats['avg_resolution_time']
        
        if baseline_avg_time > 0:
            # Simple z-score approach
            z_score = abs(current_avg_time - baseline_avg_time) / baseline_avg_time
            
            if z_score > z_threshold:
                anomalies.append({
                    'type': 'resolution_time_anomaly',
                    'baseline_avg': baseline_avg_time,
                    'current_avg': current_avg_time,
                    'anomaly_score': z_score,
                    'description': f"Resolution time anomaly: baseline avg {baseline_avg_time:.2f} hours, current avg {current_avg_time:.2f} hours"
                })
        
        return anomalies
    
    def _detect_success_rate_anomalies(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect anomalies in resolution success rates
        """
        anomalies = []
        resolution_helpful = self.current_stats['resolution_helpful']
        
        if len(resolution_helpful) < 10:  # Need sufficient data
            return anomalies
        
        current_success_rate = np.mean(resolution_helpful)
        baseline_success_rate = self.baseline_stats['success_rate']
        
        success_rate_diff = abs(current_success_rate - baseline_success_rate)
        
        if success_rate_diff > threshold:
            anomalies.append({
                'type': 'success_rate_anomaly',
                'baseline_rate': baseline_success_rate,
                'current_rate': current_success_rate,
                'anomaly_score': success_rate_diff,
                'description': f"Success rate anomaly: baseline {baseline_success_rate:.4f}, current {current_success_rate:.4f}"
            })
        
        return anomalies
    
    def reset_current_stats(self):
        """
        Reset current statistics for a new monitoring period
        """
        self.current_stats = defaultdict(lambda: defaultdict(int))
        self.current_stats['resolution_times'] = []
        self.current_stats['resolution_helpful'] = []

def main():
    """
    Main function to demonstrate anomaly detection
    """
    # Initialize detector
    detector = AnomalyDetector()
    
    # Load baseline statistics
    baseline_stats = detector.load_baseline_statistics()
    
    # Simulate some new tickets
    sample_tickets = [
        {
            'category': 'Technical Issue',
            'priority': 'high',
            'severity': 'P2',
            'customer_sentiment': 'frustrated',
            'product': 'DataSync Pro',
            'resolution_time_hours': 27.83,
            'resolution_helpful': True
        },
        {
            'category': 'Feature Request',
            'priority': 'critical',
            'severity': 'P2',
            'customer_sentiment': 'frustrated',
            'product': 'CloudBackup Enterprise',
            'resolution_time_hours': 3.01,
            'resolution_helpful': True
        },
        {
            'category': 'Billing Issue',
            'priority': 'medium',
            'severity': 'P3',
            'customer_sentiment': 'satisfied',
            'product': 'DataSync Pro',
            'resolution_time_hours': 1.25,
            'resolution_helpful': True
        }
    ]
    
    # Update current stats
    for ticket in sample_tickets:
        detector.update_current_stats(ticket)
    
    # Detect anomalies
    anomalies = detector.detect_anomalies()
    
    print("\n=== ANOMALY DETECTION RESULTS ===")
    if anomalies:
        for anomaly in anomalies:
            print(f"Type: {anomaly['type']}")
            print(f"Description: {anomaly['description']}")
            print(f"Anomaly Score: {anomaly['anomaly_score']:.4f}")
            print()
    else:
        print("No anomalies detected in current ticket patterns")

if __name__ == "__main__":
    main()
