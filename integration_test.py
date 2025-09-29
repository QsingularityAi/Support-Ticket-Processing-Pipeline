#!/usr/bin/env python3
"""
Comprehensive Integration Test for the 3-Layer Intelligent Support System
Tests the complete workflow: Categorization → RAG Retrieval → Anomaly Detection
"""

import requests
import json
import time
from typing import Dict, List, Any

class SupportSystemIntegrationTest:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    def test_health_check(self) -> bool:
        """Test API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            success = response.status_code == 200
            self.log_test("Health Check", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_categorization_layer(self) -> Dict[str, Any]:
        """Test Layer 1: XGBoost Categorization"""
        test_ticket = {
            "subject": "Critical database connection timeout in production",
            "description": "Our DataSync Pro application is experiencing severe database connection timeouts affecting 500+ users. Error: Connection timeout after 30 seconds. This started 2 hours ago.",
            "product": "DataSync Pro",
            "channel": "email",
            "environment": "production",
            "business_impact": "high",
            "language": "english"
        }
        
        try:
            response = requests.post(f"{self.base_url}/predict", json=test_ticket)
            if response.status_code == 200:
                result = response.json()
                self.log_test("Categorization Layer", True, 
                            f"Category: {result['category']}, Priority: {result['priority']}")
                return result
            else:
                self.log_test("Categorization Layer", False, f"Status: {response.status_code}")
                return {}
        except Exception as e:
            self.log_test("Categorization Layer", False, f"Error: {str(e)}")
            return {}
    
    def test_rag_retrieval_layer(self, category: str = "Technical Issue") -> List[Dict]:
        """Test Layer 2: RAG Solution Retrieval"""
        query_data = {
            "query": "database timeout connection error performance issues production critical",
            "category": category,
            "product": "DataSync Pro"
        }
        
        try:
            response = requests.post(f"{self.base_url}/retrieve_solutions", json=query_data)
            if response.status_code == 200:
                results = response.json()
                self.log_test("RAG Retrieval Layer", True, 
                            f"Found {len(results)} solutions, top relevance: {results[0]['relevance_score']:.3f}")
                return results
            else:
                self.log_test("RAG Retrieval Layer", False, f"Status: {response.status_code}")
                return []
        except Exception as e:
            self.log_test("RAG Retrieval Layer", False, f"Error: {str(e)}")
            return []
    
    def test_anomaly_detection_layer(self) -> List[Dict]:
        """Test Layer 3: Anomaly Detection"""
        # Create a pattern that should trigger anomaly detection
        tickets = []
        
        # Generate multiple similar tickets (should trigger volume anomaly)
        for i in range(15):
            tickets.append({
                "subject": f"Database timeout error #{i+1}",
                "description": "Connection timeout issues affecting production",
                "product": "DataSync Pro",
                "customer_tier": "Premium",
                "category": "Technical Issue",
                "priority": "critical" if i > 10 else "high",
                "customer_sentiment": "angry" if i > 8 else "frustrated"
            })
        
        # Add some normal tickets
        for i in range(3):
            tickets.append({
                "subject": f"License question #{i+1}",
                "description": "Need help with licensing",
                "product": "DataSync Pro",
                "customer_tier": "Basic",
                "category": "Account Management", 
                "priority": "low",
                "customer_sentiment": "neutral"
            })
        
        try:
            response = requests.post(f"{self.base_url}/detect_anomalies", json=tickets)
            if response.status_code == 200:
                anomalies = response.json()
                self.log_test("Anomaly Detection Layer", True, 
                            f"Detected {len(anomalies)} anomalies from {len(tickets)} tickets")
                return anomalies
            else:
                self.log_test("Anomaly Detection Layer", False, f"Status: {response.status_code}")
                return []
        except Exception as e:
            self.log_test("Anomaly Detection Layer", False, f"Error: {str(e)}")
            return []
    
    def test_end_to_end_workflow(self):
        """Test complete workflow integration"""
        print("\n🔄 Testing End-to-End Workflow Integration")
        print("=" * 60)
        
        # Step 1: Categorize a ticket
        print("Step 1: Categorizing incoming ticket...")
        categorization = self.test_categorization_layer()
        
        if not categorization:
            self.log_test("E2E Workflow", False, "Categorization failed")
            return
        
        # Step 2: Use category to improve retrieval
        print("Step 2: Retrieving relevant solutions...")
        category = categorization.get('category', 'Technical Issue')
        solutions = self.test_rag_retrieval_layer(category)
        
        if not solutions:
            self.log_test("E2E Workflow", False, "Solution retrieval failed")
            return
        
        # Step 3: Check for anomalies
        print("Step 3: Analyzing for anomalies...")
        anomalies = self.test_anomaly_detection_layer()
        
        # Verify integration
        workflow_success = bool(categorization and solutions)
        self.log_test("E2E Workflow Integration", workflow_success,
                     f"Category→Solutions→Anomalies: {len(solutions)} solutions, {len(anomalies)} anomalies")
    
    def test_system_performance(self):
        """Test system performance under load"""
        print("\n⚡ Testing System Performance")
        print("=" * 40)
        
        # Test prediction speed
        start_time = time.time()
        for i in range(5):
            self.test_categorization_layer()
        prediction_time = (time.time() - start_time) / 5
        
        # Test retrieval speed  
        start_time = time.time()
        for i in range(5):
            self.test_rag_retrieval_layer()
        retrieval_time = (time.time() - start_time) / 5
        
        performance_ok = prediction_time < 1.0 and retrieval_time < 2.0
        self.log_test("System Performance", performance_ok,
                     f"Avg prediction: {prediction_time:.3f}s, Avg retrieval: {retrieval_time:.3f}s")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Intelligent Product Support System - Integration Test Suite")
        print("=" * 70)
        
        # Basic connectivity
        if not self.test_health_check():
            print("❌ Cannot connect to API. Make sure server is running on port 8001")
            return
        
        # Test individual layers
        print("\n🧪 Testing Individual System Layers")
        print("=" * 45)
        self.test_categorization_layer()
        self.test_rag_retrieval_layer()
        self.test_anomaly_detection_layer()
        
        # Test integration
        self.test_end_to_end_workflow()
        
        # Test performance
        self.test_system_performance()
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 20)
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        print(f"Tests Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All systems working properly! The 3-layer architecture is fully functional.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            
        return passed == total

if __name__ == "__main__":
    tester = SupportSystemIntegrationTest()
    tester.run_all_tests()