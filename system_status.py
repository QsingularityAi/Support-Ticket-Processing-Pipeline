#!/usr/bin/env python3
"""
System Status Checker for Intelligent Support System
Provides comprehensive health check and performance monitoring
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Optional
from datetime import datetime

class SystemStatusChecker:
    """
    Comprehensive system status checker
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.services = {
            "API": "http://localhost:8001/health",
            "Prometheus": "http://localhost:9090/-/healthy", 
            "Grafana": "http://localhost:3000/api/health",
            "MLflow": "http://localhost:5000/health"
        }
        
    def check_service_health(self, name: str, url: str) -> Dict:
        """Check individual service health"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            duration = time.time() - start_time
            
            return {
                "name": name,
                "status": "✅ Healthy" if response.status_code == 200 else f"❌ Error ({response.status_code})",
                "response_time": f"{duration*1000:.1f}ms",
                "healthy": response.status_code == 200
            }
        except requests.exceptions.ConnectionError:
            return {
                "name": name,
                "status": "❌ Not Running",
                "response_time": "N/A",
                "healthy": False
            }
        except requests.exceptions.Timeout:
            return {
                "name": name,
                "status": "⏰ Timeout",
                "response_time": ">10s",
                "healthy": False
            }
        except Exception as e:
            return {
                "name": name,
                "status": f"❌ Error: {str(e)}",
                "response_time": "N/A",
                "healthy": False
            }
    
    def check_api_endpoints(self) -> Dict:
        """Check API endpoint functionality"""
        endpoints = {
            "Health": "/health",
            "Model Info": "/model_info",
            "Metrics": "/metrics"
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            url = f"{self.base_url}{endpoint}"
            result = self.check_service_health(name, url)
            results[name] = result
            
        return results
    
    def test_prediction_endpoint(self) -> Dict:
        """Test the prediction endpoint with sample data"""
        try:
            test_data = {
                "subject": "Test database connection issue",
                "description": "Testing the prediction endpoint functionality",
                "product": "DataSync Pro",
                "environment": "production"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/predict", json=test_data, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "✅ Working",
                    "response_time": f"{duration*1000:.1f}ms",
                    "category": data.get("category", "N/A"),
                    "confidence": f"{data.get('confidence_scores', {}).get('category', 0):.3f}",
                    "healthy": True
                }
            else:
                return {
                    "status": f"❌ Error ({response.status_code})",
                    "response_time": f"{duration*1000:.1f}ms",
                    "healthy": False
                }
        except Exception as e:
            return {
                "status": f"❌ Error: {str(e)}",
                "response_time": "N/A",
                "healthy": False
            }
    
    def test_retrieval_endpoint(self) -> Dict:
        """Test the solution retrieval endpoint"""
        try:
            test_data = {
                "query": "database timeout connection error",
                "category": "Technical Issue",
                "product": "DataSync Pro"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/retrieve_solutions", json=test_data, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "✅ Working",
                    "response_time": f"{duration*1000:.1f}ms",
                    "results_count": len(data),
                    "top_relevance": f"{data[0]['relevance_score']:.3f}" if data else "N/A",
                    "healthy": True
                }
            else:
                return {
                    "status": f"❌ Error ({response.status_code})",
                    "response_time": f"{duration*1000:.1f}ms",
                    "healthy": False
                }
        except Exception as e:
            return {
                "status": f"❌ Error: {str(e)}",
                "response_time": "N/A",
                "healthy": False
            }
    
    def check_docker_services(self) -> Dict:
        """Check Docker services status"""
        try:
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                services = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            service = json.loads(line)
                            services.append({
                                "name": service.get("Service", "Unknown"),
                                "state": service.get("State", "Unknown"),
                                "status": "✅ Running" if service.get("State") == "running" else f"❌ {service.get('State', 'Unknown')}"
                            })
                        except json.JSONDecodeError:
                            continue
                
                return {"services": services, "healthy": all(s["state"] == "running" for s in services)}
            else:
                return {"services": [], "healthy": False, "error": "Docker Compose not available"}
                
        except subprocess.TimeoutExpired:
            return {"services": [], "healthy": False, "error": "Timeout checking Docker services"}
        except FileNotFoundError:
            return {"services": [], "healthy": False, "error": "Docker Compose not installed"}
        except Exception as e:
            return {"services": [], "healthy": False, "error": str(e)}
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            # Check disk space
            result = subprocess.run(["df", "-h", "."], capture_output=True, text=True)
            disk_info = result.stdout.split('\n')[1].split() if result.returncode == 0 else []
            
            # Check memory (on macOS/Linux)
            try:
                if sys.platform == "darwin":  # macOS
                    result = subprocess.run(["vm_stat"], capture_output=True, text=True)
                    memory_info = "Available (use Activity Monitor for details)"
                else:  # Linux
                    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
                    memory_info = result.stdout.split('\n')[1] if result.returncode == 0 else "Unknown"
            except:
                memory_info = "Unknown"
            
            return {
                "disk_usage": disk_info[4] if len(disk_info) > 4 else "Unknown",
                "disk_available": disk_info[3] if len(disk_info) > 3 else "Unknown",
                "memory_info": memory_info,
                "healthy": True
            }
        except Exception as e:
            return {
                "disk_usage": "Unknown",
                "disk_available": "Unknown", 
                "memory_info": "Unknown",
                "healthy": False,
                "error": str(e)
            }
    
    def generate_report(self) -> None:
        """Generate comprehensive system status report"""
        print("🔍 Intelligent Support System - Status Report")
        print("=" * 60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check services
        print("🏥 Service Health Check")
        print("-" * 30)
        all_healthy = True
        
        for name, url in self.services.items():
            result = self.check_service_health(name, url)
            print(f"{result['status']:<20} {name:<15} ({result['response_time']})")
            if not result['healthy']:
                all_healthy = False
        
        print()
        
        # Check API endpoints
        print("📡 API Endpoints")
        print("-" * 20)
        api_results = self.check_api_endpoints()
        for name, result in api_results.items():
            print(f"{result['status']:<20} {name:<15} ({result['response_time']})")
            if not result['healthy']:
                all_healthy = False
        
        print()
        
        # Test prediction
        print("🎯 ML Model Testing")
        print("-" * 25)
        prediction_result = self.test_prediction_endpoint()
        print(f"{prediction_result['status']:<20} Prediction ({prediction_result['response_time']})")
        if prediction_result['healthy']:
            print(f"   Category: {prediction_result.get('category', 'N/A')}")
            print(f"   Confidence: {prediction_result.get('confidence', 'N/A')}")
        else:
            all_healthy = False
        
        print()
        
        # Test retrieval
        print("🔍 RAG Retrieval Testing")
        print("-" * 30)
        retrieval_result = self.test_retrieval_endpoint()
        print(f"{retrieval_result['status']:<20} Retrieval ({retrieval_result['response_time']})")
        if retrieval_result['healthy']:
            print(f"   Results: {retrieval_result.get('results_count', 'N/A')}")
            print(f"   Top Relevance: {retrieval_result.get('top_relevance', 'N/A')}")
        else:
            all_healthy = False
        
        print()
        
        # Check Docker services
        print("🐳 Docker Services")
        print("-" * 20)
        docker_result = self.check_docker_services()
        if docker_result['healthy']:
            for service in docker_result['services']:
                print(f"{service['status']:<20} {service['name']}")
        else:
            print(f"❌ Docker Error: {docker_result.get('error', 'Unknown')}")
            all_healthy = False
        
        print()
        
        # Check system resources
        print("💻 System Resources")
        print("-" * 25)
        resource_result = self.check_system_resources()
        if resource_result['healthy']:
            print(f"Disk Usage: {resource_result['disk_usage']}")
            print(f"Disk Available: {resource_result['disk_available']}")
            print(f"Memory: {resource_result['memory_info']}")
        else:
            print(f"❌ Resource Check Error: {resource_result.get('error', 'Unknown')}")
        
        print()
        
        # Overall status
        print("📊 Overall System Status")
        print("-" * 30)
        if all_healthy:
            print("🎉 ✅ ALL SYSTEMS OPERATIONAL")
            print("   The Intelligent Support System is running optimally!")
        else:
            print("⚠️  ❌ ISSUES DETECTED")
            print("   Some components need attention. Check the details above.")
        
        print()
        print("🔗 Quick Links:")
        print("   • API Docs:    http://localhost:8001/docs")
        print("   • Prometheus:  http://localhost:9090")
        print("   • Grafana:     http://localhost:3000")
        print("   • MLflow:      http://localhost:5000")
        print()

def main():
    """Main function"""
    checker = SystemStatusChecker()
    checker.generate_report()

if __name__ == "__main__":
    main()