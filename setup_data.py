#!/usr/bin/env python3
"""
Dataset setup script - generates training data and initializes artifacts
Run this after cloning the repository to create the necessary large files
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import random

def create_directories():
    """Create necessary directories"""
    dirs = [
        "artifacts/models",
        "artifacts/feature_transformers", 
        "data/splits",
        "logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Created directory structure")

def generate_sample_data():
    """Generate synthetic ticket data for development/testing"""
    print("🔄 Generating sample ticket data...")
    
    # Load the sample tickets
    with open('sample_tickets.json', 'r') as f:
        sample_tickets = json.load(f)
    
    # Categories and subcategories
    categories = ["Technical Issue", "Account Management", "Feature Request", "Data Issue", "Security"]
    subcategories = ["Database", "Authentication", "Billing", "API", "Performance", "Compatibility", 
                    "Bug", "Configuration", "Integration", "Licensing"]
    priorities = ["low", "medium", "high", "critical"] 
    severities = ["P0", "P1", "P2", "P3", "P4"]
    products = ["DataSync Pro", "CloudBackup Enterprise", "Analytics Platform", "Security Suite"]
    
    # Generate additional tickets based on the sample pattern
    generated_tickets = []
    base_ticket = sample_tickets[0]  # Use first ticket as template
    
    for i in range(10000):  # Generate 10k tickets for training
        ticket = base_ticket.copy()
        
        # Update fields
        ticket['ticket_id'] = f"TICKET-{i+2:06d}"
        ticket['created_at'] = (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat() + "Z"
        ticket['customer_id'] = f"CUST-{random.randint(1, 1000):04d}"
        ticket['product'] = random.choice(products)
        ticket['subject'] = f"Sample ticket {i+2} - {random.choice(['timeout', 'error', 'performance', 'access'])} issue"
        ticket['category'] = random.choice(categories)
        ticket['subcategory'] = random.choice(subcategories)
        ticket['priority'] = random.choice(priorities)
        ticket['severity'] = random.choice(severities)
        
        # Make description slightly different
        ticket['description'] = f"This is a generated test ticket {i+2} for development purposes."
        
        generated_tickets.append(ticket)
    
    # Combine sample + generated
    all_tickets = sample_tickets + generated_tickets
    
    # Convert to DataFrame
    df = pd.DataFrame(all_tickets)
    
    # Split data
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    # Save splits
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)
    
    print(f"✅ Generated {len(all_tickets)} tickets")
    print(f"   - Train: {len(train_df)} tickets")
    print(f"   - Validation: {len(val_df)} tickets")
    print(f"   - Test: {len(test_df)} tickets")

def create_placeholder_files():
    """Create placeholder files for missing artifacts"""
    print("🔄 Creating placeholder files...")
    
    # Create empty artifact files (these will be populated after training)
    placeholder_files = [
        "artifacts/models/xgboost_category/model.pkl",
        "artifacts/models/xgboost_category/label_encoder.pkl",
        "artifacts/models/xgboost_subcategory/model.pkl", 
        "artifacts/models/xgboost_subcategory/label_encoder.pkl",
        "artifacts/models/xgboost_priority/model.pkl",
        "artifacts/models/xgboost_priority/label_encoder.pkl",
        "artifacts/models/xgboost_severity/model.pkl",
        "artifacts/models/xgboost_severity/label_encoder.pkl",
        "artifacts/feature_transformers/text_vectorizers.pkl",
        "artifacts/feature_transformers/label_encoders.pkl",
        "artifacts/feature_transformers/scalers.pkl",
        "artifacts/optimized_retrieval_index.pkl"
    ]
    
    for file_path in placeholder_files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write("# Placeholder - run 'python complete_pipeline.py' to generate actual files\n")
    
    print("✅ Created placeholder artifact files")

def main():
    """Main setup function"""
    print("🚀 Setting up Support Ticket Processing Pipeline data...")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Generate sample data
    generate_sample_data()
    
    # Step 3: Create placeholder files
    create_placeholder_files()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run: python complete_pipeline.py")
    print("2. Start API: PYTHONPATH=. python src/api/main.py")
    print("3. Test: python api_test_comprehensive.py")

if __name__ == "__main__":
    main()
