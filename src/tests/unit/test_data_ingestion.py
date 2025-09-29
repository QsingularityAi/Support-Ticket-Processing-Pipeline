import pytest
import pandas as pd
import json
from src.data.ingestion import ingest_support_tickets, validate_ticket_data

def test_ingest_support_tickets():
    """Test that support tickets are correctly ingested from JSON file"""
    # Create a sample JSON file for testing
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
        }
    ]
    
    # Test ingestion
    df = ingest_support_tickets(sample_tickets)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "ticket_id" in df.columns
    assert "subject" in df.columns
    assert "description" in df.columns

def test_validate_ticket_data():
    """Test that ticket data validation works correctly"""
    # Valid ticket data
    valid_tickets = pd.DataFrame([
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
        }
    ])
    
    # Invalid ticket data (missing required fields)
    invalid_tickets = pd.DataFrame([
        {
            "ticket_id": "TICKET-001",
            # Missing created_at
            "customer_id": "CUST-001",
            "subject": "Issue with product",
            # Missing description
            "category": "Technical",
            "subcategory": "Software",
            "priority": "High",
            "severity": "Critical"
        }
    ])
    
    # Test valid data
    is_valid, errors = validate_ticket_data(valid_tickets)
    assert is_valid == True
    assert len(errors) == 0
    
    # Test invalid data
    is_valid, errors = validate_ticket_data(invalid_tickets)
    assert is_valid == False
    assert len(errors) > 0
