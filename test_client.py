# =============================================================================
# CLIENT EXAMPLE - test_client.py
# =============================================================================

import requests
import json
from datetime import datetime

# API Base URL
BASE_URL = "http://localhost:8000"

def test_single_prediction():
    """Test single fraud prediction"""
    
    # Sample application data
    application_data = {
            "age": 22,
            "gender": "M",
            "employment_status": "Student",
            "housing_status": "Rent",
            "income": 12000,
            "phone_home_valid": False,
            "phone_mobile_valid": False,
            "email_is_free": True,
            "source": "INTERNET",
            "device_os": "Android",
            "session_length_in_minutes": 4.2,
            "foreign_request": True,
            "proposed_credit_limit": 50000,
            "intended_balcon_amount": 15000,
            "payment_type": "WIRE",
            "velocity_6h": 8,
            "velocity_24h": 12,
            "velocity_4w": 35,
            "zip_count_4w": 67,
            "current_address_months_count": 3,
            "prev_address_months_count": 8,
            "customer_age": 0,
            "bank_months_count": 1,
            "bank_branch_count_8w": 89,
            "has_other_cards": False,
            "name_email_similarity": 0.1,
            "date_of_birth_distinct_emails_4w": 12,
            "keep_alive_session": False,
            "month": 11
    }
    
    # Make prediction request
    response = requests.post(f"{BASE_URL}/predict", json=application_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print(f"Application ID: {result['application_id']}")
        print(f"Fraud Probability: {result['fraud_probability']:.3f}")
        print(f"Fraud Prediction: {result['fraud_prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Top Risk Factors:")
        for factor in result['top_risk_factors']:
            print(f"  • {factor['feature']}: {factor['importance']:.3f}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(response.text)

def test_batch_prediction():
    """Test batch fraud prediction"""
    
    # Sample batch data (2 applications)
    batch_data = {
        "applications": [
            {
                "age": 35,
                "gender": "F",
                "employment_status": "Employed",
                "housing_status": "Own",
                "income": 65000,
                "phone_home_valid": True,
                "phone_mobile_valid": True,
                "email_is_free": False,
                "source": "BRANCH",
                "device_os": "iOS",
                "session_length_in_minutes": 25.0,
                "foreign_request": False,
                "proposed_credit_limit": 8000,
                "intended_balcon_amount": 0,
                "payment_type": "CARD",
                "velocity_6h": 1,
                "velocity_24h": 1,
                "velocity_4w": 1,
                "zip_count_4w": 3,
                "current_address_months_count": 48,
                "prev_address_months_count": 96,
                "customer_age": 60,
                "bank_months_count": 72,
                "bank_branch_count_8w": 8,
                "has_other_cards": True,
                "name_email_similarity": 0.8,
                "date_of_birth_distinct_emails_4w": 1,
                "keep_alive_session": True,
                "month": 6
            },
            {
                "age": 22,
                "gender": "M",
                "employment_status": "Student",
                "housing_status": "Rent",
                "income": 12000,
                "phone_home_valid": False,
                "phone_mobile_valid": False,
                "email_is_free": True,
                "source": "INTERNET",
                "device_os": "Android",
                "session_length_in_minutes": 4.2,
                "foreign_request": True,
                "proposed_credit_limit": 50000,
                "intended_balcon_amount": 15000,
                "payment_type": "WIRE",
                "velocity_6h": 8,
                "velocity_24h": 12,
                "velocity_4w": 35,
                "zip_count_4w": 67,
                "current_address_months_count": 3,
                "prev_address_months_count": 8,
                "customer_age": 0,
                "bank_months_count": 1,
                "bank_branch_count_8w": 89,
                "has_other_cards": False,
                "name_email_similarity": 0.1,
                "date_of_birth_distinct_emails_4w": 12,
                "keep_alive_session": False,
                "month": 11
            }
        ],
        "batch_id": "TEST_BATCH_001"
    }
    
    # Make batch prediction request
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Batch prediction successful!")
        print(f"Batch ID: {result['batch_id']}")
        print(f"Applications: {result['total_applications']}")
        print(f"Fraud Detected: {result['fraud_detected']}")
        print(f"Fraud Rate: {result['fraud_rate']:.2%}")
        print(f"Processing Time: {result['processing_time_seconds']:.2f}s")
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("🧪 Testing Fraud Detection API")
    print("\n1. Testing Single Prediction:")
    test_single_prediction()
    
    print("\n2. Testing Batch Prediction:")
    test_batch_prediction()
