# =============================================================================
# Importing Necessary Packages
# =============================================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Union, Annotated
import joblib
import pickle
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
from pathlib import Path
import uvicorn
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# REQUEST/RESPONSE Flow
# =============================================================================

class AccountApplication(BaseModel):
    """Input schema for fraud detection prediction"""
    
    # Demographic Information
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    gender: str = Field(..., pattern=r"^(M|F|Other)$", description="Applicant gender")
    employment_status: str = Field(..., description="Employment status")
    housing_status: str = Field(..., description="Housing situation")
    income: float = Field(..., ge=0, le=1000000, description="Annual income")
    
    # Contact Information
    phone_home_valid: bool = Field(..., description="Home phone validation status")
    phone_mobile_valid: bool = Field(..., description="Mobile phone validation status")
    email_is_free: bool = Field(..., description="Free email provider indicator")
    
    # Application Details  
    source: str = Field(..., description="Application source channel")
    device_os: str = Field(..., description="Device operating system")
    session_length_in_minutes: float = Field(..., ge=0, le=300, description="Session length")
    foreign_request: bool = Field(..., description="Foreign IP address indicator")
    
    # Financial Information
    proposed_credit_limit: float = Field(..., ge=0, le=100000, description="Requested credit limit")
    intended_balcon_amount: Optional[float] = Field(0, ge=0, description="Balance transfer amount")
    payment_type: str = Field(..., description="Preferred payment method")
    
    # Behavioral Patterns
    velocity_6h: int = Field(..., ge=0, le=20, description="Applications in 6 hours")
    velocity_24h: int = Field(..., ge=0, le=50, description="Applications in 24 hours") 
    velocity_4w: int = Field(..., ge=0, le=100, description="Applications in 4 weeks")
    zip_count_4w: int = Field(..., ge=0, le=500, description="ZIP applications in 4 weeks")
    
    # Address Information
    current_address_months_count: int = Field(..., ge=0, le=600, description="Months at current address")
    prev_address_months_count: int = Field(..., ge=0, le=600, description="Months at previous address")
    
    # Banking History
    customer_age: int = Field(..., ge=0, le=300, description="Customer relationship age")
    bank_months_count: int = Field(..., ge=0, le=300, description="Months since first bank interaction")
    bank_branch_count_8w: int = Field(..., ge=0, le=1000, description="Branch applications in 8 weeks")
    has_other_cards: bool = Field(..., description="Has cards with other institutions")
    
    # Identity Verification
    name_email_similarity: float = Field(..., ge=0, le=1, description="Name-email similarity score")
    date_of_birth_distinct_emails_4w: int = Field(..., ge=0, le=50, description="Emails with same DOB")
    keep_alive_session: bool = Field(..., description="Session persistence indicator")
    
    # Temporal
    month: int = Field(..., ge=1, le=12, description="Application month")
    
    @field_validator('employment_status')
    @classmethod
    def validate_employment(cls, v: str) -> str:
        valid_statuses = ['Employed', 'Unemployed', 'Student', 'Retired', 'Self-employed']
        if v not in valid_statuses:
            raise ValueError(f'Employment status must be one of {valid_statuses}')
        return v
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        valid_sources = ['INTERNET', 'BRANCH', 'PHONE', 'MOBILE']
        if v not in valid_sources:
            raise ValueError(f'Source must be one of {valid_sources}')
        return v
    
    @field_validator('housing_status')
    @classmethod
    def validate_housing(cls, v: str) -> str:
        valid_housing = ['Own', 'Rent', 'Mortgage', 'Other']
        if v not in valid_housing:
            raise ValueError(f'Housing status must be one of {valid_housing}')
        return v
    
    @field_validator('device_os')
    @classmethod
    def validate_device_os(cls, v: str) -> str:
        valid_os = ['Windows', 'iOS', 'Android', 'MacOS', 'Linux']
        if v not in valid_os:
            raise ValueError(f'Device OS must be one of {valid_os}')
        return v
    
    @field_validator('payment_type')
    @classmethod
    def validate_payment_type(cls, v: str) -> str:
        valid_payment = ['ACH', 'CARD', 'WIRE', 'CHECK']
        if v not in valid_payment:
            raise ValueError(f'Payment type must be one of {valid_payment}')
        return v

class FraudPredictionResponse(BaseModel):
    """Response schema for fraud detection prediction"""
    
    application_id: str = Field(..., description="Unique application identifier")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    risk_level: str = Field(..., description="Risk categorization")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence")
    processing_timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    
    # Risk factors
    top_risk_factors: List[Dict[str, Union[str, float]]] = Field(..., description="Top contributing risk factors")
    recommendation: str = Field(..., description="Business recommendation")

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests"""
    
    applications: List[AccountApplication] = Field(..., max_length=1000, description="Batch of applications")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_applications: int = Field(..., description="Number of applications processed")
    fraud_detected: int = Field(..., description="Number of fraud cases detected")
    fraud_rate: float = Field(..., description="Percentage of fraud in batch")
    processing_time_seconds: float = Field(..., description="Total processing time")
    predictions: List[FraudPredictionResponse] = Field(..., description="Individual predictions")

# =======================
# Model Loader, Fast API
# =======================

class FraudDetectionModel:
    """Fraud detection model wrapper for production serving"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model = None
        self.encoders = None
        self.config = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load model and preprocessing components"""
        try:
            # Load model
            model_path = self.model_dir / "best_model.joblib"
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded from {model_path}")
            
            # Load encoders
            encoders_path = self.model_dir / "encoders.pkl"
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            logger.info(f"✅ Encoders loaded from {encoders_path}")
            
            # Load configuration
            config_path = self.model_dir / "model_config.json"
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.feature_names = self.config['feature_names']
            logger.info(f"✅ Configuration loaded: {self.config['model_name']}")
            logger.info(f"📊 Model Performance: Recall={self.config.get('recall_achieved', 'N/A'):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def preprocess_application(self, application: AccountApplication) -> pd.DataFrame:
        """Apply feature engineering pipeline to single application"""
        
        # Convert Pydantic model to dictionary
        app_dict = application.model_dump()  # Updated for Pydantic V2
        
        # Create DataFrame
        df = pd.DataFrame([app_dict])
        
        # Apply categorical encoding (same as training pipeline)
        categorical_features = ['gender', 'employment_status', 'housing_status', 'source', 
                              'device_os', 'payment_type']
        
        for feature in categorical_features:
            if feature in df.columns:
                # Label encoding
                if f"{feature}_label" in self.encoders:
                    encoder = self.encoders[f"{feature}_label"]
                    try:
                        df[f"{feature}_encoded"] = encoder.transform(df[feature].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df[f"{feature}_encoded"] = 0
                        logger.warning(f"Unseen category in {feature}: {df[feature].iloc[0]}")
                
                # Frequency encoding
                if f"{feature}_frequency" in self.encoders:
                    freq_map = self.encoders[f"{feature}_frequency"]
                    df[f"{feature}_frequency"] = df[feature].map(freq_map).fillna(1)
                
                # Target encoding
                if f"{feature}_target" in self.encoders:
                    target_encoder = self.encoders[f"{feature}_target"]
                    target_means = target_encoder['target_means']
                    global_mean = target_encoder['global_mean']
                    df[f"{feature}_target_encoded"] = df[feature].map(target_means).fillna(global_mean)
        
        # Select only the features used in training
        available_features = [col for col in self.feature_names if col in df.columns]
        missing_features = [col for col in self.feature_names if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with 0 or mean values
            for feature in missing_features:
                df[feature] = 0
        
        return df[self.feature_names]
    
    def predict_fraud(self, application: AccountApplication) -> Dict:
        """Generate fraud prediction for single application"""
        
        # Preprocess application
        processed_data = self.preprocess_application(application)
        
        # Get prediction probability
        fraud_probability = self.model.predict_proba(processed_data)[0, 1]
        
        # Apply optimal threshold
        optimal_threshold = self.config['optimal_threshold']
        fraud_prediction = fraud_probability >= optimal_threshold
        
        # Calculate confidence and risk level
        confidence_score = abs(fraud_probability - 0.5) * 2  # Distance from decision boundary
        
        if fraud_probability >= 0.8:
            risk_level = "HIGH"
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"  
        elif fraud_probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"
        
        # Get feature importance (simplified)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_risk_factors = []
            
            for idx in top_features_idx:
                feature_name = self.feature_names[idx]
                feature_value = processed_data.iloc[0, idx]
                importance = feature_importance[idx]
                
                top_risk_factors.append({
                    "feature": feature_name,
                    "value": float(feature_value),
                    "importance": float(importance)
                })
        else:
            top_risk_factors = []
        
        # Business recommendation
        if fraud_prediction:
            if fraud_probability >= 0.9:
                recommendation = "REJECT - High fraud probability"
            elif fraud_probability >= 0.7:
                recommendation = "MANUAL_REVIEW - Manual verification required"
            else:
                recommendation = "ENHANCED_VERIFICATION - Additional checks recommended"
        else:
            recommendation = "APPROVE - Low fraud risk"
        
        return {
            'fraud_probability': float(fraud_probability),
            'fraud_prediction': bool(fraud_prediction),
            'risk_level': risk_level,
            'confidence_score': float(confidence_score),
            'top_risk_factors': top_risk_factors,
            'recommendation': recommendation
        }

# Initialize FastAPI app
app = FastAPI(
    title="Bank Account Fraud Detection API",
    description="Real-time fraud detection system for bank account applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model instance
fraud_model = None

@app.on_event("startup")
async def startup_event():
    """Load model on application startup with comprehensive error handling"""
    global fraud_model
    
    logger.info("🚀 Starting Fraud Detection API...")
    
    # Debug current directory
    current_dir = Path(".").absolute()
    logger.info(f"📁 Current directory: {current_dir}")
    logger.info(f"📋 Directory contents: {[item.name for item in current_dir.iterdir()]}")
    
    # Look for model directories
    model_directories = [d for d in current_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("fraud_model_")]
    
    logger.info(f"🔍 Found {len(model_directories)} model directories: {[d.name for d in model_directories]}")
    
    if not model_directories:
        # Try looking in subdirectories
        logger.info("🔍 Searching subdirectories for model...")
        for item in current_dir.iterdir():
            if item.is_dir():
                sub_models = [d for d in item.iterdir() 
                             if d.is_dir() and d.name.startswith("fraud_model_")]
                if sub_models:
                    model_directories.extend(sub_models)
                    logger.info(f"📂 Found models in {item.name}: {[d.name for d in sub_models]}")
    
    if not model_directories:
        error_msg = (
            "❌ No fraud model directories found!\n"
            "Expected directory name pattern: 'fraud_model_YYYYMMDD_HHMMSS'\n"
            f"Current directory: {current_dir}\n"
            f"Contents: {[item.name for item in current_dir.iterdir()]}"
        )
        logger.error(error_msg)
        raise RuntimeError("No fraud model directory found")
    
    # Use the most recent model directory
    model_directory = sorted(model_directories, key=lambda x: x.name)[-1]
    logger.info(f"📂 Selected model directory: {model_directory}")
    
    # Verify required files
    required_files = ["best_model.joblib", "encoders.pkl", "model_config.json"]
    missing_files = []
    
    for file_name in required_files:
        file_path = model_directory / file_name
        if file_path.exists():
            logger.info(f"✅ Found: {file_name} ({file_path.stat().st_size} bytes)")
        else:
            missing_files.append(file_name)
            logger.error(f"❌ Missing: {file_name}")
    
    if missing_files:
        logger.error(f"📁 {model_directory} contents: {[f.name for f in model_directory.iterdir()]}")
        raise RuntimeError(f"Missing required files: {missing_files}")
    
    # Load the model
    try:
        fraud_model = FraudDetectionModel(str(model_directory))
        logger.info("✅ Fraud Detection API started successfully!")
        logger.info(f"🎯 Model: {fraud_model.config['model_name']}")
        logger.info(f"📊 Performance: Recall={fraud_model.config.get('recall_achieved', 'N/A'):.1%}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Bank Account Fraud Detection API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": fraud_model is not None,
        "model_info": fraud_model.config if fraud_model else None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_name": fraud_model.config['model_name'],
        "model_performance": {
            "recall": fraud_model.config.get('recall_achieved'),
            "precision": fraud_model.config.get('precision_achieved')
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(application: AccountApplication):
    """Single fraud prediction endpoint"""
    
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate unique application ID
        application_id = f"APP_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Get prediction
        prediction_result = fraud_model.predict_fraud(application)
        
        # Create response
        response = FraudPredictionResponse(
            application_id=application_id,
            fraud_probability=prediction_result['fraud_probability'],
            fraud_prediction=prediction_result['fraud_prediction'],
            risk_level=prediction_result['risk_level'],
            confidence_score=prediction_result['confidence_score'],
            processing_timestamp=datetime.now().isoformat(),
            model_version=fraud_model.config['model_name'],
            top_risk_factors=prediction_result['top_risk_factors'],
            recommendation=prediction_result['recommendation']
        )
        
        logger.info(f"Prediction made for {application_id}: {prediction_result['fraud_prediction']}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Batch fraud prediction endpoint"""
    
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        batch_id = request.batch_id or f"BATCH_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        predictions = []
        fraud_count = 0
        
        # Process each application
        for i, application in enumerate(request.applications):
            application_id = f"{batch_id}_APP_{i:04d}"
            
            prediction_result = fraud_model.predict_fraud(application)
            
            response = FraudPredictionResponse(
                application_id=application_id,
                fraud_probability=prediction_result['fraud_probability'],
                fraud_prediction=prediction_result['fraud_prediction'],
                risk_level=prediction_result['risk_level'],
                confidence_score=prediction_result['confidence_score'],
                processing_timestamp=datetime.now().isoformat(),
                model_version=fraud_model.config['model_name'],
                top_risk_factors=prediction_result['top_risk_factors'],
                recommendation=prediction_result['recommendation']
            )
            
            predictions.append(response)
            if prediction_result['fraud_prediction']:
                fraud_count += 1
        
        # Calculate batch statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        fraud_rate = fraud_count / len(request.applications) if request.applications else 0
        
        batch_response = BatchPredictionResponse(
            batch_id=batch_id,
            total_applications=len(request.applications),
            fraud_detected=fraud_count,
            fraud_rate=fraud_rate,
            processing_time_seconds=processing_time,
            predictions=predictions
        )
        
        logger.info(f"Batch processed: {batch_id}, {len(request.applications)} apps, {fraud_count} fraud detected")
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_config": fraud_model.config,
        "feature_count": len(fraud_model.feature_names),
        "encoders_loaded": len(fraud_model.encoders),
        "model_type": type(fraud_model.model).__name__
    }

if __name__ == "__main__":
    uvicorn.run(
        "fraud_detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )