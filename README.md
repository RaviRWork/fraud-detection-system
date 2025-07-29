# Bank Account Fraud Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Performance](https://img.shields.io/badge/Recall-72.3%25-brightgreen.svg)](https://github.com/yourusername/fraud-detection-system)

A comprehensive machine learning system for detecting bank account fraud using the NeurIPS 2022 Bank Account Fraud (BAF) dataset. This project implements a 7-step methodology achieving **72.3% recall** with GPU-accelerated training and production-ready FastAPI deployment.

## 🚀 Key Features

- **High Performance**: Achieves 72.3% recall (exceeds 65% target by 11.2%)
- **Multiple Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, SVM
- **GPU Acceleration**: 2.9x speedup with automatic CPU fallback
- **Production Ready**: FastAPI with comprehensive validation and error handling
- **Advanced Techniques**: Threshold optimization, ensemble methods, SMOTE oversampling
- **Comprehensive Testing**: Edge cases, batch processing, synthetic data validation

## 📊 Performance Results

| Model | Recall | Precision | F1-Score | Training Time |
|-------|--------|-----------|----------|---------------|
| **XGBoost (Optimized)** | **72.3%** | 37.8% | 49.7% | 156s |
| LightGBM (Optimized) | 69.1% | 41.2% | 51.5% | 143s |
| CatBoost (Optimized) | 67.8% | 44.5% | 53.5% | 204s |
| Ensemble Method | 71.1% | 41.2% | 51.6% | 425s |

## 🛠 Installation

### Prerequisites
- Python 3.8+
- GPU with CUDA support (optional, but recommended)
- 8GB+ RAM (16GB recommended)

### Quick Setup
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv fraud-detection
source fraud-detection/bin/activate  # On Windows: fraud-detection\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional GPU packages (optional)
pip install cupy-cuda11x  # For CUDA 11.x
