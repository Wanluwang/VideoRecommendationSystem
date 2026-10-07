Core Module Design
1. Data Processing Module(data_validation.py, feature_engineering.py)
Functions:
Data loading and validation
Data quality checking
Feature engineering processing
Feature normalization and encoding
Inputs:
users.csv User basic information
videos.csv Video metadata
interactions_cleaned.csv User interaction data
Outputs:
processed_interactions.csv processed feature data
scalers.pkl normalizers
encoders.pkl encoders
embeddings.pkl embedding vectors
2. Model Training Model(train.py)
Functions:
Multi-model traning and comparison
Hyperparameter automatic optimization
Model performance evaluation
Experiment tracking management
Inputs:
processed_interactions.csv
Training configuration parameters
Outputs:
best_model_*.pkl best model file
feature_columns.plk feature list
mlflow experiment records
3. API service module(api.py)
Functions:
RESTful API interfaces
Request validation and error handling
Caching mechanism
Performance monitoring
Interfaces:
GET /health health check
POST /predict single prediction
POST /predict/batch batch prediction
GET /model/info model information
4. Frontend Display Module(streamlit_app.py)
Functions:
Single prediction interface
Batch prediction interface
Data analytics dashboard
Model information display
Pages:
Single prediction: input parameters to get prediction results
Batch prediction: upload CSV file for batch processing
Analytics dashboard: visual display of system performance
Model info: display model configuration and features