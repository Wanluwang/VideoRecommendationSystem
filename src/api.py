from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import time
from typing import Optional, List
import logging
import redis
import json
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from contextlib import asynccontextmanager
from .feature_engineering import create_temporal_features, create_interaction_features
import joblib
import os
import pandas as pd

cur_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(cur_dir)
model_dir = os.path.join(root_dir, 'models')
data_dir = os.path.join(root_dir, 'data')

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
MODEL_URI = "models:/VideoRecommender@champion"
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    app.state.redis_client = redis.from_url(redis_url, decode_responses=True)
    # Verify Redis is actually available
    app.state.redis_client.ping()
    print("Redis connected!")
    
    # Then create RateLimiter using Redis
    app.state.rate_limiter = RateLimiter(
        redis_client=app.state.redis_client
    )
    
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        "VideoRecommender",
        "champion"
    )

    app.state.model_version = model_version.version
    
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    print("Champion model loaded successfully")
    #app.state.model = None
    
    # Preprocessing artifacts
    app.state.embeddings = joblib.load(os.path.join(model_dir, "embeddings.pkl"))
    app.state.encoder = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    app.state.scaler = joblib.load(os.path.join(model_dir, "scalers.pkl"))
    
    app.state.user_features = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    app.state.video_features = pd.read_csv(os.path.join(data_dir, "video_features.csv"))
    
    yield
    app.state.redis_client.close()
    app.state.model = None
    
app = FastAPI(
    title="YouTube Shorts Recommendation API",
    version="1.0.0",
    description="Production-ready ML recommendation service",
    lifespan=lifespan
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
#Automatically add Prometheus metrics
Instrumentator().instrument(app).expose(app)
#Access metrics: http://localhost:8000/metrics
#Define metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'probability']
)
request_counter = Counter(
    "api_requests_total",
    "Total API requests"
)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)
model_score_gauge = Gauge(
    'model_probability',
    'Last prediction probability'
)
prediction_probability = Histogram(
    "prediction_probability",
    "Prediction probability"
)

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Health check interface
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    
@app.get("/health", response_model=HealthResponse)
async def health():
    startup_time = time.time()
    return HealthResponse(
        status="healthy" if app.state.model else "degraded",
        timestamp=datetime.now(),
        model_loaded=app.state.model is not None,
        redis_connected=app.state.redis_client is not None,
        uptime_seconds=time.time() - startup_time
    )
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'timestamp': datetime.now().isoformat(),
            'path': str(request.url)
        }
    )
#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  
def log_prediction_metrics(user_id, video_id, probability, response_time):
    """Log prediction metrics"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'video_id': video_id,
        'probability': probability,
        'response_time_ms': response_time,
        'event_type': 'prediction'
    }  
    logger.info(f"PREDICTION_METRICS: {json.dumps(metrics)}")
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }
    )

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID (e.g., user_123)", example='user_123')
    video_id: str = Field(..., description="Video ID (e.g., video_456)", example='video_456')
    watch_time: float = Field(..., ge=0, le=300, description="Watch time in seconds", example=45.0)
    timestamp: datetime = Field(..., description="Timestamp of the interaction", example="2024-01-20 18:00:25")
    
    @field_validator('user_id')
    def validate_user_id(cls, value):
        if not value.startswith('user_'):
            raise ValueError('Video ID must start with user_')
        return value
    
    @field_validator('video_id')
    def validate_video_id(cls, v):
        if not v.startswith('video_'):
            raise ValueError('Video ID must start with video_')
        return v
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "video_id": "video_456",
                "watch_time": 45.0,
                "timestamp": "2024-01-20 18:00:25"
            }
        }
class RateLimiter:
    def __init__(self, redis_client, max_requests=100, window_seconds=60):
        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    def is_allowed(self, client_id: str) -> bool:
        pipe = self.redis_client.pipeline()
        now = time.time()
        window_start = now - self.window_seconds
        pipe.zremrangebyscore(f"rate_limit:{client_id}", 0, window_start)
        pipe.zcard(f"rate_limit:{client_id}")
        pipe.zadd(f"rate_limit:{client_id}", {str(now): now})
        pipe.expire(f"rate_limit:{client_id}", self.window_seconds)
        results = pipe.execute()
        return results[1] < self.max_requests
def check_rate_limit(request: Request):
    client_id = request.headers.get('X-Forwarded-For', request.client.host)
    if not app.state.rate_limiter.is_allowed(client_id):
        raise HTTPException(429, "Rate limit exceeded")
    
async def log_prediction(user_id: str, video_id: str, probability:float):
    """Background task: log prediction"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'video_id': video_id,
        'probability': probability
    }
    app.state.redis_client.lpush('prediction_logs', json.dumps(metrics))
class PredictionResponse(BaseModel):
    user_id: str
    video_id: str
    probability: float
    model_version: str
    response_time_ms: float
    timestamp: datetime
def preprocess_request(request: PredictionRequest):
    user_id = request.user_id
    video_id = request.video_id
    watch_time = request.watch_time
    timestamp = request.timestamp
    
    df = pd.DataFrame([{
        "user_id": user_id,
        "video_id": video_id,
        "watch_time": watch_time,
        "timestamp": timestamp
    }])
    df = create_temporal_features(df)
    return create_interaction_features(
        df, 
        app.state.video_features, 
        app.state.user_features,
        app.state.embeddings['user_embeddings'],
        app.state.embeddings['video_embeddings'],
        app.state.embeddings['user_to_idx'],
        app.state.embeddings['video_to_idx'],
        app.state.encoder,
        app.state.scaler)
    
    
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit)])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    #Generate cache key
    cache_key = f"pred:{request.user_id}:{request.video_id}:{request.watch_time}"
    #Check cache
    cached = app.state.redis_client.get(cache_key)
    if cached:
        print(f"Find result for {cache_key} in Cache.")
        return json.loads(cached)
    
    start_time = time.time()
    
    #Feature preprocessing
    features = preprocess_request(request)
    
    #model prediction
    with prediction_latency.time():
        probability = app.state.model.predict(features)[0]
    response_time = (time.time() - start_time) * 1000
    print("Predicted result using model.")
    response = PredictionResponse(
        user_id=request.user_id,
        video_id=request.video_id,
        probability=probability,
        model_version=app.state.model_version,
        response_time_ms=response_time,
        timestamp=datetime.now()
    )
    prediction_counter.labels(
        model_version=app.state.model_version,
        probability=probability
    ).inc()
    request_counter.inc()
    prediction_probability.observe(probability)
    #model_score_gauge.set(response['probability'])
    #store in cache(5 minutes expiration)
    app.state.redis_client.setex(cache_key, 300, json.dumps(response.model_dump(), default=str))
    background_tasks.add_task(
        log_prediction,
        request.user_id,
        request.video_id,
        probability
    )
    return response
    
class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    results = []
    for req in request.requests:
        features = preprocess_request(req)
        with prediction_latency.time():
            probability = app.state.model.predict(features)[0]
        results.append({
            'user_id': req.user_id,
            'video_id': req.video_id,
            'probability': probability,
        })
        request_counter.inc()
        prediction_probability.observe(probability)
    return {
        'results': results,
        'batch_size': len(request.requests),
        'timestamp': datetime.now()
    }
    
# @app.get('/model/info')
# async def model_info():
#     return {}
#     return {
#         'model_type': type(model).__name__,
#         'feature_count': len(feature_columns),
#         'features': feature_columns[:10],
#         'status': 'loaded' if model else 'not loaded'
#     }