import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
timestamp = datetime.now()
st.header("🎯 Single Prediction")
#User input
user_id = st.text_input("User ID", value="user_123")
video_id = st.text_input("Video ID", value="video_456")
watch_time = st.slider("Watch Time (seconds)", 0.0, 180.0, 45.0)
timestamp = timestamp.now()
API_URL = "http://api:8000"
def display_result(result):
    #probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['probability'],
        title={'text': "Engagement Probability"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#ff4b4b"},
            'steps': [
                {'range': [0, 0.3], 'color': "#ffebee"},
                {'range': [0.3, 0.7], 'color': "#fff8e1"},
                {'range': [0.7, 1], 'color': "#e8f5e9"} 
            ]
        }
    ))
    st.plotly_chart(fig)
    
if st.button("🚀 Predict"):
    payload = {
        "user_id": user_id,
        "video_id": video_id,
        "watch_time": watch_time,
        "timestamp": timestamp.isoformat()
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=payload
    )
    if response.status_code == 200:
        result = response.json()
        display_result(result)
    else:
        st.error(f"Prediction failed: {response.text}")
        
st.header("📊 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.sample(n=10)
    st.dataframe(df.head(10))
if st.button("🚀 Run Batch Prediction"):
    requests_list = []
    for _, row in df.iterrows():
        requests_list.append({
            "user_id": row['user_id'],
            "video_id": row['video_id'],
            "watch_time": row['watch_time'],
            "timestamp": row['timestamp']
        })
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"requests": requests_list}
    )
    if response.status_code == 200:
        data = response.json()
        results_df = pd.DataFrame(data["results"])
        st.dataframe(results_df)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        fig = px.histogram(
            results_df,
            x='probability',
            title="Prediction Probability Distribution",
            nbins=20
        )
        st.plotly_chart(fig)
    else:
        st.error(f"API error: {response.status_code}")
        st.write(response.text)
    
    

st.header("📈 Analytics Dashboard")  
#Generate sample data
dates = pd.date_range(start=datetime.now()-timedelta(days=7),
                      end=datetime.now(), freq='h')
PROMETHEUS_URL = "http://prometheus:9090"
response = requests.get(
    f"{PROMETHEUS_URL}/api/v1/query_range",
    params={
        "query": "increase(api_requests_total[1h])",
        "start": (datetime.now() - timedelta(days=7)).timestamp(),
        "end": datetime.now().timestamp(),
        "step": 3600,
    }
)
def query_prometheus_instant(query):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()
    result = data["data"]["result"]

    if not result:
        return None

    return float(result[0]["value"][1])
data = response.json()
result = data["data"]["result"]
if result:
    values = result[0]["values"]
    sample_data = pd.DataFrame(
        values,
        columns=["timestamp", "requests_per_hour"]
        )
    sample_data["timestamp"] = pd.to_datetime(
        sample_data["timestamp"],
        unit="s"
    )
    sample_data["requests_per_hour"] = pd.to_numeric(
        sample_data["requests_per_hour"]
    )
    fig = px.line(sample_data, x='timestamp', y='requests_per_hour', markers=True,
              title="Request Volume Trend")
    st.plotly_chart(fig)
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": "increase(api_requests_total[7d])"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    result = data["data"]["result"]
    
    total_requests = float(result[0]["value"][1]) if result else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", f"{total_requests:.0f}" if total_requests is not None else "N/A")
    avg_prediction_time = query_prometheus_instant("""
        rate(prediction_latency_seconds_sum[1h])
        /
        rate(prediction_latency_seconds_count[1h])
        * 1000
    """)
    col2.metric("Avg Prediction Time", f"{avg_prediction_time:.2f}" if avg_prediction_time is not None else "N/A")
    avg_probability = query_prometheus_instant("""
        rate(prediction_probability_sum[1h])
        /
        rate(prediction_probability_count[1h])
    """)
    col3.metric("Avg Probability", f"{avg_probability:.3f}" if avg_probability is not None else "N/A")

#Get feature importance from model
# importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': feature_importance
# }).sort_values('importance', ascending=False).head(10)
# fig = px.bar(importance_df, x='importance', y='feature',
#              orientation='h', title="Top 10 Feature Importance")
# st.plotly_chart(fig)