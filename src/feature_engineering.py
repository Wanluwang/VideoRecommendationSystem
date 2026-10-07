import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

#Get path
current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
data_dir = os.path.join(root_dir, 'data')
interactions_path = os.path.join(data_dir, 'interactions_cleaned.csv')
users_path = os.path.join(data_dir, 'users.csv')
videos_path = os.path.join(data_dir, 'videos.csv')
#model dir
models_dir = os.path.join(root_dir, 'models')
#Construct dataframe
interactions_df = pd.read_csv(interactions_path)
videos_df = pd.read_csv(videos_path)
users_df = pd.read_csv(users_path)

numeric_cols = [    'watch_time',
                    'user_watch_time_mean', 
                    'user_watch_time_std', 
                    'user_total_view_count',
                    'user_watch_time_sum',
                    'user_watch_time_max',
                    'user_liked_sum',
                    'user_shared_sum',
                    'user_recency_days_min',
                    'user_user_avg_session_duration_seconds',
                    'user_age',
                    'user_account_age',
                    'video_average_watch_time',
                    'video_watch_time_std',
                    'video_watch_time_count',
                    'video_total_watch_time',
                    'video_liked_sum',
                    'video_shared_sum',
                    'video_duration_sec',
                    'video_view_count',
                    'video_video_age_days']

def get_day_period(hour):
    if 0 <= hour < 6: return 'night'
    elif 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'afternoon'
    else: return 'evening'
    
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['day_period'] = df['hour_of_day'].apply(get_day_period)
    max_date = datetime.now()
    df['recency_days'] = (max_date - df['timestamp']).dt.days
    #cyclic features (trigonometric encoding)
    df['hour_sin'] = np.sin(2*np.pi*df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour_of_day']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    return df
    
def create_sessionization_features(interactions_df):
    df = interactions_df.sort_values(['user_id', 'timestamp'])
    df['time_gap'] = df.groupby('user_id')['timestamp'].diff()
    df['new_session'] = (df['time_gap'] > pd.Timedelta(minutes=30))
    df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
    #session statistics
    session_stats = df.groupby(['user_id', 'session_id']).agg({
        'watch_time': ['count', 'sum'],
        'timestamp': ['min', 'max']
    })
    session_stats['session_duration_seconds'] = (
        (session_stats['timestamp']['max'] - session_stats['timestamp']['min']).dt.total_seconds()
    )
    avg_session_duration = session_stats.groupby(['user_id'])['session_duration_seconds'].mean().reset_index(name='user_avg_session_duration_seconds')
    return avg_session_duration

def create_user_features(interactions_df, users_df, videos_df):
    #user behavior
    user_stats = interactions_df.groupby('user_id').agg({
        'watch_time': ['mean', 'std', 'count', 'sum', 'max'],
        'liked': ['mean', 'sum'],
        'shared': ['mean', 'sum'],
        'recency_days': ['min']
    })
    user_stats[('watch_time', 'std')] = user_stats[('watch_time', 'std')].fillna(0)
    user_stats.columns = ['_'.join(col) for col in user_stats.columns]
    user_stats.rename(columns={
        'watch_time_count': 'total_view_count',
        'liked_mean': 'like_rate',
        'shared_mean': 'share_rate'
    }, inplace=True)
    session_stats = create_sessionization_features(interactions_df)
    user_stats['engagement_rate'] = (user_stats['liked_sum']+user_stats['shared_sum'])/user_stats['total_view_count']
    user_stats = user_stats.merge(session_stats, how='left', on='user_id')
    #User account features
    max_date = datetime.now()
    users_df['join_date'] = pd.to_datetime(users_df['join_date'])
    users_df['account_age'] = (max_date-users_df['join_date']).dt.days
    user_stats = user_stats.merge(users_df, how='right', on='user_id')
    #User category diversity
    user_category_diversity = interactions_df.merge(
        videos_df[['video_id', 'category']], on='video_id'
    ).groupby('user_id')['category'].nunique()
    user_stats = user_stats.merge(user_category_diversity, how='left', on='user_id')
    user_stats.rename(columns={
        col: f"user_{col}" for col in user_stats.columns if col != 'user_id'
    }, inplace=True)
    user_stats = user_stats.fillna(user_stats.mode().iloc[0])
    return user_stats

def create_video_features(interactions_df, videos_df):
    video_stats = interactions_df.groupby('video_id').agg({
        'watch_time': ['mean', 'std', 'count', 'sum'],
        'liked': ['mean', 'sum'],
        'shared': ['mean', 'sum'],
    })
    video_stats[('watch_time', 'std')] = video_stats[('watch_time', 'std')].fillna(0)
    video_stats.columns = ['_'.join(col) for col in video_stats.columns]
    video_stats.rename(columns={
        'watch_time_sum': 'total_watch_time',
        'watch_time_mean': 'average_watch_time',
        'liked_mean': 'liked_rate',
        'shared_mean': 'shared_rate',
    }, inplace=True)
    video_stats['engagement_rate'] = video_stats['liked_rate']+video_stats['shared_rate']
    max_date = datetime.now()
    videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
    videos_df['video_age_days'] = (max_date-videos_df['upload_date']).dt.days
    video_stats = video_stats.merge(videos_df, on='video_id', how='right')
    video_stats = video_stats.fillna(video_stats.mode().iloc[0])
    video_stats.rename(columns={
            col: f"video_{col}" for col in video_stats.columns if col != 'video_id'
        }, inplace=True)
    return video_stats

from sklearn.decomposition import TruncatedSVD
def create_embeddings(interactions_df):
    interaction_matrix = pd.pivot_table(
        interactions_df,
        values='liked',
        index='user_id',
        columns='video_id',
        fill_value=0
    )
    svd = TruncatedSVD(n_components=50,random_state=42)
    user_embeddings = svd.fit_transform(interaction_matrix)
    video_embeddings = svd.components_.T
    user_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(interaction_matrix.index)
        }
    video_to_idx = {
        video_id: idx
        for idx, video_id in enumerate(interaction_matrix.columns)
    }
    return user_embeddings, video_embeddings, user_to_idx, video_to_idx
leakage_features = [
    "video_liked_rate",
    "video_liked_sum",
    "video_engagement_rate",
    "user_like_rate",
    "user_liked_sum",
    "user_engagement_rate",
]
from sklearn.preprocessing import MinMaxScaler
def create_interaction_features(interactions_df, videos_df, users_df, user_embeddings, video_embeddings, user_to_idx, video_to_idx, encoder, scaler):
    df = interactions_df.merge(videos_df, how='left', on='video_id')
    df['completion_rate'] = np.clip(df['watch_time']/df['video_duration_sec'], 0, 1)
    df['hour_weekend_interaction'] = df['hour_of_day'] * df['is_weekend']
    df = df.merge(users_df, how='left', on= 'user_id')
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df['category_match'] = (
        df['user_preferred_category'] == df['video_category']
    ).astype(int)
    svd_features = []
    for _, row in interactions_df.iterrows():
        user_vector = user_embeddings[
            user_to_idx[row['user_id']]
        ] if row['user_id'] in user_to_idx else np.zeros(user_embeddings.shape[1])
        video_vector = video_embeddings[
            video_to_idx[row['video_id']]
        ] if row['video_id'] in video_to_idx else np.zeros(video_embeddings.shape[1])
        combined = np.concatenate([
            user_vector, video_vector
        ])
        svd_features.append(combined)
    svd_columns = [
        f'user_svd_{i+1}' for i in range(user_embeddings.shape[1])
        ] + [
        f'video_svd_{i+1}' for i in range(video_embeddings.shape[1])
    ]
    svd_df = pd.DataFrame(
        svd_features, columns=svd_columns, index=df.index
    )
    categorical_cols = ['user_gender', 'video_category', 'user_preferred_category', 'day_period']
    encoded = encoder.transform(df[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(
        encoded, columns=encoded_columns, index=df.index
    )
    df.drop(columns=
            ['user_id', 'video_id', 'timestamp', 'user_join_date', 'video_upload_date', 'video_creator_id'] + categorical_cols
            ,inplace=True)
    df.drop(columns=leakage_features, inplace=True)
    if 'liked' in df.columns:
        df.drop(columns='liked', inplace=True)
    if 'shared' in df.columns:
        df.drop(columns='shared', inplace=True)
    X = pd.concat([df, svd_df, encoded_df], axis=1)
    return X

from sklearn.preprocessing import OneHotEncoder
def encode_feature(interactions_df, user_features, video_features):
    df = pd.merge(interactions_df, user_features, on='user_id', how='left')
    df = pd.merge(df, video_features, on='video_id', how='left')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_cols = ['user_gender', 'video_category', 'user_preferred_category', 'day_period']
    encoder.fit_transform(df[categorical_cols])
    return encoder
def scale_feature(df,videos_df, users_df):
    df = df.merge(videos_df, on='video_id', how='left')
    df = df.merge(users_df, on='user_id', how='left')
    scaler = MinMaxScaler()
    scaler.fit_transform(df[numeric_cols])
    return scaler

from sklearn.model_selection import train_test_split
# Sort by time
interactions_df = interactions_df.sort_values('timestamp')

# Time split: first 64% training, 16% validation, 20% test
train_size = int(len(interactions_df) * 0.64)
val_size = int(len(interactions_df) * 0.16)

train_df = interactions_df[:train_size]
y_train = interactions_df['liked'][:train_size]
val_df = interactions_df[train_size:train_size+val_size]
y_val = interactions_df['liked'][train_size:train_size+val_size]
test_df = interactions_df[train_size+val_size:]
y_test = interactions_df['liked'][train_size+val_size:]

train_df = create_temporal_features(train_df)
user_features = create_user_features(train_df, users_df, videos_df)
video_features = create_video_features(train_df, videos_df)
user_embeddings, video_embeddings, user_to_idx, video_to_idx = create_embeddings(interactions_df)
encoder = encode_feature(train_df, user_features, video_features)
scaler = scale_feature(train_df, video_features, user_features)
X = create_interaction_features(train_df, video_features, user_features, user_embeddings, video_embeddings, user_to_idx, video_to_idx, encoder, scaler)

val_df = create_temporal_features(val_df)
X_val = create_interaction_features(val_df, video_features, user_features, user_embeddings, video_embeddings, user_to_idx, video_to_idx, encoder, scaler)


test_df = create_temporal_features(test_df)
X_test = create_interaction_features(test_df, video_features, user_features, user_embeddings, video_embeddings, user_to_idx, video_to_idx, encoder, scaler)

import joblib
joblib.dump(encoder, os.path.join(models_dir, 'encoders.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scalers.pkl'))
embeddings = {
    'user_embeddings': user_embeddings,
    'video_embeddings': video_embeddings,
    'user_to_idx': user_to_idx,
    'video_to_idx': video_to_idx
}
joblib.dump(embeddings, os.path.join(models_dir, 'embeddings.pkl'))

train_interactions = pd.concat([X.reset_index(drop=True), y_train.reset_index(drop=True).rename('liked')], axis=1)
val_interactions = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True).rename('liked')], axis=1)
test_interactions = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename('liked')], axis=1)
train_interactions.to_csv(os.path.join(data_dir, 'train_interactions.csv'), index=False)
test_interactions.to_csv(os.path.join(data_dir, 'test_interactions.csv'), index=False)
val_interactions.to_csv(os.path.join(data_dir, 'val_interactions.csv'), index=False)
user_features.to_csv(os.path.join(data_dir, 'user_features.csv'), index=False)
video_features.to_csv(os.path.join(data_dir, 'video_features.csv'), index=False)