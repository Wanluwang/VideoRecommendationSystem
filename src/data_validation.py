import pandas as pd
import os
#Get directory of data
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir,'data')
#Output file
output_file = os.path.join(current_dir,'data_validation_report.txt')
f = open(output_file, "w")
#Read data
users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
videos_df = pd.read_csv(os.path.join(data_dir, 'videos.csv'))
interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions_cleaned.csv'))
#Convert data types
users_df['join_date'] = pd.to_datetime(users_df['join_date'])
videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
interactions_df['liked'] = interactions_df['liked'].astype('bool')
interactions_df['shared'] = interactions_df['shared'].astype('bool')
users_df.to_csv(os.path.join(data_dir, 'processed_users.csv'))
videos_df.to_csv(os.path.join(data_dir, 'processed_videos.csv'))
interactions_df.to_csv(os.path.join(data_dir, 'processed_interactions.csv'))
#Check missing values: `df.isnull().sum()`
def check_missing_values(df, data_name):
    missing_values = df.isnull().sum()
    columns_with_missing_values = []
    for i in range(len(missing_values)):
        index_name = missing_values.index[i]
        if missing_values.iloc[i] != 0:
            columns_with_missing_values.append(index_name)
    if len(columns_with_missing_values) == 0:
        f.write(f"All clear for {data_name}: There are no columns with missing values!\n")
    else:
        f.write(f"Notice {data_name} these columns {columns_with_missing_values} with missing values!\n")
    return columns_with_missing_values
check_missing_values(users_df, 'users')
check_missing_values(videos_df, 'videos')
check_missing_values(interactions_df, 'interactions')

#Check duplicate values: `df.duplicated().sum()`
def check_duplicate_values(df, data_name):
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows == 0:
        f.write(f"There are no duplicated rows for {data_name}\n")
    else:
        f.write(f"There are {duplicate_rows} duplicated rows for {data_name}.\n")
    return duplicate_rows
check_duplicate_values(users_df, 'users')
check_duplicate_values(videos_df, 'videos')
check_duplicate_values(interactions_df, 'interactions')

#Count records in each table
def count_records(df, data_name):
    records_count = df.shape[0]
    f.write(f"There are {records_count} in {data_name}.\n")
    return records_count
count_records(users_df, 'users')
count_records(videos_df, 'videos')
count_records(interactions_df, 'interactions')

#Validate foreign key constraints (whether user_id and video_id in interactions exist)
invalid_users = ~interactions_df['user_id'].isin(users_df['user_id'])
invalid_videos = ~interactions_df['video_id'].isin(videos_df['video_id'])
f.write(f"invalid users: {invalid_users.sum()}\n")
f.write(f"invalid videos: {invalid_videos.sum()}\n")

user_age_distribution = users_df['age'].describe()
video_category_distribution = videos_df['category'].describe()
interaction_time_distribution = interactions_df['watch_time'].describe()
like_rate = interactions_df['liked'].mean()
share_rate = interactions_df['shared'].mean()
f.write(f"user age distribution: {user_age_distribution}\n")
f.write(f"video_category_distribution: {video_category_distribution}\n")
f.write(f"interaction_time_distribution: {interaction_time_distribution}\n")
f.write(f"like_rate: {like_rate}\n")
f.write(f"share_rate: {share_rate}\n")
f.close()