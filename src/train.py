from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
import mlflow
import pandas as pd
import lightgbm as lgb
import pandas as pd
import joblib
import os
import optuna
import skops.io as skops_io
from mlflow.tracking import MlflowClient
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MLFLOW_EXPERIMENT_NAME = "video_recommendation_exp_v2"
MLFLOW_REGISTERED_MODEL = "VideoRecommender"
MLFLOW_UI_PORT = 5000
cur_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(cur_dir)
model_dir = os.path.join(root_dir, 'models')
data_dir = os.path.join(root_dir, 'data')

# Initialize or restore the MLflow experiment so every run is recorded under the same experiment.
def setup_mlflow_experiment():
    # Specify the MLflow tracking backend URI; every run is written to this SQLite database.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Create an MLflow client instance to query and restore experiments explicitly.
    tracking_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # Look up an existing experiment by name.
    existing_experiment = tracking_client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    # If the experiment does not exist, create a new one so set_experiment can reuse it.
    if existing_experiment is None:
        # Create the experiment and capture the returned experiment ID.
        experiment_id = tracking_client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        # If the experiment was soft-deleted, restore it before use so every run can append records.
        if existing_experiment.lifecycle_stage == "deleted":
            # Call the restore API to reactivate the experiment.
            tracking_client.restore_experiment(existing_experiment.experiment_id)
            # Print a restore message for the classroom demo narration.
            print(f"Restored experiment: {MLFLOW_EXPERIMENT_NAME}")
        # Reuse the existing experiment ID.
        experiment_id = existing_experiment.experiment_id
    # Switch the active experiment to the target one; all subsequent mlflow.* calls will belong to it.
    mlflow.set_experiment(experiment_id=experiment_id)
    # Return the latest experiment object so the caller can read its name and other metadata.
    return tracking_client.get_experiment(experiment_id)
trusted_types = [
    "sklearn.tree._tree.Tree"
]
def train_baseline_models(X_train, y_train, X_val, y_val):
    lr_model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=5
    )
    gb_model.fit(X_train, y_train)
    models = {
        "LogisticRegression": lr_model,
        "RandomForestClassifier": rf_model,
        "GradientBoostingClassifier": gb_model
    }
    
    for model_name, model in models.items():
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Metric calculation
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        report = classification_report(y_val, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(report)
        with mlflow.start_run(run_name="baseline"):
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(model, model_name, skops_trusted_types=trusted_types)

X_train = pd.read_csv(os.path.join(data_dir, 'train_interactions.csv'))
y_train = X_train['liked']
X_train = X_train.drop(columns='liked')
X_val = pd.read_csv(os.path.join(data_dir, 'val_interactions.csv'))
y_val = X_val['liked']
X_val = X_val.drop(columns='liked')
#train_baseline_models(X_train, y_train, X_val, y_val)

def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(0)
    ]
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print("LightGBM AUC:", auc)
    
#train_lightgbm(X_train, y_train, X_val, y_val)

# Compute the main classification metrics.
def evaluate_classifier(y_true, y_pred, y_probability):
    # Assemble the accuracy metric.
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    # Assemble the precision metric.
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    # Assemble the recall metric.
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    # Assemble the F1 metric.
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    # Assemble the ROC AUC metric.
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    # Return the metrics dictionary.
    return metrics

def objective(trial, X_train, y_train, X_val, y_val):
    # Hyperparameter search space
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
        "lambda_l1": trial.suggest_float(
                "lambda_l1", 0.0, 5.0
        ),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 0.0, 5.0
        ),
        'verbose': -1,
        'random_state': 42,
        'is_unbalance': True
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Evaluation
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return auc

def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    current_experiment = setup_mlflow_experiment()
    # Print the experiment name for classroom narration.
    print(f"Experiment name: {current_experiment.name}")
    # Print the experiment ID so database records are easy to locate.
    print(f"Experiment ID: {current_experiment.experiment_id}")
    #create study object
    study = optuna.create_study(direction='maximize')
    
    #execute optimization
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val
        ), n_trials=50)
    
    #get best parameters
    best_params = study.best_params
    best_auc = study.best_value
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best Parameters: {best_params}")
    final_params = {
        **best_params,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        'is_unbalance': True
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data
    )
    final_model = lgb.train(
        final_params,
        train_data,
        valid_sets = [val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    y_val_proba = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    # ---------- Log the best model with MLflow ----------
    # Record overall training configuration and the final model selection in a parent run, treating it as a complete training session.
    with mlflow.start_run(run_name=f"video_recommendation_lgb") as parent_run:       
        mlflow.log_params(best_params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_val))
    
        metrics = evaluate_classifier(y_val, y_val_pred, y_val_proba)
        # Write each test set metric into MLflow one by one.
        for metric_name, metric_value in metrics.items():
            # Log each metric separately so the UI can plot trend lines.
            mlflow.log_metric(metric_name, metric_value)
            # Also log the cross-validation AUC as a metric so it can be compared with the test set AUC.
        mlflow.log_metric("optuna_best_auc", best_auc)
    
        trusted_types = skops_io.get_untrusted_types(data=skops_io.dumps(final_model))
        # Print a hint when additional trusted types are needed for this model.
        if trusted_types:
            print(f"Using extended skops trusted types: {trusted_types}")
        # Save and register the model once, passing the complete trusted types list up front.
        model_uri = mlflow.sklearn.log_model(sk_model=final_model, name="video_recom_pipeline", registered_model_name=MLFLOW_REGISTERED_MODEL, skops_trusted_types=trusted_types)
        # mlflow 3 returns a ModelInfo object; support the legacy string form for older versions.
        mlflow_model_uri = getattr(model_uri, "model_uri", model_uri)
        # Print the model save URI for classroom display.
        print(f"Model saved to: {mlflow_model_uri}")
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MLFLOW_REGISTERED_MODEL}'"
        )
        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )
        client.set_registered_model_alias(
            name=MLFLOW_REGISTERED_MODEL,
            alias="champion",
            version=latest_version.version
        )
        print(f"Champion → version {latest_version.version}")

optimize_hyperparameters(X_train, y_train, X_val, y_val)
#train_baseline_models(X_train, y_train, X_val, y_val)
#train_lightgbm(X_train, y_train, X_val, y_val)