# Databricks notebook source
##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline for Home Loan Default Prediction
# using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``boq_mlops_project/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data table.
# * target_table_path (required)    - Path to the target table containing default labels.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog.
#
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Unity Catalog table containing the account data.
dbutils.widgets.text(
    "training_data_path",
    "ananyaroy.boq_mlops.account_master",
    label="Path to the training data",
)

# Path to the target table containing default labels
dbutils.widgets.text(
    "target_table_path",
    "ananyaroy.boq_mlops.target",
    label="Path to the target table",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-boq-loan-mlops-experiment",
    label="MLflow experiment name",
)


# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "ananyaroy.boq_mlops.boq-loan-default-model", label="Full (Three-Level) Model Name"
)

# Account features table name
dbutils.widgets.text(
    "account_features_table",
    "ananyaroy.boq_mlops.account_features",
    label="Account Features Table",
)

# Delinquency features table name
dbutils.widgets.text(
    "delinquency_features_table",
    "ananyaroy.boq_mlops.delinquency_features",
    label="Delinquency Features Table",
)

# Payment features table name
dbutils.widgets.text(
    "payment_features_table",
    "ananyaroy.boq_mlops.payment_features",
    label="Payment Features Table",
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
training_data_path = dbutils.widgets.get("training_data_path")
target_table_path = dbutils.widgets.get("target_table_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Load training data and target
# Load only account_id from account master (features will come from Feature Store)
account_data = spark.table(training_data_path).select("account_id")

# Load target table (contains default_12m labels)
target_data = spark.table(target_table_path)

# Join account data with target
# The base dataframe should only contain account_id and target - all features come from Feature Store
training_data = account_data.join(target_data, on="account_id", how="inner")

print(f"Training data shape: {training_data.count()} rows")
training_data.display()

# COMMAND ----------

# DBTITLE 1, Helper functions
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

# DBTITLE 1, Create FeatureLookups
from databricks.feature_engineering import FeatureLookup
import mlflow

account_features_table = dbutils.widgets.get("account_features_table")
delinquency_features_table = dbutils.widgets.get("delinquency_features_table")
payment_features_table = dbutils.widgets.get("payment_features_table")

# Account-level features (static characteristics)
account_feature_lookups = [
    FeatureLookup(
        table_name=account_features_table,
        lookup_key=["account_id"],
    ),
]

# Delinquency history features
delinquency_feature_lookups = [
    FeatureLookup(
        table_name=delinquency_features_table,
        lookup_key=["account_id"],
    ),
]

# Payment behavior features
payment_feature_lookups = [
    FeatureLookup(
        table_name=payment_features_table,
        lookup_key=["account_id"],
    ),
]

# COMMAND ----------

# DBTITLE 1, Create Training Dataset

from databricks.feature_engineering import FeatureEngineeringClient

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# Columns to exclude from training (IDs, dates, etc.)
exclude_columns = ["account_open_date", "bank"]

fe = FeatureEngineeringClient()

# Create the training set that includes the raw input data merged with corresponding features from all feature tables
training_set = fe.create_training_set(
    df=training_data,
    feature_lookups=account_feature_lookups + delinquency_feature_lookups + payment_feature_lookups,
    label="default_12m",  # Binary target: 0 = no default, 1 = default
    exclude_columns=exclude_columns,
)


# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe with features from Feature Store
print(f"Training dataframe columns: {training_df.columns}")
print(f"Training dataframe shape: {training_df.count()} rows, {len(training_df.columns)} columns")
training_df.display()

# COMMAND ----------

# Check class distribution
training_df.groupBy("default_12m").count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM binary classification model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureEngineeringClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

# DBTITLE 1, Train model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow.lightgbm
from mlflow.tracking import MlflowClient


features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

# Split data into train and test sets
train, test = train_test_split(data, random_state=123, stratify=data["default_12m"])
X_train = train.drop(["default_12m"], axis=1)
X_test = test.drop(["default_12m"], axis=1)
y_train = train.default_12m
y_test = test.default_12m

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Default rate in train: {y_train.mean():.4f}")
print(f"Default rate in test: {y_test.mean():.4f}")

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

# Binary classification parameters
param = {
    "num_leaves": 32,
    "objective": "binary",  # Binary classification
    "metric": "auc",  # ROC-AUC for evaluation
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1
}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(
    param,
    train_lgb_dataset,
    num_rounds,
    valid_sets=[test_lgb_dataset],
    valid_names=['test']
)

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Log the trained model with MLflow and package it with feature lookup information.
fe.log_model(
    model=model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,
)


# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
