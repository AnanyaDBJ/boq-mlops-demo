"""
Delinquency features for home loan default prediction.
Aggregates historical delinquency patterns from delinquency_history table.
"""
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window


def compute_features_fn(input_df, timestamp_column=None, start_date=None, end_date=None):
    """Compute delinquency behavior features from delinquency history.

    :param input_df: Input dataframe from delinquency_history table.
    :param timestamp_column: observation_date column.
    :param start_date: Not used (processing latest snapshot).
    :param end_date: Not used (processing latest snapshot).
    :return: Output dataframe containing aggregated delinquency features per account.
    """

    # Get current delinquency (most recent observation)
    window_latest = Window.partitionBy("account_id").orderBy(F.col("observation_date").desc())
    current_delinq = input_df.withColumn("rank", F.row_number().over(window_latest)) \
        .filter(F.col("rank") == 1) \
        .select(
            "account_id",
            F.col("delinquency_bucket").alias("current_delinquency")
        )

    # Aggregate features over different time windows based on months_back
    delinquency_features = input_df.groupBy("account_id").agg(
        # Max delinquency in last 3 months
        F.max(F.when(F.col("months_back") <= 3, F.col("delinquency_bucket")).otherwise(0)).alias("max_delinquency_3m"),

        # Max delinquency in last 6 months
        F.max(F.when(F.col("months_back") <= 6, F.col("delinquency_bucket")).otherwise(0)).alias("max_delinquency_6m"),

        # Max delinquency in last 12 months
        F.max(F.when(F.col("months_back") <= 12, F.col("delinquency_bucket")).otherwise(0)).alias("max_delinquency_12m"),

        # Count of 30+ DPD in last 6 months
        F.sum(F.when((F.col("months_back") <= 6) & (F.col("delinquency_bucket") >= 30), 1).otherwise(0)).alias("count_30dpd_6m"),

        # Count of 60+ DPD in last 6 months
        F.sum(F.when((F.col("months_back") <= 6) & (F.col("delinquency_bucket") >= 60), 1).otherwise(0)).alias("count_60dpd_6m"),

        # Count of 90+ DPD in last 12 months
        F.sum(F.when((F.col("months_back") <= 12) & (F.col("delinquency_bucket") >= 90), 1).otherwise(0)).alias("count_90dpd_12m"),

        # Ever 90+ DPD flag (across all history)
        F.max(F.when(F.col("delinquency_bucket") >= 90, 1).otherwise(0)).alias("ever_90dpd"),
    )

    # Join with current delinquency
    final_features = delinquency_features.join(current_delinq, on="account_id", how="left")

    # Fill nulls and cast to appropriate types
    final_features = final_features.select(
        "account_id",
        F.coalesce(F.col("current_delinquency"), F.lit(0)).cast(IntegerType()).alias("current_delinquency"),
        F.col("max_delinquency_3m").cast(IntegerType()),
        F.col("max_delinquency_6m").cast(IntegerType()),
        F.col("max_delinquency_12m").cast(IntegerType()),
        F.col("count_30dpd_6m").cast(IntegerType()),
        F.col("count_60dpd_6m").cast(IntegerType()),
        F.col("count_90dpd_12m").cast(IntegerType()),
        F.col("ever_90dpd").cast(IntegerType()),
    )

    return final_features
