"""
Payment behavior features for home loan default prediction.
Aggregates payment patterns and behaviors from payment_behavior table.
"""
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.window import Window


def compute_features_fn(input_df, timestamp_column=None, start_date=None, end_date=None):
    """Compute payment behavior features from payment history.

    :param input_df: Input dataframe from payment_behavior table.
    :param timestamp_column: observation_date column.
    :param start_date: Not used (processing latest snapshot).
    :param end_date: Not used (processing latest snapshot).
    :return: Output dataframe containing aggregated payment features per account.
    """

    # Calculate payment ratio (actual / scheduled)
    df_with_ratio = input_df.withColumn(
        "payment_ratio",
        F.when(F.col("scheduled_payment") > 0, F.col("actual_payment") / F.col("scheduled_payment")).otherwise(1.0)
    )

    # Aggregate payment features over last 6 months (months_back <= 6)
    payment_features = df_with_ratio.filter(F.col("months_back") <= 6).groupBy("account_id").agg(
        # Average payment ratio over 6 months
        F.avg("payment_ratio").alias("avg_payment_ratio_6m"),

        # Minimum payment ratio (worst payment month)
        F.min("payment_ratio").alias("min_payment_ratio_6m"),

        # Total additional principal payments
        F.sum("additional_principal").alias("total_additional_principal_6m"),

        # Average offset balance
        F.avg("offset_balance").alias("avg_offset_balance_6m"),

        # Count of months with missed/low payments (payment ratio < 0.9)
        F.sum(F.when(F.col("payment_ratio") < 0.9, 1).otherwise(0)).alias("months_with_missed_payment"),

        # Balance trend: calculate correlation coefficient between months_back and account_balance
        # Positive = increasing balance (bad), Negative = decreasing balance (good)
        F.corr("months_back", "account_balance").alias("balance_trend_6m"),
    )

    # Cast to appropriate types and handle nulls
    payment_features = payment_features.select(
        "account_id",
        F.coalesce(F.col("avg_payment_ratio_6m"), F.lit(1.0)).cast(FloatType()).alias("avg_payment_ratio_6m"),
        F.coalesce(F.col("min_payment_ratio_6m"), F.lit(1.0)).cast(FloatType()).alias("min_payment_ratio_6m"),
        F.coalesce(F.col("total_additional_principal_6m"), F.lit(0.0)).cast(FloatType()).alias("total_additional_principal_6m"),
        F.coalesce(F.col("avg_offset_balance_6m"), F.lit(0.0)).cast(FloatType()).alias("avg_offset_balance_6m"),
        F.coalesce(F.col("months_with_missed_payment"), F.lit(0)).cast(IntegerType()).alias("months_with_missed_payment"),
        F.coalesce(F.col("balance_trend_6m"), F.lit(0.0)).cast(FloatType()).alias("balance_trend_6m"),
    )

    return payment_features
