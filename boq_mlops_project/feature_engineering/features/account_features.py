"""
Account-level features for home loan default prediction.
Extracts static account characteristics from the account_master table.
"""
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType


def compute_features_fn(input_df, timestamp_column=None, start_date=None, end_date=None):
    """Compute account-level features from account master data.

    :param input_df: Input dataframe from account_master table.
    :param timestamp_column: Not used for account features (static data).
    :param start_date: Not used for account features.
    :param end_date: Not used for account features.
    :return: Output dataframe containing account-level features.
    """

    account_features = input_df.select(
        F.col("account_id"),

        # Loan age and term
        F.col("months_on_book").cast(IntegerType()),
        F.col("loan_term_months").cast(IntegerType()),

        # Account flags
        F.col("is_investor").cast(IntegerType()),
        F.col("is_interest_only").cast(IntegerType()),
        F.col("has_multiple_loans").cast(IntegerType()),

        # Loan amounts and utilization
        F.col("original_loan_amount").cast(FloatType()),
        F.col("current_balance").cast(FloatType()),
        (F.col("current_balance") / F.col("original_loan_amount")).alias("loan_utilization").cast(FloatType()),

        # Interest rate
        F.col("interest_rate").cast(FloatType()),

        # Bank (encode as binary: BOQ=0, ME=1)
        F.when(F.col("bank") == "BOQ", 0).otherwise(1).alias("bank_encoded").cast(IntegerType()),
    )

    return account_features
