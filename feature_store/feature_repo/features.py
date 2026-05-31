from __future__ import annotations

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int32, Int64
from feast.value_type import ValueType

# Define user entity
user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    join_keys=["user_id"],
    description="User ID for identifying customer transactions",
)

# Define data sources (paths relative to feature_store.yaml location)
user_features_source = FileSource(
    path="../../data/processed/user_features.parquet",
    timestamp_field="event_timestamp",
)

transaction_features_source = FileSource(
    path="../../data/processed/transactions.parquet",
    timestamp_field="event_timestamp",
)

# Feature view for historical user-level aggregates
user_features_view = FeatureView(
    name="user_features",
    entities=[user_entity],
    schema=[
        Field(name="user_txn_count", dtype=Int64),
        Field(name="user_avg_amount", dtype=Float32),
        Field(name="user_std_amount", dtype=Float32),
    ],
    source=user_features_source,
    ttl=None, # No TTL for static aggregates
)

# Feature view for transaction-level features (PCA components + engineered features)
transaction_features_schema = [
    Field(name="transaction_hour", dtype=Int32),
    Field(name="log_amount", dtype=Float32),
    Field(name="amount_to_mean_ratio", dtype=Float32),
    Field(name="amount_zscore_user", dtype=Float32),
    Field(name="Amount", dtype=Float32),
] + [
    Field(name=f"V{i}", dtype=Float32) for i in range(1, 29)
]

transaction_features_view = FeatureView(
    name="transaction_features",
    entities=[user_entity],
    schema=transaction_features_schema,
    source=transaction_features_source,
    ttl=None, # Keep TTL high or None for historical lookups
)
