"""
Pipeline Synthesis (Prep)

- Builds full cleaning flow: imputation (median/mode/fwd-fill), encoding (one-hot/hash), scaling (RobustScaler)
- Feature engineering: date→components, binning outliers
- Sequential cells with asserts + final "ready dataset" summary
"""

