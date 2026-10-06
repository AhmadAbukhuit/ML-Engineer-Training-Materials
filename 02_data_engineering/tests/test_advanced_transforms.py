import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advanced_transforms import OutOfFoldTargetEncoder, YeoJohnsonTransformer


def test_out_of_fold_target_encoder():
    df = pd.DataFrame({
        "city": ["New York", "London", "Paris", "New York", "London", "Paris", "Tokyo"],
        "price": [100.0, 80.0, 90.0, 110.0, 85.0, 95.0, 120.0]
    })
    y = np.array([1, 0, 0, 1, 0, 1, 1])

    encoder = OutOfFoldTargetEncoder(categorical_cols=["city"], m_smoothing=5.0, n_splits=3)
    df_encoded = encoder.fit_transform(df, y)

    assert "city_encoded" in df_encoded.columns
    # Must have no NaNs
    assert not df_encoded["city_encoded"].isna().any()

    # Test transform on unseen data
    df_test = pd.DataFrame({"city": ["New York", "UnknownCity"]})
    df_test_encoded = encoder.transform(df_test)
    assert not df_test_encoded["city_encoded"].isna().any()


def test_yeo_johnson_transformer():
    # Heavily skewed exponential data with positive, zero, and negative values
    skewed_data = np.array([-5.0, 0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0]).reshape(-1, 1)

    transformer = YeoJohnsonTransformer()
    transformed = transformer.fit_transform(skewed_data)

    assert transformed.shape == skewed_data.shape
    # Check transformed data is normalized (mean ~ 0, std ~ 1)
    assert abs(np.mean(transformed)) < 1e-5
    assert abs(np.std(transformed) - 1.0) < 1e-5
