import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_pipeline import letterbox_image, voc_box_to_yolo
from feature_engineering import build_preprocessor, fit_and_export_pipeline, load_pipeline


def test_letterbox_image_aspect_ratio():
    # Create synthetic rectangular image (100h x 200w, 3 channels)
    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    target_size = (300, 300)

    padded, scale, (pad_w, pad_h) = letterbox_image(img, target_size=target_size, fill_value=0)

    assert padded.shape == (300, 300, 3)
    # Scale should be 300 / 200 = 1.5
    assert scale == 1.5
    # Scaled height = 150, Scaled width = 300. Vertical padding = (300 - 150) // 2 = 75
    assert pad_w == 0
    assert pad_h == 75
    # The padding area should have fill_value 0
    assert np.all(padded[0:70, :, :] == 0)


def test_voc_box_to_yolo():
    # Image size: 1000 x 500 (w x h)
    # VOC box: [100, 50, 300, 250]
    box = (100.0, 50.0, 300.0, 250.0)
    img_w, img_h = 1000, 500

    x_c, y_c, w, h = voc_box_to_yolo(box, img_w, img_h)

    # Box width = 200, Height = 200
    # Center X = 100 + 100 = 200 -> 200 / 1000 = 0.2
    # Center Y = 50 + 100 = 150 -> 150 / 500 = 0.3
    # Norm W = 200 / 1000 = 0.2
    # Norm H = 200 / 500 = 0.4
    assert np.isclose(x_c, 0.2)
    assert np.isclose(y_c, 0.3)
    assert np.isclose(w, 0.2)
    assert np.isclose(h, 0.4)


def test_feature_engineering_pipeline(tmp_path):
    import pandas as pd

    # Mock dataset with missing values
    df_train = pd.DataFrame({
        "age": [25.0, np.nan, 45.0, 35.0, 50.0],
        "income": [50000.0, 60000.0, np.nan, 80000.0, 120000.0],
        "department": ["Sales", "Engineering", "Sales", np.nan, "HR"]
    })

    num_cols = ["age", "income"]
    cat_cols = ["department"]

    preprocessor = build_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    artifact_path = tmp_path / "preprocessor.joblib"

    # Fit and export
    fit_and_export_pipeline(preprocessor, df_train, artifact_path)
    assert artifact_path.exists()

    # Load back
    loaded_pipe = load_pipeline(artifact_path)
    transformed = loaded_pipe.transform(df_train)

    # Output should have no NaNs
    assert not np.isnan(transformed).any()
    # 2 numeric + one-hot columns (at least 3 categories)
    assert transformed.shape[1] >= 4
