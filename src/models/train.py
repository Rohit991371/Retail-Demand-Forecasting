from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

from src.evaluation.evaluate import evaluate_predictions, print_metrics
from src.models.baseline import evaluate_baselines


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)

ARTIFACTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
)


def load_model_data():
    """Load the processed feature dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    return df


def time_based_split(df):
    """
    Split data chronologically.

    70% -> training
    15% -> validation
    15% -> test
    """

    unique_dates = sorted(
        df["Date"].unique()
    )

    n_dates = len(unique_dates)

    train_end = int(
        n_dates * 0.70
    )

    validation_end = int(
        n_dates * 0.85
    )

    train_end_date = unique_dates[
        train_end
    ]

    validation_end_date = unique_dates[
        validation_end
    ]

    train_df = df[
        df["Date"] < train_end_date
    ].copy()

    validation_df = df[
        (df["Date"] >= train_end_date)
        & (df["Date"] < validation_end_date)
    ].copy()

    test_df = df[
        df["Date"] >= validation_end_date
    ].copy()

    print("\nTime-based split")
    print("----------------")

    print(
        f"Train:      {train_df['Date'].min().date()} "
        f"→ {train_df['Date'].max().date()} "
        f"({len(train_df):,} rows)"
    )

    print(
        f"Validation: {validation_df['Date'].min().date()} "
        f"→ {validation_df['Date'].max().date()} "
        f"({len(validation_df):,} rows)"
    )

    print(
        f"Test:       {test_df['Date'].min().date()} "
        f"→ {test_df['Date'].max().date()} "
        f"({len(test_df):,} rows)"
    )

    return train_df, validation_df, test_df


def prepare_features(df):
    """Separate target from model input columns."""

    target = "target"

    drop_columns = [
        target,
        "Date",
    ]

    X = df.drop(
        columns=drop_columns
    )

    y = df[target]

    return X, y


def build_preprocessor(X):
    """Create preprocessing for numerical and categorical features."""

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


def train_model(name, model, preprocessor, X_train, y_train):
    """Build and train a complete preprocessing + model pipeline."""

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


def evaluate_model(
    name,
    pipeline,
    X,
    y
):
    """Generate predictions and evaluate the model."""

    predictions = pipeline.predict(X)

    metrics = evaluate_predictions(
        y,
        predictions
    )

    print_metrics(
        name,
        metrics
    )

    return metrics


def main():

    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------

    df = load_model_data()

    print(
        f"Loaded modelling dataset: {df.shape}"
    )

    # --------------------------------------------------------------
    # 2. Time-based split
    # --------------------------------------------------------------

    train_df, validation_df, test_df = (
        time_based_split(df)
    )

    # --------------------------------------------------------------
    # 3. Prepare X/y
    # --------------------------------------------------------------

    X_train, y_train = prepare_features(
        train_df
    )

    X_validation, y_validation = prepare_features(
        validation_df
    )

    X_test, y_test = prepare_features(
        test_df
    )

    # --------------------------------------------------------------
    # 4. Baselines
    # --------------------------------------------------------------

    evaluate_baselines(
        df,
        test_df["Date"].min()
    )

    # --------------------------------------------------------------
    # 5. Preprocessing
    # --------------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------------------
    # 6. Models
    # --------------------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    validation_results = {}

    # --------------------------------------------------------------
    # 7. Train + validation evaluation
    # --------------------------------------------------------------

    for name, model in models.items():

        pipeline = train_model(
            name,
            model,
            preprocessor,
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            f"{name} - Validation",
            pipeline,
            X_validation,
            y_validation,
        )

        trained_models[name] = pipeline
        validation_results[name] = metrics

    # --------------------------------------------------------------
    # 8. Select model using validation MAE
    # --------------------------------------------------------------

    best_model_name = min(
        validation_results,
        key=lambda name:
        validation_results[name]["MAE"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print(
        f"\nSelected model: {best_model_name}"
    )

    # --------------------------------------------------------------
    # 9. Final test evaluation
    # --------------------------------------------------------------

    test_metrics = evaluate_model(
        f"{best_model_name} - Test",
        best_model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------------
    # 10. Save trained model
    # --------------------------------------------------------------

    ARTIFACTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        ARTIFACTS_PATH
        / "demand_forecasting_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    # --------------------------------------------------------------
    # 11. Save test predictions
    # --------------------------------------------------------------

    predictions = best_model.predict(
        X_test
    )

    prediction_output = test_df[
        [
            "Date",
            "Store ID",
            "Product ID",
            "target",
        ]
    ].copy()

    prediction_output[
        "prediction"
    ] = predictions

    prediction_path = (
        ARTIFACTS_PATH
        / "test_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Test predictions saved to: {prediction_path}"
    )

    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )


if __name__ == "__main__":
    main()