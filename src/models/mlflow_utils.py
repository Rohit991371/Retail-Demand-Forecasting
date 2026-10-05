import mlflow
import mlflow.sklearn


EXPERIMENT_NAME = "retail-demand-forecasting"


def setup_mlflow():
    """Configure the MLflow experiment."""

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def log_model_run(
    model_name,
    model,
    params,
    validation_metrics,
    test_metrics=None,
):
    """
    Log one model experiment to MLflow.

    Test metrics are optional because the test set
    should only be evaluated after model selection.
    """

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------------
        # Tags
        # ----------------------------------------------------------

        mlflow.set_tag(
            "model_type",
            model_name
        )

        mlflow.set_tag(
            "problem",
            "next_day_demand_forecasting"
        )

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------

        mlflow.log_params(params)

        # ----------------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------------

        mlflow.log_metrics({
            f"validation_{key.lower()}": value
            for key, value
            in validation_metrics.items()
        })

        # ----------------------------------------------------------
        # Test metrics
        #
        # Only logged when explicitly provided.
        # ----------------------------------------------------------

        if test_metrics is not None:

            mlflow.log_metrics({
                f"test_{key.lower()}": value
                for key, value in test_metrics.items()
            })

        # ----------------------------------------------------------
        # Model artifact
        # ----------------------------------------------------------

        # mlflow.sklearn.log_model(
        #     model,
        #     name="model"
        # )

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
            # skops_trusted_types=[
            #     "numpy.dtype",
            #     # "sklearn.tree._tree.Tree",
            # ],
        )

        # mlflow.sklearn.log_model(
        #     sk_model=model,  # Your trained XGBoost model or Pipeline
        #     artifact_path="model",
        #     skops_trusted_types=[
        #         "xgboost.core.Booster",
        #         "xgboost.sklearn.XGBRegressor"
        #     ]
        # )

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

    return run_id
