"""Backfill the RQSA-MIL experiment history into MLflow.

Registers the paper's model comparison (dense evaluation on the DS1 test set)
as MLflow runs so the model registry reflects the real research lineage, and
registers rqsa_v1 as the production model.

Usage (MLflow server from mlops/docker-compose.yml must be running):
    python mlops/mlflow_backfill.py --tracking-uri http://127.0.0.1:5000
"""

import argparse

import mlflow

# Dense-eval metrics on the full DS1 test set, from the RQSA-MIL paper.
RUNS = [
    {
        "name": "rqsa_v1",
        "params": {"arch": "mobilenet_v2_rqsa_mil", "attention": "iou_conditioned",
                   "epochs_trained": 22, "pseudo_iou": 0.5},
        "metrics": {"recall": 0.8789, "specificity": 0.8552, "auc": 0.9298,
                    "tpr_at_1pct_fpr": 0.4865},
        "register": True,
    },
    {
        "name": "sag_wmil_alpha1_negw15",
        "params": {"arch": "mobilenet_v2_sag_wmil", "attention": "spatial_gate"},
        "metrics": {"recall": 0.8003, "specificity": 0.8671, "auc": 0.9019,
                    "tpr_at_1pct_fpr": 0.3237},
        "register": False,
    },
    {
        "name": "wmil_no_attention",
        "params": {"arch": "mobilenet_v2_wmil", "attention": "none"},
        "metrics": {"recall": 0.6992, "specificity": 0.9247, "auc": 0.8808,
                    "tpr_at_1pct_fpr": 0.2772},
        "register": False,
    },
    {
        "name": "googlenet_bce",
        "params": {"arch": "googlenet", "attention": "none"},
        "metrics": {"recall": 0.4703, "specificity": 0.9830, "auc": 0.8987,
                    "tpr_at_1pct_fpr": 0.3883},
        "register": False,
    },
    {
        "name": "mobilenet_bce",
        "params": {"arch": "mobilenet_v2", "attention": "none"},
        "metrics": {"recall": 0.9996, "specificity": 0.2718, "auc": 0.8761,
                    "tpr_at_1pct_fpr": 0.0},
        "register": False,
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("rqsa-mil-dense-eval")

    for run in RUNS:
        with mlflow.start_run(run_name=run["name"]) as active:
            mlflow.log_params(run["params"])
            mlflow.log_metrics(run["metrics"])
            mlflow.set_tag("dataset", "DS1-test-dense-eval")
            mlflow.set_tag("source", "RQSA-MIL paper")
            if run["register"]:
                mlflow.register_model(f"runs:/{active.info.run_id}", "rqsa-mil")
            print(f"Logged run: {run['name']}")

    print("Backfill complete. rqsa_v1 registered as model 'rqsa-mil'.")


if __name__ == "__main__":
    main()
