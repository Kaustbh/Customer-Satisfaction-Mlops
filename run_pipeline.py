from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/home/kaustubh/mlops_projects/Customer_Satifaction/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-ui "file:/home/kaustubh/.config/zenml/local_stores/ce1672d2-a7be-4023-ad6a-eebca24b856a/mlruns"