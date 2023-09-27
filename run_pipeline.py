from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    #Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline('/mnt/c/Decodr/G3/B/CUSTOMER_SATISFACTION_2/data/olist_customers_dataset.csv')