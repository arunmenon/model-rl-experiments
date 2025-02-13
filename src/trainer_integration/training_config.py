# src/trainer_integration/training_config.py
class TrainingConfig:
    def __init__(
        self,
        model_name="gpt2",
        dataset_path="data/example_dataset.csv",
        output_dir="output",
        num_train_steps=1000,
        # more hyperparams...
        remove_unused_columns=False,
        # etc.
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_train_steps = num_train_steps
        self.remove_unused_columns = remove_unused_columns
        # ...
