# src/trainer_integration/training_config.py
class TrainingConfig:
    def __init__(
        self,
        model_name="gpt2",
        dataset_path="data/example_dataset.csv",
        output_dir="output",
        num_train_steps=1000,
        batch_size=2,
        remove_unused_columns=False,
        # add other hyperparams as needed
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.remove_unused_columns = remove_unused_columns
