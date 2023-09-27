from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import os


class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(self, mlflow_logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_validation_end(trainer, pl_module)
        run_id = self.mlflow_logger.run_id
        os.system(f'mkdir {"/".join(self.best_model_path.split("/")[:5])}')
        os.system(f'touch {self.best_model_path}')
        self.mlflow_logger.experiment.log_artifact(run_id,
                                                   self.best_model_path)