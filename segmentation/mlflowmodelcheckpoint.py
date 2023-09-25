import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint
import os


class MLflowModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 monitor,
                 mode,
                 dirpath=None,
                 filename=None,
                 save_top_k=None,
                 verbose=True,
                 save_last=False):
        super().__init__(monitor=monitor,
                         mode=mode,
                         dirpath=dirpath,
                         filename=filename,
                         save_top_k=save_top_k,
                         verbose=verbose,
                         save_last=save_last)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Get the path to the saved checkpoint
        checkpoint_path = self.dirpath

        # Log the checkpoint as an MLflow artifact
        mlflow.log_artifact(checkpoint_path)
