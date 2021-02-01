from data_modules.data_module import GrenadeDataModule
from pl_bolts.models.regression import LogisticRegression
from pytorch_lightning import Trainer
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback

class MetricsCallback(Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def objective(trial):
    bias = trial.suggest_categorical("bias", [True, False])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log = True)
    l1_strength = trial.suggest_float("l1_strength", 1e-10, 1e2)
    l2_strength = trial.suggest_float("l2_strength", 1e-10, 1e2)


    metrics_callback =  MetricsCallback()
    datamodule = GrenadeDataModule()
    model = LogisticRegression(input_dim = 12, num_classes = 2, bias = bias, learning_rate = learning_rate, l1_strength = l1_strength, l2_strength = l2_strength)
    trainer = Trainer(max_epochs = 200, gpus = 1, callbacks = [metrics_callback, PyTorchLightningPruningCallback(trial, monitor = "val_acc")])
    trainer.fit(model = model, datamodule = datamodule)

    return metrics_callback.metrics[-1]["val_acc"].item()

if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction = "maximize", pruner = pruner)
    study.optimize(objective, n_trials = 100, timeout = 900)
    trial = study.best_trial
    print(f"Best trial: {trial.value}")
    print("Best trial params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")