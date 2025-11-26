import wandb


class WBLogger:
    def __init__(self, project: str, config: dict, name: str | None = None):
        self.run = wandb.init(project=project, config=config, name=name)

    def log(self, metrics: dict, step: int | None = None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
