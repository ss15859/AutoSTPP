import os
import torch
from pytorch_lightning.cli import ArgsType
from pytorch_lightning.callbacks import ModelCheckpoint

from data.lightning.sliding_window import SlidingWindowDataModule
from models.lightning.stpp import BaseSTPointProcess
from cli import MyLightningCLI
from utils import find_ckpt_path, increase_u_limit

CHECKPOINT_ROOT = "output_data/"

def cli_main(args: ArgsType = None):
    torch.set_float32_matmul_precision('medium')

    cli = MyLightningCLI(
        BaseSTPointProcess, 
        SlidingWindowDataModule, 
        subclass_mode_model=True, 
        subclass_mode_data=True,
        save_config_callback=None,
        run=False
    )

    # Extract experiment name from config
    experiment_name = cli.config["trainer"]["logger"]["init_args"].get("experiment", "default_experiment")
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, experiment_name)

    # Ensure checkpoint subdirectory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_nll",  
        mode="min",
        save_top_k=1
    )
    cli.trainer.callbacks.append(checkpoint_callback)

    return cli, checkpoint_dir


if __name__ == '__main__':
    cli, checkpoint_dir = cli_main()
    increase_u_limit()

    cli.trainer.logger.log_hyperparams({'seed': cli.config['seed_everything']})

    # Train model
    cli.trainer.fit(cli.model, cli.datamodule)

    # Find best checkpoint path inside the experiment-named subdir
    best_ckpt_path = os.path.join(checkpoint_dir, "best.ckpt")
    if os.path.exists(best_ckpt_path):
        print(f"Using best checkpoint: {best_ckpt_path}")
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=best_ckpt_path)
    else:
        print("Warning: Best checkpoint not found, testing without checkpoint.")
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=None)
