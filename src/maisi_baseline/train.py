import argparse
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.maisi_baseline.config import DEFAULT_CONFIG
from src.maisi_baseline.trainer import MAISITrainer
from src.mri2ct.utils import cleanup_gpu


def main():
    parser = argparse.ArgumentParser(description="Train MAISI ControlNet Baseline")
    parser.add_argument("--subjects", nargs="*", help="Specific subjects for single image optimization (e.g., 1ABA005)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["total_epochs"])
    parser.add_argument("--wandb", type=str, default="True", choices=["True", "False"], help="Enable/disable wandb")
    parser.add_argument("--resume_wandb_id", type=str, help="WandB Run ID to resume from")

    args = parser.parse_args()

    use_wandb = args.wandb == "True"

    config_dict = {"subjects": args.subjects, "batch_size": args.batch_size, "lr": args.lr, "total_epochs": args.epochs, "wandb": use_wandb, "resume_wandb_id": args.resume_wandb_id}

    if args.subjects:
        print(f"🔬 RUNNING SINGLE SUBJECT TEST: {args.subjects}")
        # Tutorial settings: 1000 epochs, 5e-4 LR, 1 batch
        config_dict.update(
            {
                "total_epochs": 1000,
                "lr": 5e-4,
                "batch_size": 1,
                "wandb_note": f"MAISI Overfitting Test (Tutorial Settings) - {args.subjects}",
                "val_interval": 50,
                "model_save_interval": 100,
            }
        )

    try:
        trainer = MAISITrainer(config_dict)
        trainer.train()
        cleanup_gpu()
    except KeyboardInterrupt:
        print("Interrupted.")
        cleanup_gpu()
    except Exception as _:
        import traceback

        traceback.print_exc()
        cleanup_gpu()


if __name__ == "__main__":
    main()
