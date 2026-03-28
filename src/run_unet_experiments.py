import subprocess
import sys


# Why this file exists:
# - Keep experiment execution separate from model implementation.
# - Make repeated runs easy by listing experiment configs in one place.
EXPERIMENTS = [
    {
        "name": "unet_baseline_e10_img256",
        "train_args": [
            "--epochs", "10",
            "--image-size", "256",
            "--batch-size", "4",
            "--output-dir", "outputs/unet_baseline_e10_img256",
        ],
        "predict_args": [
            "--checkpoint", "outputs/unet_baseline_e10_img256/best_unet.pt",
            "--output-dir", "outputs/unet_baseline_e10_img256/test_predictions",
        ],
    }
]


def run_command(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    # One experiment = train first, then inference on test images.
    for exp in EXPERIMENTS:
        print(f"\n=== Experiment: {exp['name']} ===")

        train_cmd = [sys.executable, "train_unet_baseline.py", *exp["train_args"]]
        run_command(train_cmd)

        predict_cmd = [sys.executable, "predict_unet_baseline.py", *exp["predict_args"]]
        run_command(predict_cmd)


if __name__ == "__main__":
    main()
