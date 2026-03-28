import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

from train_unet_baseline import UNet, VALID_EXTENSIONS, dice_score_from_logits


def collect_images(images_dir: Path) -> List[Path]:
    # Load test images in a deterministic order for easier comparison.
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])


def preprocess_image(image_path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    # Keep original size so prediction can be resized back before saving.
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    image_arr = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).unsqueeze(0)
    return image_tensor, original_size


def load_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    # Optional mask loader used only when test masks exist.
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    mask_arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)


def build_arg_parser(
    description: str = "Run U-Net baseline inference on test images",
    default_checkpoint: str = "outputs/unet_baseline/best_unet.pt",
    default_output_dir: str = "outputs/unet_baseline/test_predictions",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--test-root", type=str, default="data/test")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    return parser


def main(
    description: str = "Run U-Net baseline inference on test images",
    default_checkpoint: str = "outputs/unet_baseline/best_unet.pt",
    default_output_dir: str = "outputs/unet_baseline/test_predictions",
) -> None:
    # 1) Read runtime arguments.
    args = build_arg_parser(
        description=description,
        default_checkpoint=default_checkpoint,
        default_output_dir=default_output_dir,
    ).parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2) Load trained model weights and training args from checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    image_size = args.image_size if args.image_size > 0 else int(train_args.get("image_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA (GPU: {gpu_name})")
    else:
        print("Using device: CPU")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3) Prepare I/O folders.
    test_root = Path(args.test_root)
    test_images_dir = test_root / "images"
    test_masks_dir = test_root / "masks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images folder not found: {test_images_dir}")

    image_paths = collect_images(test_images_dir)
    if not image_paths:
        raise RuntimeError("No test images found.")

    # 4) Run prediction and save binary masks to output directory.
    dice_scores: List[float] = []
    with torch.no_grad():
        for image_path in image_paths:
            image_tensor, original_size = preprocess_image(image_path, image_size)
            image_tensor = image_tensor.to(device)

            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            pred = (probs > args.threshold).float()

            # Save prediction mask in original image size for easy visual check.
            pred_arr = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            pred_img = Image.fromarray(pred_arr, mode="L").resize(original_size, Image.Resampling.NEAREST)
            pred_name = f"{image_path.stem}_pred.png"
            pred_img.save(output_dir / pred_name)

            # If test mask exists, compute Dice score to quantify quality.
            mask_candidates = [test_masks_dir / f"{image_path.stem}{ext}" for ext in VALID_EXTENSIONS]
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if mask_path is not None:
                gt_mask = load_mask(mask_path, image_size).to(device)
                dice = dice_score_from_logits(logits, gt_mask).item()
                dice_scores.append(dice)

    print(f"Saved {len(image_paths)} prediction masks to: {output_dir}")
    if dice_scores:
        print(f"Mean test Dice: {sum(dice_scores) / len(dice_scores):.4f}")
    else:
        print("No matching test masks found. Saved only prediction images.")


if __name__ == "__main__":
    main()
