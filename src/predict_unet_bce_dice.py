from predict_unet_baseline import main


if __name__ == "__main__":
    main(
        description="Run U-Net BCE+DICE inference on test images",
        default_checkpoint="outputs/unet_bce_dice/best_unet.pt",
        default_output_dir="outputs/unet_bce_dice/test_predictions",
    )
