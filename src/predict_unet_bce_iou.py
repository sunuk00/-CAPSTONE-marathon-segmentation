from predict_unet_baseline import main


if __name__ == "__main__":
    main(
        description="Run U-Net BCE+IoU inference on test images",
        default_checkpoint="outputs/unet_bce_iou/best_unet.pt",
        default_output_dir="outputs/unet_bce_iou/test_predictions",
    )
