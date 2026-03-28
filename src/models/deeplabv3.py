"""
DeepLabV3 모델 래퍼
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3Model(nn.Module):
	def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
		super().__init__()
		if in_channels != 3:
			raise ValueError("DeepLabV3Model currently supports in_channels=3 only.")

		# weights=None avoids any external download and keeps behavior deterministic.
		self.model = deeplabv3_resnet50(weights=None, weights_backbone=None)
		self.model.classifier = DeepLabHead(2048, out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.model(x)["out"]

		# Keep a strict output contract: logits shape must match input HxW.
		if out.shape[-2:] != x.shape[-2:]:
			out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

		return out
