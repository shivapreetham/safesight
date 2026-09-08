"""Export the RQSA-MIL model to ONNX and produce an int8-quantized variant.

The exported graph takes (images (N, 3, 224, 224), iou (N, 1)) and returns
per-region logits, matching MobileNetMILRQSA.forward's first output.

Usage:
    python -m ml.export_onnx --checkpoint weights/rqsa_v1_clean.pt \
        --out weights/rqsa_v1.onnx
"""

import argparse
import os

import torch

from ml.torch_backend import TorchScorer


class ExportWrapper(torch.nn.Module):
    """ONNX-friendly wrapper: single tensor output (logits only)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, iou):
        logits, _ = self.model(images, iou)
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/rqsa_v1_clean.pt")
    parser.add_argument("--out", default="weights/rqsa_v1.onnx")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    model = TorchScorer(args.checkpoint, device="cpu").model
    wrapper = ExportWrapper(model).eval()

    dummy_images = torch.randn(1, 3, 224, 224)
    dummy_iou = torch.full((1, 1), 0.5)

    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_iou),
        args.out,
        input_names=["images", "iou"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "iou": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if not args.skip_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quant_path = args.out.replace(".onnx", "_int8.onnx")
        quantize_dynamic(args.out, quant_path, weight_type=QuantType.QUInt8)
        print(f"Quantized model written to {quant_path} "
              f"({os.path.getsize(quant_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
