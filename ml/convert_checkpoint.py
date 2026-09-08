"""Convert the original research checkpoint into a clean state_dict file.

The original rqsa_best_rqsa_v1.pt was saved with a pickled Config object from
the training script, which forces torch.load(weights_only=False) and a stub
class injection to deserialize. This script loads it once, extracts only the
tensor state_dict plus JSON-safe metadata, and re-saves it so the serving code
can always load with weights_only=True.

Usage:
    python -m ml.convert_checkpoint --src weights/rqsa_best_rqsa_v1.pt \
        --dst weights/rqsa_v1_clean.pt
"""

import argparse
import sys
import types

import torch


class Config:
    """Stub so pickle can deserialize the Config saved inside the checkpoint."""


def _inject_config_stub():
    main_mod = sys.modules.get("__main__", types.ModuleType("__main__"))
    if not hasattr(main_mod, "Config"):
        main_mod.Config = Config
        sys.modules["__main__"] = main_mod
    # Training script may also be referenced by module name in the pickle.
    for name in ("train_rqsa_mil", "train_wmil_less-bagsize"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.Config = Config
            sys.modules[name] = mod


def convert(src: str, dst: str) -> dict:
    _inject_config_stub()
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    state = ckpt.get("model_state", ckpt)
    if "backbone.0.0.weight" not in state or "attention.iou_embed.0.weight" not in state:
        raise ValueError("Checkpoint does not look like an RQSA-MIL state_dict")

    meta = {}
    for key in ("epoch", "backbone_type", "tag"):
        value = ckpt.get(key)
        if isinstance(value, (int, float, str, bool)):
            meta[key] = value

    clean = {
        "format": "safesight-rqsa-v1",
        "arch": "mobilenet_v2_rqsa_mil",
        "pseudo_iou": 0.5,
        "state_dict": {k: v.cpu() for k, v in state.items()},
        "meta": meta,
    }
    torch.save(clean, dst)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    meta = convert(args.src, args.dst)
    print(f"Converted {args.src} -> {args.dst}")
    print(f"Metadata: {meta}")

    # Round-trip check: the clean file must load with weights_only=True.
    reloaded = torch.load(args.dst, map_location="cpu", weights_only=True)
    n_tensors = len(reloaded["state_dict"])
    print(f"Round-trip OK: {n_tensors} tensors, format={reloaded['format']}")


if __name__ == "__main__":
    main()
