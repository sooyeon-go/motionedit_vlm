#!/usr/bin/env python3
"""Repair torch/torchvision mismatches in the motionedit conda env.

Usage:
  python tools/repair_torch_stack.py
  python tools/repair_torch_stack.py --strategy match-current
  python tools/repair_torch_stack.py --strategy motionedit
"""

from __future__ import annotations

import argparse
import subprocess
import sys


TORCH_INDICES = (
    "https://download.pytorch.org/whl/cu130",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu124",
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pip(*args: str) -> None:
    run([sys.executable, "-m", "pip", *args])


def torch_versions() -> tuple[str, str]:
    import torch

    tv = "not installed"
    try:
        import torchvision

        tv = torchvision.__version__
    except Exception:
        pass
    return torch.__version__, tv


def vision_import_ok() -> bool:
    try:
        import torch  # noqa: F401
        from torchvision.transforms import InterpolationMode  # noqa: F401

        return True
    except Exception as exc:
        print(f"[repair] torchvision import failed: {exc}", flush=True)
        return False


def cuda_index_from_torch() -> str:
    import torch

    tag = torch.__version__.split("+")[-1] if "+" in torch.__version__ else "cu124"
    if tag.startswith("cu"):
        return f"https://download.pytorch.org/whl/{tag}"
    return "https://download.pytorch.org/whl/cu124"


def uninstall_torch_stack() -> None:
    print("[repair] uninstalling torch / torchvision / torchaudio", flush=True)
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
    )


def install_motionedit_stack() -> None:
    uninstall_torch_stack()
    last_error: Exception | None = None
    for index_url in TORCH_INDICES:
        try:
            print(f"[repair] trying motionedit stack via {index_url}", flush=True)
            pip(
                "install",
                "torch==2.6.0",
                "torchvision==0.21.0",
                "torchaudio==2.6.0",
                "--index-url",
                index_url,
                "--force-reinstall",
                "--no-cache-dir",
            )
            if vision_import_ok():
                return
        except Exception as exc:
            last_error = exc
            print(f"[repair] failed with {index_url}: {exc}", flush=True)
    raise RuntimeError("Could not install motionedit torch 2.6 stack") from last_error


def install_match_current_stack() -> None:
    import torch

    index_url = cuda_index_from_torch()
    print(
        f"[repair] upgrading torchvision/torchaudio to match torch {torch.__version__} "
        f"via {index_url}",
        flush=True,
    )
    pip(
        "install",
        "torchvision",
        "torchaudio",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--index-url",
        index_url,
    )
    if not vision_import_ok():
        raise RuntimeError("torchvision still incompatible after upgrade")


def detect_strategy() -> str:
    import torch
    from packaging.version import Version

    torch_ver = Version(torch.__version__.split("+")[0])
    if torch_ver >= Version("2.11.0"):
        return "match-current"
    return "motionedit"


def install_hf_stack(strategy: str) -> None:
    if strategy == "match-current":
        torchao_spec = "torchao>=0.17.0"
    else:
        torchao_spec = "torchao>=0.16.0,<0.17.0"

    print(f"[repair] installing HF stack with {torchao_spec}", flush=True)
    pip(
        "install",
        "transformers>=4.57.0,<5.0",
        "peft>=0.18.0",
        "diffusers==0.36.0",
        torchao_spec,
        "huggingface-hub>=0.34.0",
        "qwen-vl-utils",
        "accelerate",
        "safetensors",
        "packaging",
    )


def verify_all(strategy: str) -> None:
    import peft
    import torch
    import torchao
    import transformers
    from packaging.version import Version
    from transformers import AutoProcessor, HybridCache

    import torchvision

    print(f"[repair] torch={torch.__version__}", flush=True)
    print(f"[repair] torchvision={torchvision.__version__}", flush=True)
    print(f"[repair] transformers={transformers.__version__}, peft={peft.__version__}", flush=True)
    print(f"[repair] torchao={torchao.__version__}", flush=True)

    from torchvision.transforms import InterpolationMode  # noqa: F401

    tv = Version(transformers.__version__)
    assert Version("4.57.0") <= tv < Version("5.0.0")
    _ = HybridCache
    _ = AutoProcessor
    torch_ver = Version(torch.__version__.split("+")[0])
    ta = Version(torchao.__version__)
    if strategy == "motionedit":
        assert torch.__version__.startswith("2.6.")
        assert torchvision.__version__.startswith("0.21.")
        assert Version("0.16.0") <= ta < Version("0.17.0")
    else:
        assert ta >= Version("0.17.0")
    print("[repair] environment OK", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair torch/torchvision/HF dependency stack.")
    parser.add_argument(
        "--strategy",
        choices=("auto", "motionedit", "match-current"),
        default="auto",
        help="auto: try motionedit pin, then match-current if needed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[repair] current:", torch_versions(), flush=True)

    if args.strategy in {"motionedit", "auto"}:
        try:
            install_motionedit_stack()
        except Exception as exc:
            if args.strategy == "motionedit":
                raise
            print(f"[repair] motionedit pin failed: {exc}", flush=True)

    if args.strategy == "match-current" or (
        args.strategy == "auto" and not vision_import_ok()
    ):
        install_match_current_stack()

    if not vision_import_ok():
        raise RuntimeError(
            "torch/torchvision still broken. Run:\n"
            "  python tools/repair_torch_stack.py --strategy match-current"
        )

    strategy = detect_strategy() if args.strategy == "auto" else args.strategy
    install_hf_stack(strategy)
    verify_all(strategy)


if __name__ == "__main__":
    main()
