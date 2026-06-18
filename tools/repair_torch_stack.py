#!/usr/bin/env python3
"""Repair torch/torchvision mismatches in the motionedit conda env.

Important: torch, torchvision, and torchaudio must be installed together
as a matched triple in ONE pip command. Installing torchvision alone can
leave torchaudio on the wrong version and cause `torchvision::nms` errors.

Usage:
  python tools/repair_torch_stack.py --strategy motionedit
  python tools/repair_torch_stack.py --strategy match-current
"""

from __future__ import annotations

import argparse
import subprocess
import sys

MOTIONEDIT_STACK = ("2.6.0", "0.21.0", "2.6.0")
MATCH_CURRENT_STACK = ("2.12.1", "0.27.1", "2.12.1")

TORCH_INDICES = (
    "https://download.pytorch.org/whl/cu130",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu124",
)

VISION_CHECK = """
import torch
import torchvision
from torchvision.transforms import InterpolationMode
print(torch.__version__, torchvision.__version__)
"""


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pip(*args: str) -> None:
    run([sys.executable, "-m", "pip", *args])


def torch_versions() -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "not installed", "not installed"
    tv = "not installed"
    try:
        import torchvision

        tv = torchvision.__version__
    except Exception:
        pass
    return torch.__version__, tv


def vision_import_ok_subprocess() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", VISION_CHECK],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[repair] torchvision import failed:", flush=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)
        return False
    print(f"[repair] vision check OK: {result.stdout.strip()}", flush=True)
    return True


def cuda_index_for_stack(stack: tuple[str, str, str]) -> str:
    if stack == MATCH_CURRENT_STACK:
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu124"


def uninstall_torch_related() -> None:
    print("[repair] uninstalling torch stack (+ xformers if present)", flush=True)
    for _ in range(2):
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "xformers",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )


def install_torch_triple(
    stack: tuple[str, str, str],
    index_urls: tuple[str, ...],
) -> None:
    torch_v, tv_v, ta_v = stack
    last_error: Exception | None = None
    for index_url in index_urls:
        try:
            print(
                f"[repair] installing torch=={torch_v} torchvision=={tv_v} "
                f"torchaudio=={ta_v} via {index_url}",
                flush=True,
            )
            pip(
                "install",
                f"torch=={torch_v}",
                f"torchvision=={tv_v}",
                f"torchaudio=={ta_v}",
                "--index-url",
                index_url,
                "--force-reinstall",
                "--no-cache-dir",
            )
            if vision_import_ok_subprocess():
                return
        except Exception as exc:
            last_error = exc
            print(f"[repair] failed with {index_url}: {exc}", flush=True)
    raise RuntimeError(
        f"Could not install torch triple {stack}. Last error: {last_error}"
    ) from last_error


def install_motionedit_stack() -> None:
    uninstall_torch_related()
    install_torch_triple(MOTIONEDIT_STACK, TORCH_INDICES)
    print("[repair] reinstalling xformers for torch 2.6", flush=True)
    pip("install", "xformers==0.0.29.post3")


def install_match_current_stack() -> None:
    uninstall_torch_related()
    install_torch_triple(
        MATCH_CURRENT_STACK,
        (cuda_index_for_stack(MATCH_CURRENT_STACK),),
    )


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
        "numpy<2",
    )


def verify_all(strategy: str) -> None:
    if not vision_import_ok_subprocess():
        raise RuntimeError("torch/torchvision import check failed after repair")

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

    tv = Version(transformers.__version__)
    assert Version("4.57.0") <= tv < Version("5.0.0")
    _ = HybridCache
    _ = AutoProcessor
    ta = Version(torchao.__version__)
    if strategy == "motionedit":
        assert torch.__version__.startswith("2.6.")
        assert torchvision.__version__.startswith("0.21.")
        assert Version("0.16.0") <= ta < Version("0.17.0")
    else:
        assert torch.__version__.startswith("2.12.")
        assert torchvision.__version__.startswith("0.27.")
        assert ta >= Version("0.17.0")
    print("[repair] environment OK", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair torch/torchvision/HF dependency stack.")
    parser.add_argument(
        "--strategy",
        choices=("auto", "motionedit", "match-current"),
        default="auto",
        help="motionedit=torch2.6 (recommended for this repo), match-current=torch2.12+cu130",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[repair] current:", torch_versions(), flush=True)

    if args.strategy == "motionedit":
        install_motionedit_stack()
    elif args.strategy == "match-current":
        install_match_current_stack()
    else:
        try:
            install_motionedit_stack()
        except Exception as exc:
            print(f"[repair] motionedit pin failed, trying match-current: {exc}", flush=True)
            install_match_current_stack()

    strategy = detect_strategy() if args.strategy == "auto" else args.strategy
    install_hf_stack(strategy)
    verify_all(strategy)


if __name__ == "__main__":
    main()
