"""Progressive pose editing with MotionNFT, VLM planning, flow, and identity checks.

This script builds on the MotionEdit README inference path:
Qwen-Image-Edit-2509 + the MotionEdit LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS_DIR = REPO_ROOT / "train" / "scripts"
if str(TRAIN_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(TRAIN_SCRIPTS_DIR))


# ============================================================
# PROMPTS
# ============================================================

PLANNING_SYSTEM = """You are an expert visual geometry analyst and image editing planner.

Your task: given a SOURCE image and a TARGET image containing objects of POSSIBLY
DIFFERENT categories, plan a minimal sequence of small incremental image edits that
progressively transform the SOURCE object's spatial configuration (viewpoint, scale,
pose, deformation) to match the TARGET object's — while keeping everything else
(object identity, texture, color, material, background) completely unchanged.

Key principle:
  You are NOT replacing the source object with the target.
  You are ONLY changing its geometric configuration.
  The source object must remain exactly what it is.

Output valid JSON only. No prose, no markdown fences outside JSON."""

PLANNING_USER = """Images provided:
  IMAGE 1 = SOURCE — the object to be edited
  IMAGE 2 = TARGET — reference for the desired geometric configuration

Goal: produce exactly {n_steps} incremental editing steps that move the SOURCE object
from its current configuration toward the TARGET object's configuration.
The TARGET image is a geometry/pose reference only. If SOURCE and TARGET have
different identities or categories, preserve the SOURCE identity/category and use
only analogous TARGET geometry, viewpoint, scale, pose, deformation, and placement.

════════════════════════════════════════════
PHASE A — ANALYSIS (fill every field)
════════════════════════════════════════════

A1. OBJECT CLASSIFICATION
  Identify each object and classify its deformability:
  - "rigid":        shape does not change (car, bottle, chair without joints)
  - "articulated":  rigid parts connected by joints (human, animal, robotic arm,
                    folding chair, scissors, umbrella)
  - "deformable":   continuously deformable surface (cloth, bag, rope, dough)

  For CROSS-CATEGORY pairs (cat→dog, wooden chair→metal chair), identify
  ANALOGOUS PARTS across the two objects for mapping.

A2. TRANSFORM GAP ANALYSIS — assess each axis:
  (a) VIEWPOINT
      - Azimuth change: clockwise/counterclockwise, estimated degrees
        (e.g., "source: front-facing → target: ~45° right-turn")
      - Elevation change: up/down tilt, estimated degrees
      - In-plane rotation: rotation within the image plane, estimated degrees

  (b) SCALE
      - How much larger/smaller is the target object relative to the source?
        Express as a ratio (e.g., "target is ~1.4× larger")

  (c) DEFORMATION / ARTICULATION (skip if both objects are rigid)
      For each movable part, describe:
      - source state: current configuration
      - target state: desired configuration
      - estimated delta: specific and measurable
        (e.g., "left knee: 160°→90° flex", "chair back: vertical→60° recline")

  (d) TRANSLATION
      - Where is the object in the image frame?
        (e.g., "source: center → target: slightly upper-left")

  (e) OVERALL COMPLEXITY
      - Which axes require the most change?
      - Which axes require NO change? (mark "no change needed" explicitly)

════════════════════════════════════════════
PHASE B — STEP PLANNING
════════════════════════════════════════════

Produce exactly {n_steps} steps following these rules:

ORDERING PRIORITY (coarse → fine):
  1. Largest viewpoint change first (azimuth before elevation)
  2. Scale adjustment
  3. Global deformation / articulation (compound multi-part)
  4. Per-part articulation (individual joints)
  5. Translation and fine adjustments last

STEP SIZE CONSTRAINT:
  Each step must be small enough that EVERY part of the object remains individually
  identifiable and trackable from the immediately preceding frame.
  This is critical: the output sequence will be used for correspondence matching,
  so each frame pair must be matchable by a standard feature matcher.
  Rule of thumb:
    - Viewpoint: ≤ 20° per step
    - In-plane rotation: ≤ 15° per step
    - Scale: ≤ 20% change per step
    - Articulation: ≤ 20° per joint per step

INSTRUCTION FORMAT — be specific and image-space concrete:
  Good: "Rotate the entire object ~15° clockwise within the image plane"
  Good: "Scale the object up by ~15%, keeping it centered"
  Good: "Fold the left armrest down by ~25° toward the seat"
  Bad:  "Adjust the object forward" (not measurable)
  Bad:  "Change the pose" (too vague)
  Bad:  "Make it look like the target" (not actionable)

HARD CONSTRAINTS (enforce on every step):
  - Do NOT change: object category, texture, color, material, surface details
  - Do NOT change: background, lighting, shadows (unless unavoidably caused by viewpoint)
  - Do NOT perform two dominant transforms in one step
  - Each step must move the configuration CLOSER to the target, never sideways

cumulative_progress: fraction of total gap closed after this step (0.0 → 1.0)

════════════════════════════════════════════
OUTPUT JSON SCHEMA
════════════════════════════════════════════

{{
  "analysis": {{
    "source_object": {{
      "description": "...",
      "deformability": "rigid | articulated | deformable"
    }},
    "target_object": {{
      "description": "...",
      "deformability": "rigid | articulated | deformable"
    }},
    "part_mapping": [
      {{"source_part": "...", "target_part": "...", "analogous": true}}
    ],
    "transform_gaps": {{
      "viewpoint": {{
        "azimuth": "...",
        "elevation": "...",
        "inplane_rotation": "..."
      }},
      "scale": "...",
      "articulation": [
        {{
          "part": "...",
          "source_state": "...",
          "target_state": "...",
          "delta": "..."
        }}
      ],
      "translation": "...",
      "dominant_axes": ["..."],
      "no_change_axes": ["..."]
    }}
  }},
  "steps": [
    {{
      "step": 1,
      "instruction": "...",
      "transform_type": "viewpoint_azimuth | viewpoint_elevation | inplane_rotation | scale | articulation | translation | fine_adjustment",
      "affected_parts": ["..."],
      "expected_change": "...",
      "magnitude_estimate": "...",
      "cumulative_progress": 0.0,
      "identity_warning": "none | check_texture | check_silhouette"
    }}
  ]
}}"""

VERIFY_SYSTEM = """You are an image editing quality verifier.
Answer every question with exactly 'Yes' or 'No'.
Be strict and precise."""

VERIFY_USER = """SOURCE → BEFORE → AFTER edit sequence.
IMAGE 1 = BEFORE this step
IMAGE 2 = AFTER this step
IMAGE 3 = TARGET configuration (final reference)

Important: IMAGE 3 is only a geometry/pose reference. Do not require the edited
object to adopt the TARGET object's identity, category, texture, color, or material.
The edited object must preserve the SOURCE identity and only move closer to TARGET
geometry/configuration.

This step's instruction: "{instruction}"
Transform type: {transform_type}
Expected change: {expected_change}

Answer Yes or No for each:
1. Was the instructed geometric change applied? (viewpoint/scale/pose as specified)
2. Is the source object's identity preserved? (same category, texture, color, material)
3. Is the background unchanged?
4. Is the change physically/geometrically plausible?
5. Are there no visual artifacts?
6. After this edit, is the source object's configuration CLOSER to the TARGET than before?

Format:
1. Yes/No
2. Yes/No
3. Yes/No
4. Yes/No
5. Yes/No
6. Yes/No"""

REPLAN_SYSTEM = """You are a motion editing planner.
A previous editing step failed verification.
Analyze the failure and produce a corrected instruction.
Output valid JSON only."""

REPLAN_USER = """IMAGE 1 = BEFORE edit
IMAGE 2 = AFTER edit (FAILED)
IMAGE 3 = TARGET configuration (final reference)

Important: IMAGE 3 is only a geometry/pose reference. The corrected instruction
must preserve SOURCE identity/category/texture/material and must not ask the editor
to copy TARGET identity.

Original instruction: "{failed_instruction}"
Transform type: {transform_type}
Expected change: {expected_change}
Identity warning: {identity_warning}
Failure reason: {failure_reason}

Analyze why it failed and provide a CORRECTED instruction:
{{
  "instruction": "...",
  "transform_type": "...",
  "affected_parts": ["..."],
  "expected_change": "...",
  "magnitude_estimate": "...",
  "cumulative_progress": ...,
  "identity_warning": "none | check_texture | check_silhouette"
}}"""


# ============================================================
# DATA TYPES
# ============================================================


@dataclass
class SubInstruction:
    step: int
    instruction: str
    transform_type: str
    affected_parts: list[str]
    expected_change: str
    magnitude_estimate: str
    cumulative_progress: float
    identity_warning: str = "none"


@dataclass
class Analysis:
    source_description: str
    source_deformability: str
    target_description: str
    target_deformability: str
    part_mapping: list[dict[str, Any]]
    viewpoint_azimuth: str
    viewpoint_elevation: str
    inplane_rotation: str
    scale: str
    articulation: list[dict[str, Any]]
    translation: str
    dominant_axes: list[str]
    no_change_axes: list[str]


@dataclass
class VLMVerifyResult:
    geometric_change_applied: bool
    identity_preserved: bool
    background_unchanged: bool
    physically_plausible: bool
    no_artifacts: bool
    closer_to_target: bool


@dataclass
class VerifyResult:
    flow_direction_ok: bool
    flow_magnitude_ok: bool
    identity_ok: bool
    texture_ok: bool
    silhouette_ok: bool
    semantic_ok: bool
    background_ok: bool
    overall_ok: bool
    failure_reason: Optional[str] = None


@dataclass
class PipelineResult:
    source_img: Image.Image
    target_img: Image.Image
    analysis: Analysis
    trajectory: list[Image.Image]
    instructions: list[SubInstruction]
    verify_results: list[VerifyResult]
    final_img: Image.Image


# ============================================================
# PARSING
# ============================================================


def parse_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from a VLM response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_yes_no(text: str) -> dict[int, bool]:
    answers: dict[int, bool] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*[\.\):\-]\s*(yes|no)\b", line, re.IGNORECASE)
        if match:
            answers[int(match.group(1))] = match.group(2).lower() == "yes"
    if len(answers) < 6:
        tokens = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        for idx, token in enumerate(tokens[:6], start=1):
            answers.setdefault(idx, token.lower() == "yes")
    return answers


def build_analysis(parsed: dict[str, Any]) -> Analysis:
    analysis = parsed["analysis"]
    source = analysis.get("source_object", {})
    target = analysis.get("target_object", {})
    gaps = analysis.get("transform_gaps", {})
    viewpoint = gaps.get("viewpoint", {})
    return Analysis(
        source_description=source.get("description", ""),
        source_deformability=source.get("deformability", ""),
        target_description=target.get("description", ""),
        target_deformability=target.get("deformability", ""),
        part_mapping=analysis.get("part_mapping", []),
        viewpoint_azimuth=viewpoint.get("azimuth", ""),
        viewpoint_elevation=viewpoint.get("elevation", ""),
        inplane_rotation=viewpoint.get("inplane_rotation", ""),
        scale=gaps.get("scale", ""),
        articulation=gaps.get("articulation", []),
        translation=gaps.get("translation", ""),
        dominant_axes=gaps.get("dominant_axes", []),
        no_change_axes=gaps.get("no_change_axes", []),
    )


def build_step(raw_step: dict[str, Any], fallback_step: int) -> SubInstruction:
    return SubInstruction(
        step=int(raw_step.get("step", fallback_step)),
        instruction=str(raw_step["instruction"]),
        transform_type=str(raw_step["transform_type"]),
        affected_parts=list(raw_step.get("affected_parts", ["object"])),
        expected_change=str(raw_step.get("expected_change", raw_step["instruction"])),
        magnitude_estimate=str(raw_step.get("magnitude_estimate", "")),
        cumulative_progress=float(raw_step.get("cumulative_progress", 0.0)),
        identity_warning=str(raw_step.get("identity_warning", "none")),
    )


# ============================================================
# MODEL WRAPPERS
# ============================================================


class QwenVLMClient:
    """Small wrapper around Qwen VL chat generation."""

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class MotionNFTEditor:
    """MotionNFT executor based on README's Qwen-Image-Edit + MotionEdit LoRA path."""

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str],
        device: str,
        device_map: Optional[str],
        dtype: torch.dtype,
        num_inference_steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        seed: int,
    ) -> None:
        from diffusers import QwenImageEditPlusPipeline

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            self.pipe.to(device)
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.guidance_scale = guidance_scale
        self.seed = seed

        adapter_source = lora_path or "elaine1wan/motionedit"
        self.pipe.load_lora_weights(
            adapter_source,
            weight_name="adapter_model_converted.safetensors",
            adapter_name="lora",
        )
        self.pipe.set_adapters(["lora"], adapter_weights=[1])

    @torch.no_grad()
    def edit(self, source_img: Image.Image, instruction: str, step_seed: int) -> Image.Image:
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(self.seed + step_seed)
        image = self.pipe(
            num_inference_steps=self.num_inference_steps,
            image=source_img,
            prompt=instruction,
            negative_prompt=" ",
            true_cfg_scale=self.true_cfg_scale,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]
        return image.convert("RGB")


class UniMatchFlowEstimator:
    def __init__(
        self,
        ckpt_path: Path,
        device: str,
        resize_to: Optional[int] = None,
    ) -> None:
        from unimatch.unimatch import UniMatch

        if not ckpt_path.exists():
            raise FileNotFoundError(
                "UniMatch checkpoint not found. Download "
                "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth "
                f"to {ckpt_path}"
            )

        self.device = torch.device(device)
        self.resize_to = resize_to
        self.model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task="flow",
        ).to(self.device)
        state = torch.load(ckpt_path, map_location="cpu")
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.last_flow: Optional[np.ndarray] = None

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _resize_pair(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        original_size = first.shape[-2:]
        if self.resize_to is None:
            return first, second, original_size

        height, width = original_size
        if height >= width:
            new_height = self.resize_to
            new_width = int(round(self.resize_to * width / height))
        else:
            new_width = self.resize_to
            new_height = int(round(self.resize_to * height / width))
        size = (new_height, new_width)
        return (
            F.interpolate(first, size=size, mode="bilinear", align_corners=False),
            F.interpolate(second, size=size, mode="bilinear", align_corners=False),
            original_size,
        )

    @staticmethod
    def _pad_to_factor(tensor: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = tensor.shape[-2:]
        pad_height = (factor - height % factor) % factor
        pad_width = (factor - width % factor) % factor
        return F.pad(tensor, (0, pad_width, 0, pad_height)), (pad_height, pad_width)

    @torch.no_grad()
    def estimate(self, before: Image.Image, after: Image.Image) -> np.ndarray:
        first = self._image_to_tensor(before)
        second = self._image_to_tensor(after.resize(before.size, Image.Resampling.BICUBIC))
        first, second, original_size = self._resize_pair(first, second)
        first_pad, pads = self._pad_to_factor(first, 32)
        second_pad, _ = self._pad_to_factor(second, 32)

        output = self.model(
            first_pad,
            second_pad,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            pred_bidir_flow=False,
            task="flow",
        )
        flow = output["flow_preds"][-1]
        pad_height, pad_width = pads
        if pad_height:
            flow = flow[..., :-pad_height, :]
        if pad_width:
            flow = flow[..., :-pad_width]
        if tuple(flow.shape[-2:]) != tuple(original_size):
            flow = F.interpolate(flow, size=original_size, mode="bilinear", align_corners=True)
        flow_np = flow[0].permute(1, 2, 0).detach().float().cpu().numpy()
        self.last_flow = flow_np
        return flow_np


class DINOv2IdentityScorer:
    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _features(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        output = self.model(**inputs)
        return output.last_hidden_state

    def similarity(self, before: Image.Image, after: Image.Image) -> float:
        feat_before = self._features(before)[:, 0]
        feat_after = self._features(after)[:, 0]
        return F.cosine_similarity(feat_before, feat_after).item()

    def patch_similarity(self, before: Image.Image, after: Image.Image) -> float:
        feat_before = self._features(before)[:, 1:]
        feat_after = self._features(after)[:, 1:]
        if feat_before.shape[1] != feat_after.shape[1]:
            min_tokens = min(feat_before.shape[1], feat_after.shape[1])
            feat_before = feat_before[:, :min_tokens]
            feat_after = feat_after[:, :min_tokens]
        return F.cosine_similarity(feat_before, feat_after, dim=-1).mean().item()


# ============================================================
# PLANNING / VERIFYING
# ============================================================


def plan(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    n_steps: int,
) -> tuple[Analysis, list[SubInstruction], dict[str, Any]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PLANNING_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = TARGET)"},
                {"type": "text", "text": PLANNING_USER.format(n_steps=n_steps)},
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=2048))
    analysis = build_analysis(parsed)
    steps = [build_step(step, idx + 1) for idx, step in enumerate(parsed["steps"])]
    if len(steps) != n_steps:
        raise ValueError(f"Planner returned {len(steps)} steps, expected {n_steps}.")
    return analysis, steps, parsed


def vlm_verify(
    before: Image.Image,
    after: Image.Image,
    target_img: Image.Image,
    step: SubInstruction,
    planner_vlm: QwenVLMClient,
) -> VLMVerifyResult:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": VERIFY_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before},
                {"type": "text", "text": "(IMAGE 1 = BEFORE)"},
                {"type": "image", "image": after},
                {"type": "text", "text": "(IMAGE 2 = AFTER)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 3 = TARGET)"},
                {
                    "type": "text",
                    "text": VERIFY_USER.format(
                        instruction=step.instruction,
                        transform_type=step.transform_type,
                        expected_change=step.expected_change,
                    ),
                },
            ],
        },
    ]
    answers = parse_yes_no(planner_vlm.chat(messages, max_new_tokens=128))
    return VLMVerifyResult(
        geometric_change_applied=answers.get(1, False),
        identity_preserved=answers.get(2, False),
        background_unchanged=answers.get(3, False),
        physically_plausible=answers.get(4, False),
        no_artifacts=answers.get(5, False),
        closer_to_target=answers.get(6, False),
    )


def replan(
    before: Image.Image,
    failed_after: Image.Image,
    target_img: Image.Image,
    failed_step: SubInstruction,
    failure_reason: str,
    planner_vlm: QwenVLMClient,
) -> SubInstruction:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": REPLAN_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before},
                {"type": "text", "text": "(IMAGE 1 = BEFORE edit)"},
                {"type": "image", "image": failed_after},
                {"type": "text", "text": "(IMAGE 2 = AFTER edit — FAILED)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 3 = TARGET)"},
                {
                    "type": "text",
                    "text": REPLAN_USER.format(
                        failed_instruction=failed_step.instruction,
                        transform_type=failed_step.transform_type,
                        expected_change=failed_step.expected_change,
                        identity_warning=failed_step.identity_warning,
                        failure_reason=failure_reason,
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=512))
    return build_step(parsed, failed_step.step)


# ============================================================
# VERIFICATION HELPERS
# ============================================================


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.linalg.norm(flow, axis=-1)


def cosine_similarity_np(first: np.ndarray, second: np.ndarray, eps: float = 1e-8) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    denom = np.linalg.norm(first) * np.linalg.norm(second) + eps
    return float(np.dot(first, second) / denom)


def parse_expected_direction(text: str) -> np.ndarray:
    lower = text.lower()
    direction = np.array([0.0, 0.0], dtype=np.float32)
    if "right" in lower:
        direction[0] += 1.0
    if "left" in lower:
        direction[0] -= 1.0
    if "down" in lower or "lower" in lower:
        direction[1] += 1.0
    if "up" in lower or "upper" in lower:
        direction[1] -= 1.0
    if np.linalg.norm(direction) < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return direction


def check_magnitude_in_range(roi_flow: np.ndarray) -> bool:
    if roi_flow.size == 0:
        return False
    mean_mag = np.linalg.norm(roi_flow, axis=-1).mean()
    return 0.5 < float(mean_mag) < 50.0


def check_scale_flow(flow: np.ndarray, motion_mask: np.ndarray, expected_change: str) -> tuple[bool, bool]:
    ys, xs = np.where(motion_mask)
    if len(xs) == 0:
        return False, False
    cy, cx = ys.mean(), xs.mean()
    roi_flow = flow[motion_mask]
    positions = np.stack([xs - cx, ys - cy], axis=-1).astype(np.float32)
    dots = (roi_flow * positions).sum(axis=-1)
    mean_dot = float(dots.mean())
    lower = expected_change.lower()
    if "larger" in lower or "scale up" in lower or "bigger" in lower or "enlarge" in lower:
        direction_ok = mean_dot > 0
    elif "smaller" in lower or "scale down" in lower or "shrink" in lower:
        direction_ok = mean_dot < 0
    else:
        direction_ok = abs(mean_dot) > 0
    return direction_ok, True


def check_rotation_flow(flow: np.ndarray, motion_mask: np.ndarray, expected_change: str) -> tuple[bool, bool]:
    ys, xs = np.where(motion_mask)
    if len(xs) == 0:
        return False, False
    cy, cx = ys.mean(), xs.mean()
    roi_flow = flow[motion_mask]
    rel_x = xs - cx
    rel_y = ys - cy
    cross = rel_x * roi_flow[:, 1] - rel_y * roi_flow[:, 0]
    mean_cross = float(cross.mean())
    lower = expected_change.lower()
    if "clockwise" in lower and "counterclockwise" not in lower:
        direction_ok = mean_cross > 0
    elif "counterclockwise" in lower or "anti-clockwise" in lower:
        direction_ok = mean_cross < 0
    else:
        direction_ok = abs(mean_cross) > 0
    return direction_ok, check_magnitude_in_range(roi_flow)


def check_background_preservation(before: Image.Image, after: Image.Image, border: float = 0.08) -> bool:
    before_arr = np.asarray(before.convert("RGB"), dtype=np.float32)
    after_arr = np.asarray(after.resize(before.size, Image.Resampling.BICUBIC).convert("RGB"), dtype=np.float32)
    height, width = before_arr.shape[:2]
    band_h = max(1, int(height * border))
    band_w = max(1, int(width * border))
    mask = np.zeros((height, width), dtype=bool)
    mask[:band_h, :] = True
    mask[-band_h:, :] = True
    mask[:, :band_w] = True
    mask[:, -band_w:] = True
    mse = ((before_arr[mask] - after_arr[mask]) ** 2).mean()
    return float(mse) < 250.0


def check_silhouette_direction(
    before: Image.Image,
    after: Image.Image,
    target_img: Image.Image,
    identity_scorer: DINOv2IdentityScorer,
) -> bool:
    """Avoid after-target identity similarity for cross-category source/target pairs.

    DINO similarity to the target can reward copying target identity when the target
    category differs from the source. Source identity is already checked with
    before-after DINO, while target-geometry progress is handled by flow checks and
    the VLM verifier.
    """
    del before, after, target_img, identity_scorer
    return True


def verify_edit(
    before: Image.Image,
    after: Image.Image,
    target_img: Image.Image,
    step: SubInstruction,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    planner_vlm: Optional[QwenVLMClient],
    flow_threshold: float,
    skip_vlm_verify: bool,
) -> VerifyResult:
    identity_threshold = 0.80
    skip_flow_direction = False
    check_flow_radial = False
    check_flow_roi = False
    check_flow_rotation = False

    if step.transform_type in {"viewpoint_azimuth", "viewpoint_elevation"}:
        identity_threshold = 0.60
        skip_flow_direction = True
    elif step.transform_type == "inplane_rotation":
        identity_threshold = 0.70
        check_flow_rotation = True
    elif step.transform_type == "scale":
        identity_threshold = 0.75
        check_flow_radial = True
    elif step.transform_type == "articulation":
        identity_threshold = 0.75
        check_flow_roi = True
    elif step.transform_type == "translation":
        identity_threshold = 0.80
    else:
        identity_threshold = 0.80
        check_flow_roi = True

    flow = flow_estimator.estimate(before, after)
    if skip_flow_direction:
        flow_direction_ok = True
        flow_magnitude_ok = True
    else:
        motion_mag = flow_magnitude(flow)
        motion_mask = motion_mag > flow_threshold
        if check_flow_radial:
            flow_direction_ok, flow_magnitude_ok = check_scale_flow(
                flow,
                motion_mask,
                step.expected_change,
            )
        elif check_flow_rotation:
            flow_direction_ok, flow_magnitude_ok = check_rotation_flow(
                flow,
                motion_mask,
                step.expected_change,
            )
        else:
            roi_flow = flow[motion_mask]
            if roi_flow.size == 0:
                flow_direction_ok = False
                flow_magnitude_ok = False
            else:
                mean_flow = roi_flow.mean(axis=0)
                expected_dir = parse_expected_direction(step.expected_change)
                min_cosine = 0.15 if check_flow_roi else 0.3
                flow_direction_ok = cosine_similarity_np(mean_flow, expected_dir) > min_cosine
                flow_magnitude_ok = check_magnitude_in_range(roi_flow)

    identity_score = identity_scorer.similarity(before, after)
    identity_ok = identity_score > identity_threshold
    texture_ok = True
    silhouette_ok = True
    background_ok = check_background_preservation(before, after)

    if step.identity_warning == "check_texture":
        texture_score = identity_scorer.patch_similarity(before, after)
        texture_ok = texture_score > 0.80
    elif step.identity_warning == "check_silhouette":
        silhouette_ok = check_silhouette_direction(before, after, target_img, identity_scorer)

    objective_failures = []
    if not flow_direction_ok:
        objective_failures.append("Motion direction incorrect for this transform type")
    if not flow_magnitude_ok:
        objective_failures.append("Motion magnitude out of expected range")
    if not identity_ok:
        objective_failures.append(
            f"Identity not preserved: DINO={identity_score:.2f} "
            f"(threshold={identity_threshold:.2f})"
        )
    if not texture_ok:
        objective_failures.append("Texture changed unexpectedly")
    if not silhouette_ok:
        objective_failures.append("Silhouette did not move toward target")
    if not background_ok:
        objective_failures.append("Background changed in border regions")

    if objective_failures:
        return VerifyResult(
            flow_direction_ok=flow_direction_ok,
            flow_magnitude_ok=flow_magnitude_ok,
            identity_ok=identity_ok,
            texture_ok=texture_ok,
            silhouette_ok=silhouette_ok,
            semantic_ok=False,
            background_ok=background_ok,
            overall_ok=False,
            failure_reason="; ".join(objective_failures),
        )

    if skip_vlm_verify:
        return VerifyResult(
            flow_direction_ok=flow_direction_ok,
            flow_magnitude_ok=flow_magnitude_ok,
            identity_ok=identity_ok,
            texture_ok=texture_ok,
            silhouette_ok=silhouette_ok,
            semantic_ok=True,
            background_ok=background_ok,
            overall_ok=True,
            failure_reason=None,
        )

    if planner_vlm is None:
        raise ValueError("planner_vlm is required unless --skip_vlm_verify is set.")

    vlm_result = vlm_verify(before, after, target_img, step, planner_vlm)
    background_ok = background_ok and vlm_result.background_unchanged
    overall = (
        flow_direction_ok
        and flow_magnitude_ok
        and identity_ok
        and texture_ok
        and silhouette_ok
        and vlm_result.geometric_change_applied
        and vlm_result.identity_preserved
        and vlm_result.physically_plausible
        and vlm_result.no_artifacts
        and vlm_result.closer_to_target
        and background_ok
    )

    failure_reason = None
    if not overall:
        reasons = []
        if not vlm_result.geometric_change_applied:
            reasons.append("Geometric change not applied as instructed")
        if not vlm_result.identity_preserved:
            reasons.append("VLM: identity not preserved")
        if not background_ok:
            reasons.append("Background changed")
        if not vlm_result.physically_plausible:
            reasons.append("Change not physically plausible")
        if not vlm_result.no_artifacts:
            reasons.append("Visual artifacts detected")
        if not vlm_result.closer_to_target:
            reasons.append("Not closer to target configuration")
        failure_reason = "; ".join(reasons)

    return VerifyResult(
        flow_direction_ok=flow_direction_ok,
        flow_magnitude_ok=flow_magnitude_ok,
        identity_ok=identity_ok,
        texture_ok=texture_ok,
        silhouette_ok=silhouette_ok,
        semantic_ok=vlm_result.geometric_change_applied,
        background_ok=background_ok,
        overall_ok=overall,
        failure_reason=failure_reason,
    )


def execute_progressive(
    source_img: Image.Image,
    target_img: Image.Image,
    steps: list[SubInstruction],
    editor: MotionNFTEditor,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    planner_vlm: Optional[QwenVLMClient],
    max_retries: int,
    flow_threshold: float,
    skip_vlm_verify: bool,
    output_dir: Path,
) -> tuple[list[Image.Image], list[SubInstruction], list[VerifyResult]]:
    trajectory = [source_img]
    verify_results: list[VerifyResult] = []
    final_steps: list[SubInstruction] = []

    for step_idx, step in enumerate(steps, start=1):
        retry_count = 0
        current_step = step
        before = trajectory[-1]
        edited: Optional[Image.Image] = None
        verify: Optional[VerifyResult] = None

        while retry_count <= max_retries:
            print(
                f"[step {step_idx}/{len(steps)} retry {retry_count}] "
                f"{current_step.instruction}"
            )
            edited = editor.edit(
                source_img=before,
                instruction=current_step.instruction,
                step_seed=step_idx * 100 + retry_count,
            )
            retry_path = output_dir / f"step_{step_idx:02d}_retry_{retry_count}.png"
            edited.save(retry_path)

            verify = verify_edit(
                before=before,
                after=edited,
                target_img=target_img,
                step=current_step,
                flow_estimator=flow_estimator,
                identity_scorer=identity_scorer,
                planner_vlm=planner_vlm,
                flow_threshold=flow_threshold,
                skip_vlm_verify=skip_vlm_verify,
            )

            if verify.overall_ok:
                print(f"[step {step_idx}] verification passed")
                break

            print(f"[step {step_idx}] verification failed: {verify.failure_reason}")
            if retry_count >= max_retries:
                break
            if planner_vlm is None:
                raise ValueError("Replanning requires planner_vlm.")
            current_step = replan(
                before=before,
                failed_after=edited,
                target_img=target_img,
                failed_step=current_step,
                failure_reason=verify.failure_reason or "unknown failure",
                planner_vlm=planner_vlm,
            )
            retry_count += 1

        if edited is None or verify is None:
            raise RuntimeError(f"Step {step_idx} did not produce an edited image.")
        trajectory.append(edited)
        final_steps.append(current_step)
        verify_results.append(verify)
        edited.save(output_dir / f"step_{step_idx:02d}.png")

    return trajectory, final_steps, verify_results


def progressive_pose_edit(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    editor: MotionNFTEditor,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    output_dir: Path,
    n_steps: int = 5,
    max_retries: int = 2,
    flow_threshold: float = 0.5,
    skip_vlm_verify: bool = False,
) -> PipelineResult:
    analysis, steps, raw_plan = plan(source_img, target_img, planner_vlm, n_steps)
    (output_dir / "plan.json").write_text(
        json.dumps(raw_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trajectory, final_steps, verify_results = execute_progressive(
        source_img=source_img,
        target_img=target_img,
        steps=steps,
        editor=editor,
        flow_estimator=flow_estimator,
        identity_scorer=identity_scorer,
        planner_vlm=planner_vlm,
        max_retries=max_retries,
        flow_threshold=flow_threshold,
        skip_vlm_verify=skip_vlm_verify,
        output_dir=output_dir,
    )
    return PipelineResult(
        source_img=source_img,
        target_img=target_img,
        analysis=analysis,
        trajectory=trajectory,
        instructions=final_steps,
        verify_results=verify_results,
        final_img=trajectory[-1],
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate step-by-step MotionNFT pose edits from source and target images.",
    )
    parser.add_argument("--source_image", required=True, help="Path to the source image to edit.")
    parser.add_argument("--target_image", required=True, help="Path to target pose/configuration image.")
    parser.add_argument("--output_dir", default="outputs/progressive_pose_edit")
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--editor_base_model", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--motionedit_lora_path", default=None)
    parser.add_argument("--planner_vlm", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov2_model", default="facebook/dinov2-base")
    parser.add_argument(
        "--unimatch_ckpt",
        default=str(
            REPO_ROOT
            / "train"
            / "scripts"
            / "unimatch"
            / "pretrained"
            / "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        ),
    )

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--editor_device_map", default=None)
    parser.add_argument("--vlm_device_map", default="auto")
    parser.add_argument("--flow_resize_to", type=int, default=None)
    parser.add_argument("--flow_threshold", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--skip_vlm_verify", action="store_true")
    return parser.parse_args()


def serializable_result(result: PipelineResult) -> dict[str, Any]:
    return {
        "analysis": asdict(result.analysis),
        "instructions": [asdict(step) for step in result.instructions],
        "verify_results": [asdict(verify) for verify in result.verify_results],
        "final_image": "final.png",
        "trajectory": [f"step_{idx:02d}.png" for idx in range(len(result.trajectory))],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    source_img = Image.open(args.source_image).convert("RGB")
    target_img = Image.open(args.target_image).convert("RGB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_img.save(output_dir / "step_00.png")
    source_img.save(output_dir / "source.png")
    target_img.save(output_dir / "target.png")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    planner_vlm = QwenVLMClient(
        model_id=args.planner_vlm,
        device_map=args.vlm_device_map,
        torch_dtype="auto",
    )
    editor = MotionNFTEditor(
        base_model=args.editor_base_model,
        lora_path=args.motionedit_lora_path,
        device=args.device,
        device_map=args.editor_device_map,
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    flow_estimator = UniMatchFlowEstimator(
        ckpt_path=Path(args.unimatch_ckpt),
        device=args.device,
        resize_to=args.flow_resize_to,
    )
    identity_scorer = DINOv2IdentityScorer(args.dinov2_model, args.device)

    result = progressive_pose_edit(
        source_img=source_img,
        target_img=target_img,
        planner_vlm=planner_vlm,
        editor=editor,
        flow_estimator=flow_estimator,
        identity_scorer=identity_scorer,
        output_dir=output_dir,
        n_steps=args.n_steps,
        max_retries=args.max_retries,
        flow_threshold=args.flow_threshold,
        skip_vlm_verify=args.skip_vlm_verify,
    )
    result.final_img.save(output_dir / "final.png")
    (output_dir / "result.json").write_text(
        json.dumps(serializable_result(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved progressive editing outputs to {output_dir}")


if __name__ == "__main__":
    main()
