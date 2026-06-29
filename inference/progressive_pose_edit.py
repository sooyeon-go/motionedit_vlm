"""Progressive pose editing with MotionNFT, VLM planning, flow, and identity checks.

Default model layout on shared storage:
  /data/shared-vilab/pretrained_models/Qwen-Image-Edit-2511      (editor base)
  /data/shared-vilab/pretrained_models/Qwen3-VL-8B-Instruct      (planner / verifier VLM)
  /data/shared-vilab/pretrained_models/motionedit_vlm/
    motionedit-lora/                                             (MotionEdit LoRA)
    dinov2-base/                                                 (identity scorer)
    unimatch/pretrained/gmflow-...pth                            (optical flow)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    log(f"  saved {path}")


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS_DIR = REPO_ROOT / "train" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (TRAIN_SCRIPTS_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from model_paths import (  # noqa: E402
    DINOV2_MODEL as DEFAULT_DINOV2_MODEL,
    EDITOR_BASE_MODEL as DEFAULT_EDITOR_BASE_MODEL,
    MOTIONEDIT_LORA_DIR as DEFAULT_MOTIONEDIT_LORA_PATH,
    PLANNER_VLM_MODEL as DEFAULT_PLANNER_VLM,
    QWEN_ANGLES_LORA_DIR as DEFAULT_QWEN_ANGLES_LORA_PATH,
    QWEN_ANGLES_LORA_WEIGHT,
    UNIMATCH_CKPT as DEFAULT_UNIMATCH_CKPT,
)

MOTIONEDIT_LORA_WEIGHT = "adapter_model_converted.safetensors"
ANGLE_LORA_WEIGHT = QWEN_ANGLES_LORA_WEIGHT

AZIMUTH_BINS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]
ELEVATION_BINS = [
    "low-angle shot",
    "eye-level shot",
    "elevated shot",
    "high-angle shot",
]
DISTANCE_BINS = [
    "close-up",
    "medium shot",
    "wide shot",
]


# ============================================================
# PROMPTS
# ============================================================

PREALIGN_SYSTEM = """You are an expert visual geometry pre-alignment judge.

Given a SOURCE image and a TARGET image, decide whether the SOURCE should receive
deterministic flip/rotate transforms BEFORE diffusion editing.

CRITICAL METHOD — two-landmark axis comparison:
  You MUST pick exactly TWO landmarks that are BOTH:
    1) clearly visible on the SOURCE object, AND
    2) clearly visible on the TARGET object (as analogous parts).
  These form primary_landmark + secondary_landmark and define an alignment axis.

  NEVER pick a landmark that exists on only one image.
  If you cannot find TWO shared landmarks on both source and target, skip pre-align.

  Do NOT decide flip/rotate from a single landmark alone.
  Compare the axis vector on SOURCE vs TARGET:
    - horizontal flip: when BOTH landmarks are left/right mirrored as a pair
    - in-plane rotation: when the primary→secondary axis angle differs

Good landmark pairs (pick one pair visible on BOTH images):
  - animals: head + tail, nose + tail, head + hind legs
  - humans: head + feet, head + hips, shoulders + hips
  - cars/vehicles: front bumper/headlights + rear bumper/taillights, front wheel + rear wheel
  - rigid objects: tip + base, handle + spout, narrow end + wide end

For each landmark, record its position in the IMAGE FRAME using:
  horizontal: "left" | "center" | "right"
  vertical:   "top" | "center" | "bottom"

Use TARGET only as geometry reference. Preserve SOURCE identity. Never copy target identity.

Allowed transforms:
  - horizontal flip: when the TWO-landmark axis is horizontally MIRRORED
    (primary and secondary swap left/right sides together).
  - small in-plane rotation: when the primary→secondary axis angle differs between
    source and target AFTER accounting for any needed flip.

Do NOT use pre-align for: true 3D viewpoint change, articulation, deformation, scale,
or any transform that would require generating new object content.

Output valid JSON only."""

PREALIGN_USER = """Images provided:
  IMAGE 1 = SOURCE — the image/object to transform
  IMAGE 2 = TARGET — geometry reference only

Step 1 — Pick exactly ONE landmark pair that exists on BOTH source AND target:
  primary_landmark + secondary_landmark.
  Both must be visible on SOURCE and on TARGET (analogous parts if cross-category).
Step 2 — Record BOTH landmarks' frame positions on SOURCE and on TARGET.
  If a landmark is missing on either image, mark comparable=false and skip pre-align.
Step 3 — Compare the primary→secondary axis on source vs target.
Step 4 — Decide horizontal flip from whether the pair is mirrored left/right.
Step 5 — Decide rotation_degrees from axis-angle difference (in-plane only).
Step 6 — Set confidence based on how clearly BOTH landmarks are visible.

Return this exact schema:
{{
  "object_category": "cat | car | person | ...",
  "alignment_axis": {{
    "primary_landmark": "head",
    "secondary_landmark": "tail",
    "description": "head-to-tail body axis"
  }},
  "anchor_features": [
    {{
      "name": "head",
      "role": "primary",
      "visible_on_source": true,
      "visible_on_target": true,
      "source": {{"horizontal": "left", "vertical": "center"}},
      "target": {{"horizontal": "right", "vertical": "center"}},
      "comparable": true
    }},
    {{
      "name": "tail",
      "role": "secondary",
      "visible_on_source": true,
      "visible_on_target": true,
      "source": {{"horizontal": "right", "vertical": "center"}},
      "target": {{"horizontal": "left", "vertical": "center"}},
      "comparable": true
    }}
  ],
  "horizontal_layout_match": "same | mirrored | mixed | unclear",
  "should_horizontal_flip": true,
  "rotation_degrees": 0.0,
  "rotation_basis": "head→tail axis tilted ~10° clockwise in source vs target",
  "confidence": 0.0,
  "reason": "short explanation citing BOTH landmark names and positions"
}}

Rules:
  - Each landmark MUST be visible on BOTH source and target. Set visible_on_source/target accordingly.
  - anchor_features MUST contain at least the primary and secondary landmarks, both comparable=true.
  - Do NOT recommend flip or rotation unless BOTH shared landmarks are clear on BOTH images.
  - If you cannot find two shared landmarks, set comparable=false, confidence below {min_confidence}, and skip.
  - Recommend flip ONLY when horizontal_layout_match is "mirrored" for the pair.
  - rotation_degrees: derived from primary→secondary axis angle difference.
    Positive = counterclockwise. Keep within -{max_rotation} to {max_rotation}.
  - If either landmark is missing/unclear, set confidence below {min_confidence} and skip transforms.
  - Focus only on the two object landmarks.
  - Cross-category pairs: map analogous parts (cat head ↔ dog head, car front ↔ car front)."""

PLANNING_SYSTEM = """You are an expert visual geometry analyst and image editing planner.

Your task: given a SOURCE image and an ENDPOINT geometry reference image containing
objects of POSSIBLY DIFFERENT categories, plan a sequence of trackable interpolation
edits that progressively transform the SOURCE object's spatial configuration
(viewpoint, scale, pose, deformation) to match the ENDPOINT object's geometry —
while keeping everything else (especially object identity, texture, color, and
material) unchanged.

CRITICAL — instruction style for the editor:
  Each step's "instruction" field is sent DIRECTLY to an image editing diffusion model.
  That model understands natural, descriptive action language much better than numeric
  geometry (degrees, percentages, ratios).
  Write instructions like a human photo editor would — describing what should happen
  in plain visual terms, referencing body parts and spatial relations.

Key principle:
  You are NOT replacing the source object with the target.
  You are ONLY changing its geometric configuration.
  The source object must remain exactly what it is.
  Priority 1: the final frame must match the ENDPOINT pose, camera angle, scale,
  and visible part layout.
  Priority 2: intermediate frames should form a trackable interpolation path.

Output valid JSON only. No prose, no markdown fences outside JSON."""

PLANNING_USER = """Images provided:
  IMAGE 1 = SOURCE — the object to be edited
  IMAGE 2 = ENDPOINT — reference for the desired geometric configuration

Goal: produce exactly {n_steps} incremental editing steps that move the SOURCE object
from its current configuration toward the ENDPOINT object's configuration.
The ENDPOINT image is a geometry/pose reference only. If SOURCE and ENDPOINT have
different identities or categories, preserve the SOURCE identity/category and use
only analogous ENDPOINT geometry, viewpoint, scale, pose, deformation, and placement.

Interpolation objective:
  The final step should match ENDPOINT geometry as closely as possible.
  Each intermediate step should look like a moderate-small interpolation step, similar
  to the object pose/view change between video frames about 8-10 frames apart:
  clear visible progress, but still trackable between consecutive images.
  Do not make steps so tiny that there is almost no progress, and do not make steps
  so large that major object parts cannot be tracked from the previous frame.

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

A1b. SHARED PARTS (critical — fill before planning steps)
  List every object part that is clearly visible on BOTH source AND target.
  These are the "intersection" parts — analogous structures present in both images
  (e.g. head, tail, four legs, ears for animals; wheels, headlights, windshield for cars).
  Cross-category: use analogous names (cat head ↔ dog head counts as shared).
  Parts visible on only one image must NOT be listed.
  Shared parts should remain identifiable across consecutive interpolation steps.
  If ENDPOINT geometry naturally occludes a source-visible part, make that visibility
  change gradual rather than abrupt.

A2. TRANSFORM GAP ANALYSIS — assess each axis:
  First pick a TWO-landmark axis on source and target
  (head->tail, front->rear, tip->base) and note where each landmark sits in the frame
  (left/center/right, top/center/bottom). Use this axis to reason about remaining gaps
  after any pre-alignment. Describe gaps in VISUAL terms, not numeric angles.

  (a) VIEWPOINT
      - How does the viewing angle differ? (e.g., "source faces camera, target shows
        right-side profile", "source is front-facing, target looks down from above")
      - Describe as spatial relations, NOT degrees.

  (b) SCALE
      - How does object size differ? (e.g., "target appears noticeably larger",
        "target fills more of the frame")

  (c) DEFORMATION / ARTICULATION (skip if both objects are rigid)
      For each movable part, describe source vs target in pose language:
      - source state: current configuration (e.g., "front legs straight, head up")
      - target state: desired configuration (e.g., "front legs bent, head lowered")
      - delta: what needs to change (e.g., "bend front legs and lower head")

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
  Each step must be a moderate-small interpolation change — one clear visual adjustment.
  The desired step size is comparable to an object pose/view change across about
  8-10 frames in a video: visible progress, but still trackable.
  Major parts should remain individually identifiable from the previous frame.
  Avoid part identity swaps, abrupt disappearances, topology changes, or sudden jumps.

INSTRUCTION FORMAT — natural descriptive edit prompts (NOT numeric geometry):
  The "instruction" field goes directly to the image editor. Write it as a clear,
  single-action sentence describing what should visually change. Reference specific
  body parts and spatial directions.

  GOOD examples (use this style):
    "Have the cat turn its head and upper body slightly toward the right side of the image."
    "Make the cat lower its front paws and tuck its hind legs closer to its body."
    "Have the car turn away from the camera to reveal more of its left side."
    "Make the cat appear slightly larger while keeping it centered in the frame."
    "Have the cat arch its back and raise its tail slightly upward."
    "Make the person bend forward at the waist while keeping their feet in place."

  BAD examples (do NOT use):
    "Rotate the object 90 degrees clockwise"          ← editor does not understand degrees
    "Rotate ~15° within the image plane"              ← numeric angles fail
    "Scale up by 20%"                                 ← percentages fail
    "Adjust the object forward"                       ← too vague
    "Change the pose"                                 ← too vague
    "Make it look like the target"                    ← not actionable

  Also write "expected_change" in the same natural descriptive style (no degrees/%).

HARD CONSTRAINTS (enforce on every step):
  - Do NOT change: object category, texture, color, material, surface details
  - Shared parts and major source-visible parts must remain trackable across consecutive steps
  - Do NOT plan horizontal flip or whole-image mirror transforms (handled in pre-align)
  - Do NOT plan coarse whole-object rotation unless fixing a small residual tilt
  - Use identifiable parts in instructions (head, tail, paws, wheels, hands, etc.)
  - Do NOT use degree numbers, percentages, or ratio values in "instruction"
  - Do NOT perform two dominant transforms in one step
  - Each step should make visible progress toward ENDPOINT geometry without sacrificing trackability

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
      {{"source_part": "...", "target_part": "...", "analogous": true, "visible_on_both": true}}
    ],
    "shared_parts": [
      {{
        "part_name": "head",
        "source_label": "head",
        "target_label": "head",
        "should_remain_trackable": true
      }}
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
      "instruction": "natural descriptive edit prompt sent to the image editor",
      "transform_type": "viewpoint_azimuth | viewpoint_elevation | inplane_rotation | scale | articulation | translation | fine_adjustment",
      "affected_parts": ["..."],
      "expected_change": "same natural-language description of what should visually change",
      "magnitude_estimate": "small | moderate (qualitative only, no numbers)",
      "cumulative_progress": 0.0,
      "identity_warning": "none | check_texture | check_silhouette"
    }}
  ]
}}"""

VERIFY_SYSTEM = """You are an image editing quality verifier.
Answer each checklist question with exactly 'Yes' or 'No', then provide the requested numeric scores.
Be strict and precise."""

VERIFY_USER = """SOURCE → BEFORE → AFTER edit sequence.
IMAGE 1 = BEFORE this step
IMAGE 2 = AFTER this step
IMAGE 3 = ENDPOINT configuration (final reference)

Important: IMAGE 3 is only a geometry/pose reference. Do not require the edited
object to adopt the ENDPOINT object's identity, category, texture, color, or material.
The edited object must preserve the SOURCE identity and move closer to ENDPOINT
geometry/configuration through a trackable interpolation step.

This step's instruction: "{instruction}"
Transform type: {transform_type}
Expected change: {expected_change}
{shared_parts_block}
Answer Yes or No for each:
1. Was the instructed geometric change applied? (viewpoint/scale/pose as specified)
2. Is the source object's identity preserved? (same category, texture, color, material)
3. Is the change size comparable to a short video temporal gap (about 8-10 frames),
   with visible progress but no abrupt jump?
4. Are major object parts still trackable from IMAGE 1 to IMAGE 2, with no part swaps,
   sudden disappearances, or topology changes?
5. After this edit, is the SOURCE object's pose, size, and rotation/orientation
   closer to the ENDPOINT's pose, size, and rotation (geometry only — not identity)?
{shared_parts_question}
Format:
1. Yes/No
2. Yes/No
3. Yes/No
4. Yes/No
5. Yes/No
{shared_parts_format}

Then provide diagnostic scores for offline analysis only. These scores do NOT affect
editing acceptance. Use 0.0-5.0 where 5.0 is best:
Scores:
- interpolation_quality: did IMAGE 2 form a useful, trackable intermediate sample
  from IMAGE 1, with a moderate-small step size?
- target_geometry_match: how well does IMAGE 2 move toward IMAGE 3 pose, camera
  angle, size/framing, and visible part layout?
- source_identity_preservation: how well does IMAGE 2 preserve IMAGE 1 identity,
  category, texture, color, material, and surface details?
- overall_quality: overall quality of this edited step for the progressive trajectory.

Score format:
interpolation_quality: 0.0
target_geometry_match: 0.0
source_identity_preservation: 0.0
overall_quality: 0.0"""

VERIFY_SHARED_PARTS_BLOCK = """
Shared parts that should remain trackable across this transition (present on BOTH
source and endpoint geometry). If endpoint geometry naturally occludes a part, the
visibility change should be gradual rather than abrupt:
{shared_parts_list}
"""

VERIFY_SHARED_PARTS_QUESTION = (
    "6. Are the listed shared parts still identifiable/trackable in IMAGE 2, "
    "unless they are naturally and gradually becoming occluded by endpoint geometry?"
)

VERIFY_NO_SHARED_PARTS_BLOCK = ""
VERIFY_NO_SHARED_PARTS_QUESTION = ""
VERIFY_NO_SHARED_PARTS_FORMAT = ""

PREALIGN_VERIFY_SYSTEM = """You are a strict pre-alignment geometry verifier.
Answer with valid JSON only."""

PREALIGN_VERIFY_USER = """Images:
IMAGE 1 = ORIGINAL SOURCE
IMAGE 2 = ALIGNED SOURCE after deterministic flip/rotation
IMAGE 3 = TARGET geometry reference

Pre-align decision:
{decision_json}

Verify only whether this deterministic coarse orientation alignment is useful.
Do NOT evaluate identity/texture/category: flip and rotation are deterministic and do not edit identity.

Return this exact JSON:
{{
  "overall_ok": true,
  "closer_to_target_orientation": true,
  "landmark_axis_improved": true,
  "not_over_transformed": true,
  "recommendation": "apply | rollback | retry",
  "failure_reason": "short reason if not overall_ok"
}}

Rules:
- overall_ok=true only if ALIGNED SOURCE is a better starting point than ORIGINAL SOURCE.
- If no transform was applied because no reliable shared two-landmark axis exists, accept it when ORIGINAL SOURCE is the safer starting point.
- rollback means the transform made orientation worse.
- retry means the decision likely chose poor landmarks or wrong flip/rotation.
- Do NOT check cropping: transforms are flip/rotate only and cannot crop content."""

PREALIGN_BRUTEFORCE_SYSTEM = """You are a coarse orientation matcher.
Answer with valid JSON only."""

PREALIGN_BRUTEFORCE_USER = """Images:
IMAGE 1 = GRID of {num_candidates} unique deterministic source transforms (IDs 0-{max_candidate_id})
IMAGE 2 = TARGET geometry reference

Candidate transforms:
{candidate_table}

Pick the candidate whose coarse orientation, viewpoint, and layout best match TARGET.
Ignore texture/color/identity differences; focus on geometry and viewing direction.

Return this exact JSON:
{{
  "best_candidate_id": 0,
  "reason": "short reason"
}}

Rules:
- best_candidate_id must be an integer in [0, {max_candidate_id}].
- Prefer the candidate that would need the smallest later pose edit to reach TARGET."""

SGOAL_EDIT_PROMPT = """Edit IMAGE 1 so that its object matches IMAGE 2's pose, camera view, scale, and visible part layout.

Hard requirements:
- Most important: match IMAGE 2 geometry as closely as possible (pose, camera angle, scale/framing, visible part layout).
- Preserve IMAGE 1 object's identity, category, texture, color, material, and surface details.
- Use IMAGE 2 only as geometry reference for pose, camera angle, size/framing, and visible part layout.
- Do NOT copy IMAGE 2 object's identity, texture, color, material, or category-specific appearance.
- It is okay if parts visible in IMAGE 1 become occluded when that matches IMAGE 2's visible geometry.

Return a single edited version of IMAGE 1."""

SGOAL_VERIFY_SYSTEM = """You are a strict verifier for a source-identity goal image.
Answer with valid JSON only."""

SGOAL_VERIFY_USER = """Images:
IMAGE 1 = PRE-ALIGNED SOURCE
IMAGE 2 = S_GOAL candidate, edited from IMAGE 1
IMAGE 3 = TARGET geometry reference

Goal:
Decide whether IMAGE 2 is a good endpoint for later progressive editing from IMAGE 1.
IMAGE 2 should preserve IMAGE 1 identity while matching IMAGE 3 geometry.
Endpoint geometry match is the highest priority: pose, camera angle, scale/framing,
and visible part layout should match IMAGE 3. Reachability from IMAGE 1 is useful
but secondary; do not reject a strong geometry match only because the path may be hard.

Objective identity score DINO(IMAGE 1, IMAGE 2): {identity_score:.3f}
Required minimum identity score: {identity_threshold:.3f}

Return this exact JSON:
{{
  "overall_ok": true,
  "pose_match": true,
  "camera_match": true,
  "scale_match": true,
  "identity_preserved": true,
  "not_target_copy": true,
  "target_visible_layout_match": true,
  "no_severe_artifacts": true,
  "scores": {{
    "interpolation_quality": 0.0,
    "target_geometry_match": 0.0,
    "source_identity_preservation": 0.0,
    "overall_quality": 0.0
  }},
  "recommendation": "use_as_goal | retry | fallback_to_target_path",
  "failure_reason": "short reason if not overall_ok"
}}

Rules:
- pose_match: IMAGE 2 object articulation/part layout should resemble TARGET geometry.
- camera_match: IMAGE 2 should resemble TARGET viewpoint/azimuth/elevation.
- scale_match: IMAGE 2 should resemble TARGET object scale/framing.
- identity_preserved: IMAGE 2 should still look like IMAGE 1's object.
- not_target_copy: IMAGE 2 must not copy TARGET identity/texture/material/category appearance.
- target_visible_layout_match: parts visible in TARGET should be visible and arranged similarly in IMAGE 2.
- Do not force IMAGE 1-only parts to remain visible if TARGET naturally occludes them.
- no_severe_artifacts: reject only severe geometry-breaking artifacts that would harm matching/tracking.
- scores are for offline analysis only and do not change the hard pass/fail checks.
- interpolation_quality for S_GOAL means whether IMAGE 2 is a useful endpoint candidate
  for later progressive interpolation from IMAGE 1.
- target_geometry_match scores pose, camera angle, scale/framing, and visible part layout.
- source_identity_preservation scores preservation of IMAGE 1 identity/category/texture/material.
- overall_quality is the overall S_GOAL endpoint quality.
- overall_ok=true only if all hard checks are true and identity score is above threshold."""

PLANNING_VERIFY_SYSTEM = """You are a strict editing-plan auditor.
Answer with valid JSON only."""

PLANNING_VERIFY_USER = """Images:
IMAGE 1 = CURRENT SOURCE after pre-align
IMAGE 2 = ENDPOINT geometry reference

Plan JSON:
{plan_json}

Audit the plan BEFORE any image editing.

Required policy:
- Pose/deformation/articulation steps must come before angle/size camera steps.
- MotionEdit pose steps use natural descriptive edit prompts.
- Angle/size steps must use Qwen angle-LoRA prompt format exactly:
  <sks> [azimuth] [elevation] [distance]
- Angle/size prompts must use only:
  azimuth: front view | front-right quarter view | right side view | back-right quarter view | back view | back-left quarter view | left side view | front-left quarter view
  elevation: low-angle shot | eye-level shot | elevated shot | high-angle shot
  distance: close-up | medium shot | wide shot
- Skip pose stage if no pose/deformation is needed.
- Skip angle/size stage if source/current already matches endpoint camera and distance.
- Every step must preserve source identity/category/texture/material.
- Step sizes should be moderate-small, like object pose/view change over about 8-10 video frames.
- Steps should make visible progress while keeping major parts trackable between consecutive frames.
- Shared parts should remain identifiable/trackable unless endpoint geometry gradually occludes them.
- The plan must preserve source identity and must not copy non-source identity.

Return this exact JSON:
{{
  "overall_ok": true,
  "ordering_ok": true,
  "step_count_ok": true,
  "prompt_format_ok": true,
  "identity_constraints_ok": true,
  "shared_parts_ok": true,
  "failure_reason": "short reason if not overall_ok",
  "revision_hint": "specific instruction for planner if failed"
}}"""

DIAGNOSIS_SYSTEM = """You are a visual geometry diagnostician for staged image editing.
Given SOURCE and ENDPOINT images, classify pose/deformation need and camera/size gap.
Output valid JSON only."""

DIAGNOSIS_USER = """Images:
IMAGE 1 = CURRENT SOURCE after pre-align
IMAGE 2 = ENDPOINT geometry reference only

Goal:
Decide whether to run:
1) a MotionEdit pose/deformation stage, then
2) a Qwen angle-LoRA camera/size stage.

Use only visible image evidence. Do not use dataset annotations.
The ENDPOINT geometry match is the final priority. The later progressive steps should
split this gap into trackable interpolation changes, roughly like 8-10 video-frame
pose/view differences per step.

Camera prompt vocabulary:
- azimuth: front view | front-right quarter view | right side view | back-right quarter view | back view | back-left quarter view | left side view | front-left quarter view
- elevation: low-angle shot | eye-level shot | elevated shot | high-angle shot
- distance: close-up | medium shot | wide shot
Angle-LoRA prompt format:
<sks> [azimuth] [elevation] [distance]

Pose step policy:
- rigid object with no movable-pose change: pose_steps=0
- deformable object: small=1, medium=2-3, large=3-4
- articulated object: small=1-2, medium=3-4, large=5-6
- Increase steps when independent moving part groups are many.
- Cap pose_steps at {max_pose_steps}.

Return this exact JSON:
{{
  "analysis": {{
    "source_object": {{"description": "...", "deformability": "rigid | articulated | deformable"}},
    "target_object": {{"description": "...", "deformability": "rigid | articulated | deformable"}},
    "part_mapping": [
      {{"source_part": "...", "target_part": "...", "analogous": true, "visible_on_both": true}}
    ],
    "shared_parts": [
      {{"part_name": "...", "source_label": "...", "target_label": "...", "should_remain_trackable": true}}
    ],
    "transform_gaps": {{
      "viewpoint": {{"azimuth": "...", "elevation": "...", "inplane_rotation": "..."}},
      "scale": "...",
      "articulation": [
        {{"part": "...", "source_state": "...", "target_state": "...", "delta": "..."}}
      ],
      "translation": "...",
      "dominant_axes": ["pose", "angle", "size"],
      "no_change_axes": []
    }}
  }},
  "stage_plan": {{
    "object_motion_type": "rigid | articulated | deformable",
    "pose_gap": "none | small | medium | large",
    "independent_moving_part_groups": ["head/neck", "torso", "front legs"],
    "pose_steps": 0,
    "run_pose_stage": false,
    "source_camera": {{
      "azimuth": "front view",
      "elevation": "eye-level shot",
      "distance": "medium shot"
    }},
    "target_camera": {{
      "azimuth": "right side view",
      "elevation": "elevated shot",
      "distance": "close-up"
    }},
    "run_angle_stage": true,
    "reason": "short explanation"
  }}
}}"""

POSE_PLANNING_SYSTEM = """You are a MotionEdit pose/deformation planner.
Output valid JSON only."""

POSE_PLANNING_USER = """Images:
IMAGE 1 = CURRENT SOURCE after pre-align
IMAGE 2 = ENDPOINT geometry reference only

Stage diagnosis:
{diagnosis_json}

Produce exactly {pose_steps} MotionEdit pose/deformation steps.
These steps are for pose, articulation, deformation, and part layout only.
Do NOT include camera viewpoint, angle-LoRA prompts, close-up/medium/wide shot, or size/framing changes.

Instruction style:
- Natural descriptive edit prompt.
- Use body/object parts and spatial relations.
- Preserve source identity/category/texture/material.
- Each step should be a moderate-small trackable interpolation, like an 8-10 frame
  pose change in a video: visible progress, no abrupt jump.
- Keep major parts identifiable across consecutive steps; avoid part swaps, sudden
  disappearances, or topology changes.
- If ENDPOINT geometry occludes a source-visible part, introduce that visibility
  change gradually.
- One dominant pose/deformation change per step.

Return:
{{
  "steps": [
    {{
      "step": 1,
      "instruction": "natural MotionEdit prompt",
      "transform_type": "articulation | fine_adjustment",
      "affected_parts": ["..."],
      "expected_change": "natural-language expected pose change",
      "magnitude_estimate": "small | moderate",
      "cumulative_progress": 0.0,
      "identity_warning": "none | check_texture | check_silhouette"
    }}
  ]
}}"""

REPLAN_SYSTEM = """You are a motion editing planner.
A previous editing step failed verification.
Analyze the failure and produce a corrected instruction.
The instruction goes directly to an image editing model — use natural descriptive
action language (body parts, spatial directions), NOT degrees or percentages.
Output valid JSON only."""

REPLAN_USER = """IMAGE 1 = BEFORE edit
IMAGE 2 = AFTER edit (FAILED)
IMAGE 3 = ENDPOINT configuration (final reference)

Important: IMAGE 3 is only a geometry/pose reference. The corrected instruction
must preserve SOURCE identity/category/texture/material and must not ask the editor
to copy ENDPOINT identity.

Original instruction: "{failed_instruction}"
Transform type: {transform_type}
Expected change: {expected_change}
Identity warning: {identity_warning}
Failure reason: {failure_reason}
Shared parts that should stay trackable across the corrected transition: {shared_parts_list}

Write the corrected instruction as a natural edit prompt, e.g.:
  "Have the cat turn its head and body slightly toward the left."
  "Make the cat lower its front paws while keeping its hind legs in place."
Do NOT use degree numbers or percentages.
The corrected edit should be a moderate-small trackable interpolation step, similar
to about 8-10 frames of object motion in a video. Major parts should remain trackable
from IMAGE 1 to the corrected edit, with no part swaps or abrupt disappearances.

Analyze why it failed and provide a CORRECTED instruction:
{{
  "instruction": "natural descriptive edit prompt",
  "transform_type": "...",
  "affected_parts": ["..."],
  "expected_change": "natural-language description of visual change",
  "magnitude_estimate": "small | moderate",
  "cumulative_progress": ...,
  "identity_warning": "none | check_texture | check_silhouette"
}}"""


TRAJECTORY_VERIFY_SYSTEM = """You are a trajectory quality verifier for sparse keyframe editing.

You will see an ordered sequence of edited SOURCE images (starting from a pre-aligned
source) plus an ENDPOINT image at the end as a geometry reference only.

CRITICAL:
  - The edited object must keep the SOURCE identity (category, texture, color, material)
    throughout every keyframe. Never expect it to look like the ENDPOINT object.
  - Judge only whether SOURCE geometry — pose, size, rotation/orientation, and analogous
    part layout — progressively moves toward the ENDPOINT's pose, size, and rotation.
  - The final frame should match ENDPOINT geometry well. Intermediate steps should be
    trackable interpolation frames, each roughly like an 8-10 video-frame motion gap.

These are sparse intermediate keyframes (not a dense video). Moderate-small jumps are
OK if major parts remain trackable. Flag abrupt geometry reversals, part swaps,
topology changes, or sudden disappearances."""

TRAJECTORY_VERIFY_USER = """Ordered SOURCE editing trajectory:
{step_image_notes}
Final image = ENDPOINT (geometry reference ONLY for pose, size, rotation — NOT identity).

The object in every step must remain the SOURCE object. Do not penalize the sequence
for failing to match ENDPOINT identity, texture, color, or category.

Shared SOURCE parts (also present on ENDPOINT geometry) that should stay trackable:
{shared_parts_list}

Answer Yes or No for each:
1. Does the SOURCE object progressively change its pose, size, and rotation/orientation
   toward the ENDPOINT's pose, size, and rotation across the sequence, with the final
   frame closely matching ENDPOINT geometry?
   (Geometry only. SOURCE identity must stay the same.)
2. Are there abrupt non-progressive geometry jumps, reversals, part swaps, topology
   changes, or sudden disappearances between any consecutive steps?
3. Are listed shared parts and major object parts trackable across consecutive frames,
   unless they gradually become occluded by ENDPOINT geometry?

Format:
1. Yes/No
2. Yes/No
3. Yes/No

Then provide diagnostic scores for offline analysis only. These scores do NOT affect
editing acceptance. Use 0.0-5.0 where 5.0 is best:
Scores:
- interpolation_quality: how smooth/trackable/useful are the intermediate samples
  as a sparse progressive trajectory?
- target_geometry_match: how well does the final SOURCE object match ENDPOINT pose,
  camera angle, size/framing, and visible part layout?
- source_identity_preservation: how well is SOURCE identity/category/texture/material
  preserved across the whole trajectory?
- overall_quality: overall trajectory quality.

Score format:
interpolation_quality: 0.0
target_geometry_match: 0.0
source_identity_preservation: 0.0
overall_quality: 0.0"""


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
class PreAlignDecision:
    should_horizontal_flip: bool
    rotation_degrees: float
    confidence: float
    reason: str
    object_category: str = ""
    primary_landmark: str = ""
    secondary_landmark: str = ""
    anchor_features: list[dict[str, Any]] = field(default_factory=list)
    horizontal_layout_match: str = ""
    rotation_basis: str = ""
    flip_votes_flip: int = 0
    flip_votes_no_flip: int = 0
    applied_horizontal_flip: bool = False
    applied_vertical_flip: bool = False
    applied_rotation_degrees: float = 0.0


@dataclass
class PreAlignVerifyResult:
    overall_ok: bool
    recommendation: str
    failure_reason: Optional[str] = None


@dataclass
class SGoalVerifyResult:
    overall_ok: bool
    pose_match: bool
    camera_match: bool
    scale_match: bool
    identity_preserved: bool
    not_target_copy: bool
    target_visible_layout_match: bool
    no_severe_artifacts: bool
    recommendation: str
    identity_score: float
    identity_threshold: float
    scores: dict[str, float] = field(default_factory=dict)
    failure_reason: Optional[str] = None


@dataclass
class CameraPose:
    azimuth: str
    elevation: str
    distance: str

    @property
    def prompt(self) -> str:
        return f"<sks> {self.azimuth} {self.elevation} {self.distance}"


@dataclass
class StagePlan:
    object_motion_type: str
    pose_gap: str
    independent_moving_part_groups: list[str]
    pose_steps: int
    run_pose_stage: bool
    source_camera: CameraPose
    target_camera: CameraPose
    run_angle_stage: bool
    reason: str = ""


@dataclass
class Analysis:
    source_description: str
    source_deformability: str
    target_description: str
    target_deformability: str
    part_mapping: list[dict[str, Any]]
    shared_parts: list[str]
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
    physically_plausible: bool
    no_artifacts: bool
    closer_to_target: bool
    shared_parts_visible: bool = True
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class VerifyResult:
    flow_direction_ok: bool
    flow_magnitude_ok: bool
    identity_ok: bool
    texture_ok: bool
    silhouette_ok: bool
    semantic_ok: bool
    shared_parts_ok: bool
    overall_ok: bool
    scores: dict[str, float] = field(default_factory=dict)
    failure_reason: Optional[str] = None


@dataclass
class AdjacentFlowMetrics:
    from_step: int
    to_step: int
    mean_magnitude: float
    direction: tuple[float, float]


@dataclass
class TrajectoryFlowVerifyResult:
    pair_metrics: list[AdjacentFlowMetrics]
    smooth_ok: bool
    issues: list[str]


@dataclass
class TrajectoryVLMVerifyResult:
    progressive_toward_target: bool
    abrupt_jumps: bool
    shared_parts_visible: bool
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryVerifyResult:
    flow: TrajectoryFlowVerifyResult
    vlm: Optional[TrajectoryVLMVerifyResult]
    overall_ok: bool
    failure_reason: Optional[str] = None
    suspect_steps: list[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    source_img: Image.Image
    target_img: Image.Image
    analysis: Analysis
    trajectory: list[Image.Image]
    instructions: list[SubInstruction]
    verify_results: list[VerifyResult]
    final_img: Image.Image
    pre_alignment: Optional[PreAlignDecision] = None
    s_goal: Optional[dict[str, Any]] = None
    stage_plan: Optional[StagePlan] = None
    planning_verify: Optional[dict[str, Any]] = None
    trajectory_verify: Optional[TrajectoryVerifyResult] = None
    trajectory_repair: Optional[list[dict[str, Any]]] = None


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


def parse_yes_no(text: str, num_questions: int = 6) -> dict[int, bool]:
    answers: dict[int, bool] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*[\.\):\-]\s*(yes|no)\b", line, re.IGNORECASE)
        if match:
            answers[int(match.group(1))] = match.group(2).lower() == "yes"
    if len(answers) < num_questions:
        tokens = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        for idx, token in enumerate(tokens[:num_questions], start=1):
            answers.setdefault(idx, token.lower() == "yes")
    return answers


SCORE_KEYS = (
    "interpolation_quality",
    "target_geometry_match",
    "source_identity_preservation",
    "overall_quality",
)


def _clip_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def parse_score_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {key: _clip_score(raw.get(key, 0.0)) for key in SCORE_KEYS if key in raw}


def parse_scores(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in SCORE_KEYS:
        match = re.search(rf"\b{re.escape(key)}\s*:\s*([0-5](?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            scores[key] = _clip_score(match.group(1))
    return scores


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    """Parse VLM JSON fields that should be integers but may arrive as str/list."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return max(0, int(float(text)))
        except ValueError:
            match = re.search(r"\d+", text)
            return max(0, int(match.group(0))) if match else default
    if isinstance(value, (list, tuple)):
        nums = [_coerce_non_negative_int(item, default=-1) for item in value]
        nums = [num for num in nums if num >= 0]
        return max(nums) if nums else default
    return default


def extract_shared_parts(analysis_dict: dict[str, Any]) -> list[str]:
    """Parts visible on both source and endpoint that should stay trackable."""
    raw = analysis_dict.get("shared_parts") or []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            label = (
                item.get("part_name")
                or item.get("name")
                or item.get("source_label")
                or item.get("target_label")
            )
            if label and item.get("should_remain_trackable", item.get("must_remain_visible", True)):
                names.append(str(label).strip())
        elif item:
            names.append(str(item).strip())

    if names:
        return names

    for mapping in analysis_dict.get("part_mapping") or []:
        if not mapping.get("analogous", False):
            continue
        if mapping.get("visible_on_both") is False:
            continue
        label = mapping.get("source_part") or mapping.get("target_part")
        if label:
            names.append(str(label).strip())
    return names


def format_shared_parts_list(shared_parts: list[str]) -> str:
    if not shared_parts:
        return "(none identified)"
    return ", ".join(shared_parts)


def build_verify_prompt(
    instruction: str,
    transform_type: str,
    expected_change: str,
    shared_parts: list[str],
) -> str:
    if shared_parts:
        shared_parts_block = VERIFY_SHARED_PARTS_BLOCK.format(
            shared_parts_list=format_shared_parts_list(shared_parts),
        )
        shared_parts_question = VERIFY_SHARED_PARTS_QUESTION
        shared_parts_format = "6. Yes/No"
    else:
        shared_parts_block = VERIFY_NO_SHARED_PARTS_BLOCK
        shared_parts_question = VERIFY_NO_SHARED_PARTS_QUESTION
        shared_parts_format = VERIFY_NO_SHARED_PARTS_FORMAT

    return VERIFY_USER.format(
        instruction=instruction,
        transform_type=transform_type,
        expected_change=expected_change,
        shared_parts_block=shared_parts_block,
        shared_parts_question=shared_parts_question,
        shared_parts_format=shared_parts_format,
    )


def build_analysis(parsed: dict[str, Any]) -> Analysis:
    analysis = parsed["analysis"]
    source = analysis.get("source_object", {})
    target = analysis.get("target_object", {})
    gaps = analysis.get("transform_gaps", {})
    viewpoint = gaps.get("viewpoint", {})
    shared_parts = extract_shared_parts(analysis)
    return Analysis(
        source_description=source.get("description", ""),
        source_deformability=source.get("deformability", ""),
        target_description=target.get("description", ""),
        target_deformability=target.get("deformability", ""),
        part_mapping=analysis.get("part_mapping", []),
        shared_parts=shared_parts,
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
        step=_coerce_non_negative_int(raw_step.get("step", fallback_step), default=fallback_step),
        instruction=str(raw_step["instruction"]),
        transform_type=str(raw_step["transform_type"]),
        affected_parts=list(raw_step.get("affected_parts", ["object"])),
        expected_change=str(raw_step.get("expected_change", raw_step["instruction"])),
        magnitude_estimate=str(raw_step.get("magnitude_estimate", "")),
        cumulative_progress=float(raw_step.get("cumulative_progress", 0.0)),
        identity_warning=str(raw_step.get("identity_warning", "none")),
    )


def _nearest_choice(value: Any, choices: list[str], default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    for choice in choices:
        if text == choice or choice in text:
            return choice
    for choice in choices:
        if any(part in text for part in choice.split()):
            return choice
    return default


def build_camera_pose(raw: dict[str, Any]) -> CameraPose:
    return CameraPose(
        azimuth=_nearest_choice(raw.get("azimuth"), AZIMUTH_BINS, "front view"),
        elevation=_nearest_choice(raw.get("elevation"), ELEVATION_BINS, "eye-level shot"),
        distance=_nearest_choice(raw.get("distance"), DISTANCE_BINS, "medium shot"),
    )


def build_stage_plan(parsed: dict[str, Any], max_pose_steps: int) -> StagePlan:
    raw = parsed.get("stage_plan", {})
    object_motion_type = str(raw.get("object_motion_type", "rigid")).lower()
    pose_gap = str(raw.get("pose_gap", "none")).lower()
    groups = [str(item) for item in raw.get("independent_moving_part_groups", []) if item]
    requested_pose_steps = _coerce_non_negative_int(raw.get("pose_steps", 0))

    if object_motion_type == "rigid" or pose_gap == "none":
        pose_steps = 0
    else:
        pose_steps = requested_pose_steps
        if pose_steps <= 0:
            base = 2 if object_motion_type == "articulated" else 1
            gap_bonus = {"small": 0, "medium": 2, "large": 3}.get(pose_gap, 1)
            group_bonus = 2 if len(groups) >= 5 else (1 if len(groups) >= 3 else 0)
            pose_steps = base + gap_bonus + group_bonus
    pose_steps = max(0, min(max_pose_steps, pose_steps))

    source_camera = build_camera_pose(raw.get("source_camera", {}))
    target_camera = build_camera_pose(raw.get("target_camera", {}))
    run_angle_stage = bool(raw.get("run_angle_stage", True)) and source_camera != target_camera
    return StagePlan(
        object_motion_type=object_motion_type,
        pose_gap=pose_gap,
        independent_moving_part_groups=groups,
        pose_steps=pose_steps,
        run_pose_stage=pose_steps > 0,
        source_camera=source_camera,
        target_camera=target_camera,
        run_angle_stage=run_angle_stage,
        reason=str(raw.get("reason", "")),
    )


def _circular_path(start: str, end: str, bins: list[str]) -> list[str]:
    start_idx = bins.index(start)
    end_idx = bins.index(end)
    if start_idx == end_idx:
        return [start]
    forward = (end_idx - start_idx) % len(bins)
    backward = (start_idx - end_idx) % len(bins)
    if forward <= backward:
        return [bins[(start_idx + idx) % len(bins)] for idx in range(forward + 1)]
    return [bins[(start_idx - idx) % len(bins)] for idx in range(backward + 1)]


def _linear_path(start: str, end: str, bins: list[str]) -> list[str]:
    start_idx = bins.index(start)
    end_idx = bins.index(end)
    if start_idx == end_idx:
        return [start]
    step = 1 if end_idx > start_idx else -1
    return [bins[idx] for idx in range(start_idx, end_idx + step, step)]


def generate_angle_steps(stage_plan: StagePlan, start_step: int) -> list[SubInstruction]:
    if not stage_plan.run_angle_stage:
        return []

    az_path = _circular_path(stage_plan.source_camera.azimuth, stage_plan.target_camera.azimuth, AZIMUTH_BINS)
    el_path = _linear_path(stage_plan.source_camera.elevation, stage_plan.target_camera.elevation, ELEVATION_BINS)
    dist_path = _linear_path(stage_plan.source_camera.distance, stage_plan.target_camera.distance, DISTANCE_BINS)
    n_steps = max(len(az_path), len(el_path), len(dist_path)) - 1
    if n_steps <= 0:
        return []

    steps: list[SubInstruction] = []
    for idx in range(1, n_steps + 1):
        az = az_path[min(idx, len(az_path) - 1)]
        el = el_path[min(idx, len(el_path) - 1)]
        dist = dist_path[min(idx, len(dist_path) - 1)]
        camera = CameraPose(azimuth=az, elevation=el, distance=dist)
        step_num = start_step + idx - 1
        steps.append(
            SubInstruction(
                step=step_num,
                instruction=camera.prompt,
                transform_type="angle_size",
                affected_parts=["object"],
                expected_change=f"change camera/framing toward {camera.prompt}",
                magnitude_estimate="small",
                cumulative_progress=float(idx / n_steps),
                identity_warning="check_texture",
            )
        )
    return steps


# ============================================================
# MODEL WRAPPERS
# ============================================================


def _require_transformers_for_qwen3_vl() -> None:
    """Qwen3-VL needs transformers>=4.57; transformers 5.x breaks peft HybridCache."""
    import transformers
    from packaging.version import Version

    installed = Version(transformers.__version__)
    if not (Version("4.57.0") <= installed < Version("5.0.0")):
        raise ImportError(
            f"Need transformers in [4.57.0, 5.0.0) for Qwen3-VL + MotionEdit, "
            f"but found {transformers.__version__}. Run:\n"
            "  bash tools/fix_dependencies.sh"
        )


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

        _require_transformers_for_qwen3_vl()
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()
        self._device = next(self.model.parameters()).device

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
        inputs = inputs.to(self._device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _ensure_diffusers_torchao_compat() -> None:
    """diffusers 0.36 breaks when torchao>=0.16 (logger/uint4 import bug)."""
    try:
        import diffusers
    except ImportError:
        return

    from packaging.version import Version

    installed = Version(diffusers.__version__)
    if installed < Version("0.37.0"):
        raise ImportError(
            f"diffusers {diffusers.__version__} is incompatible with torchao>=0.16.\n"
            "Upgrade with:\n"
            '  pip install "diffusers>=0.37.0,<0.39.0"'
        )


def _ensure_peft_torchao_compat() -> None:
    """peft LoRA loading needs torchao>=0.16; torchao>=0.17 needs torch>=2.11."""
    try:
        import torchao
    except ImportError:
        return

    from packaging.version import Version

    installed = Version(torchao.__version__)
    torch_ver = Version(torch.__version__.split("+")[0])

    if installed >= Version("0.17.0") and torch_ver < Version("2.11.0"):
        raise ImportError(
            f"torchao {torchao.__version__} requires torch>=2.11, "
            f"but found {torch.__version__}.\n"
            "Run:\n"
            "  python tools/repair_torch_stack.py --strategy motionedit"
        )
    if installed < Version("0.16.0"):
        raise ImportError(
            f"torchao {torchao.__version__} is too old for peft LoRA loading "
            f"(need >=0.16.0). Run:\n"
            "  python tools/repair_torch_stack.py"
        )


def _check_torch_vision_compat() -> None:
    """torch and torchvision must be installed as a matched pair."""
    try:
        import torchvision
        from torchvision.transforms import InterpolationMode  # noqa: F401
    except Exception as exc:
        tv_ver = "unknown"
        try:
            import torchvision as tv

            tv_ver = tv.__version__
        except Exception:
            pass
        raise ImportError(
            f"torch/torchvision mismatch (torch={torch.__version__}, "
            f"torchvision={tv_ver}).\n"
            "Run:\n"
            "  python tools/repair_torch_stack.py --strategy match-current\n"
            "or:\n"
            "  STRATEGY=match-current bash tools/fix_dependencies.sh"
        ) from exc


def _check_runtime_dependencies() -> None:
    """Validate the inference dependency stack before loading any models."""
    _check_torch_vision_compat()
    _require_transformers_for_qwen3_vl()
    _ensure_peft_torchao_compat()
    _ensure_diffusers_torchao_compat()
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        raise ImportError(
            "Cannot import AutoProcessor from transformers.\n"
            "Common causes:\n"
            "  1) torch/torchvision version mismatch\n"
            "  2) broken transformers install\n"
            "  3) incompatible torchao version\n"
            "Fix with:\n"
            "  python tools/repair_torch_stack.py --strategy match-current"
        ) from exc
    import torchvision

    log(f"[deps] torch {torch.__version__}, torchvision {torchvision.__version__}")


class MotionNFTEditor:
    """Qwen-Image-Edit executor with MotionEdit and optional angle-control LoRAs."""

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str],
        angle_lora_path: Optional[str],
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
        self.loaded_adapters: set[str] = set()

        _ensure_peft_torchao_compat()
        if not lora_path:
            raise ValueError(
                "motionedit_lora_path is required. "
                f"Expected local adapter at {DEFAULT_MOTIONEDIT_LORA_PATH}"
            )
        self.pipe.load_lora_weights(
            lora_path,
            weight_name=MOTIONEDIT_LORA_WEIGHT,
            adapter_name="motion",
        )
        self.loaded_adapters.add("motion")

        if angle_lora_path:
            angle_dir = Path(angle_lora_path)
            angle_weight = angle_dir / ANGLE_LORA_WEIGHT
            if angle_weight.is_file():
                self.pipe.load_lora_weights(
                    str(angle_dir),
                    weight_name=ANGLE_LORA_WEIGHT,
                    adapter_name="angle",
                )
                self.loaded_adapters.add("angle")
            else:
                log(f"[load] Angle LoRA not found at {angle_weight}; angle stage will use motion adapter fallback.")

        self.pipe.set_adapters(["motion"], adapter_weights=[1.0])

    def _enable_lora_adapters(self) -> None:
        if hasattr(self.pipe, "enable_lora"):
            self.pipe.enable_lora()

    def _disable_lora_adapters(self) -> None:
        if hasattr(self.pipe, "disable_lora"):
            self.pipe.disable_lora()
        else:
            log("[edit] Pipeline has no disable_lora(); base-only S_goal may still use loaded adapters.")

    @torch.no_grad()
    def edit(
        self,
        source_img: Image.Image,
        instruction: str,
        step_seed: int,
        adapter: str = "motion",
    ) -> Image.Image:
        if adapter not in self.loaded_adapters:
            if adapter == "angle":
                log("[edit] Angle adapter unavailable; falling back to motion adapter.")
            adapter = "motion"
        self._enable_lora_adapters()
        self.pipe.set_adapters([adapter], adapter_weights=[1.0])
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

    @torch.no_grad()
    def edit_s_goal_base(
        self,
        source_img: Image.Image,
        target_img: Image.Image,
        instruction: str,
        step_seed: int,
    ) -> Image.Image:
        """Generate S_goal from [source, target] with base Qwen Edit Plus only."""
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(self.seed + step_seed)
        self._disable_lora_adapters()
        try:
            image = self.pipe(
                num_inference_steps=self.num_inference_steps,
                image=[source_img, target_img],
                prompt=instruction,
                negative_prompt=" ",
                true_cfg_scale=self.true_cfg_scale,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]
        finally:
            self._enable_lora_adapters()
            self.pipe.set_adapters(["motion"], adapter_weights=[1.0])
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


_HORIZONTAL_POS = {"left": -1, "center": 0, "right": 1}
_VERTICAL_POS = {"top": -1, "center": 0, "bottom": 1}


def _normalize_axis_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in _HORIZONTAL_POS:
        return label
    if label in _VERTICAL_POS:
        return label
    if label in {"middle", "centre", "mid"}:
        return "center"
    if label in {"l"}:
        return "left"
    if label in {"r"}:
        return "right"
    if label in {"t", "up", "upper"}:
        return "top"
    if label in {"b", "down", "lower"}:
        return "bottom"
    return ""


def _landmark_to_point(position: dict[str, Any]) -> Optional[tuple[float, float]]:
    horizontal = _normalize_axis_label(position.get("horizontal"))
    vertical = _normalize_axis_label(position.get("vertical"))
    if horizontal not in _HORIZONTAL_POS or vertical not in _VERTICAL_POS:
        return None
    return (_HORIZONTAL_POS[horizontal], _VERTICAL_POS[vertical])


def _landmark_present_on_both(anchor: dict[str, Any]) -> bool:
    if not anchor.get("comparable", True):
        return False
    if anchor.get("visible_on_source") is False or anchor.get("visible_on_target") is False:
        return False
    source_point = _landmark_to_point(anchor.get("source") or {})
    target_point = _landmark_to_point(anchor.get("target") or {})
    return source_point is not None and target_point is not None


def _bilateral_landmark_names(anchor_features: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for anchor in anchor_features:
        name = str(anchor.get("name", "")).strip()
        if name and _landmark_present_on_both(anchor):
            names.append(name)
    return names


def _resolve_landmark_pair(parsed: dict[str, Any]) -> tuple[str, str]:
    anchors = list(parsed.get("anchor_features") or [])
    bilateral_names = set(_bilateral_landmark_names(anchors))

    def _valid(name: str) -> bool:
        return bool(name) and name in bilateral_names

    axis = parsed.get("alignment_axis") or {}
    primary = str(axis.get("primary_landmark", "")).strip()
    secondary = str(axis.get("secondary_landmark", "")).strip()
    if _valid(primary) and _valid(secondary):
        return primary, secondary

    by_role: dict[str, str] = {}
    for anchor in anchors:
        role = str(anchor.get("role", "")).strip().lower()
        name = str(anchor.get("name", "")).strip()
        if role in {"primary", "secondary"} and _valid(name):
            by_role[role] = name
    if "primary" in by_role and "secondary" in by_role:
        return by_role["primary"], by_role["secondary"]

    shared = _bilateral_landmark_names(anchors)
    if len(shared) >= 2:
        return shared[0], shared[1]
    return "", ""


def _find_landmark_anchor(
    anchor_features: list[dict[str, Any]],
    landmark_name: str,
) -> Optional[dict[str, Any]]:
    target_name = landmark_name.strip().lower()
    for anchor in anchor_features:
        if str(anchor.get("name", "")).strip().lower() == target_name:
            return anchor
    return None


def _find_landmark_positions(
    anchor_features: list[dict[str, Any]],
    landmark_name: str,
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    anchor = _find_landmark_anchor(anchor_features, landmark_name)
    if anchor is None or not _landmark_present_on_both(anchor):
        return None, None
    source_point = _landmark_to_point(anchor.get("source") or {})
    target_point = _landmark_to_point(anchor.get("target") or {})
    return source_point, target_point


def _mirror_point_x(point: tuple[float, float]) -> tuple[float, float]:
    return (-point[0], point[1])


def _axis_angle_degrees(
    primary: tuple[float, float],
    secondary: tuple[float, float],
) -> Optional[float]:
    dx = secondary[0] - primary[0]
    dy = secondary[1] - primary[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def _normalize_angle_delta(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def _pair_flip_vote(
    primary: str,
    secondary: str,
    anchor_features: list[dict[str, Any]],
) -> tuple[int, int]:
    votes_flip = 0
    votes_no_flip = 0
    for landmark_name in (primary, secondary):
        source_point, target_point = _find_landmark_positions(anchor_features, landmark_name)
        if source_point is None or target_point is None:
            continue
        if source_point[0] == 0.0 or target_point[0] == 0.0:
            continue
        if source_point[0] == target_point[0]:
            votes_no_flip += 1
        elif source_point[0] == -target_point[0]:
            votes_flip += 1
    return votes_flip, votes_no_flip


def _compute_rotation_from_landmark_pair(
    primary: str,
    secondary: str,
    anchor_features: list[dict[str, Any]],
    should_flip: bool,
    max_rotation: float,
) -> tuple[float, str]:
    src_primary, tgt_primary = _find_landmark_positions(anchor_features, primary)
    src_secondary, tgt_secondary = _find_landmark_positions(anchor_features, secondary)
    if None in (src_primary, tgt_primary, src_secondary, tgt_secondary):
        return 0.0, ""

    if should_flip:
        src_primary = _mirror_point_x(src_primary)
        src_secondary = _mirror_point_x(src_secondary)

    source_angle = _axis_angle_degrees(src_primary, src_secondary)
    target_angle = _axis_angle_degrees(tgt_primary, tgt_secondary)
    if source_angle is None or target_angle is None:
        return 0.0, ""

    delta = _normalize_angle_delta(target_angle - source_angle)
    if abs(delta) > max(max_rotation, 45.0):
        return 0.0, (
            f"{primary}->{secondary} axis gap too large for in-plane pre-align "
            f"({delta:.1f} deg)"
        )
    delta = _clamp(delta, -max_rotation, max_rotation)
    if abs(delta) < 3.0:
        return 0.0, f"{primary}->{secondary} axis already aligned ({delta:.1f} deg residual)"
    return (
        delta,
        f"{primary}->{secondary} axis: source {source_angle:.1f} deg vs "
        f"target {target_angle:.1f} deg -> rotate {delta:.1f} deg",
    )


def _anchor_horizontal_votes(anchor_features: list[dict[str, Any]]) -> tuple[int, int]:
    votes_flip = 0
    votes_no_flip = 0
    for anchor in anchor_features:
        if not anchor.get("comparable", True):
            continue
        source_h = _normalize_axis_label((anchor.get("source") or {}).get("horizontal"))
        target_h = _normalize_axis_label((anchor.get("target") or {}).get("horizontal"))
        if source_h not in _HORIZONTAL_POS or target_h not in _HORIZONTAL_POS:
            continue
        if source_h == "center" or target_h == "center":
            continue
        if _HORIZONTAL_POS[source_h] == _HORIZONTAL_POS[target_h]:
            votes_no_flip += 1
        elif _HORIZONTAL_POS[source_h] == -_HORIZONTAL_POS[target_h]:
            votes_flip += 1
    return votes_flip, votes_no_flip


def build_pre_align_decision(
    parsed: dict[str, Any],
    max_rotation: float = 30.0,
) -> PreAlignDecision:
    anchor_features = list(parsed.get("anchor_features") or [])
    primary, secondary = _resolve_landmark_pair(parsed)
    votes_flip, votes_no_flip = (
        _pair_flip_vote(primary, secondary, anchor_features)
        if primary and secondary
        else _anchor_horizontal_votes(anchor_features)
    )

    layout = str(parsed.get("horizontal_layout_match", "")).strip().lower()
    vlm_flip = bool(parsed.get("should_horizontal_flip", False))
    confidence = float(parsed.get("confidence", 0.0))

    has_landmark_pair = bool(primary and secondary)
    src_primary, tgt_primary = _find_landmark_positions(anchor_features, primary) if primary else (None, None)
    src_secondary, tgt_secondary = (
        _find_landmark_positions(anchor_features, secondary) if secondary else (None, None)
    )
    pair_positions_ok = None not in (src_primary, tgt_primary, src_secondary, tgt_secondary)

    should_flip = vlm_flip
    if pair_positions_ok and votes_flip >= 2 and votes_flip > votes_no_flip:
        should_flip = True
        confidence = max(confidence, 0.70)
    elif votes_no_flip >= 2 and votes_no_flip > votes_flip:
        should_flip = False
        if votes_flip > 0:
            confidence = min(confidence, 0.55)
    elif layout == "mirrored" and votes_flip >= 1 and votes_flip >= votes_no_flip:
        should_flip = True
    elif layout in {"same", "mixed", "unclear"} and votes_flip <= votes_no_flip:
        should_flip = False
        if layout in {"mixed", "unclear"}:
            confidence = min(confidence, 0.50)

    if not has_landmark_pair or not pair_positions_ok:
        should_flip = False
        confidence = min(confidence, 0.40)
        if not has_landmark_pair:
            reason_suffix = "No shared landmark pair found on both source and target."
        else:
            reason_suffix = (
                f"Landmarks {primary}/{secondary} are not both present on source and target."
            )
        reason = str(parsed.get("reason", "")).strip()
        if reason_suffix not in reason:
            reason = f"{reason}. {reason_suffix}".strip(". ").strip()
    else:
        reason = str(parsed.get("reason", ""))

    computed_rotation, computed_basis = (
        _compute_rotation_from_landmark_pair(
            primary,
            secondary,
            anchor_features,
            should_flip=should_flip,
            max_rotation=max_rotation,
        )
        if pair_positions_ok
        else (0.0, "")
    )

    rotation = float(parsed.get("rotation_degrees", 0.0))
    rotation_basis = str(parsed.get("rotation_basis", "")).strip()
    if computed_basis:
        rotation = computed_rotation
        rotation_basis = computed_basis
        if abs(computed_rotation) >= 3.0:
            confidence = max(confidence, 0.65)
    elif abs(rotation) >= 1.0 and not rotation_basis:
        confidence = min(confidence, 0.50)
        rotation = 0.0

    if not pair_positions_ok:
        rotation = 0.0
        rotation_basis = ""

    return PreAlignDecision(
        should_horizontal_flip=should_flip,
        rotation_degrees=rotation,
        confidence=confidence,
        reason=reason,
        object_category=str(parsed.get("object_category", "")),
        primary_landmark=primary,
        secondary_landmark=secondary,
        anchor_features=anchor_features,
        horizontal_layout_match=layout,
        rotation_basis=rotation_basis,
        flip_votes_flip=votes_flip,
        flip_votes_no_flip=votes_no_flip,
    )


def _format_anchor_summary(
    anchor_features: list[dict[str, Any]],
    primary_landmark: str = "",
    secondary_landmark: str = "",
) -> str:
    lines: list[str] = []
    if primary_landmark and secondary_landmark:
        lines.append(f"  axis: {primary_landmark} -> {secondary_landmark}")
    for anchor in anchor_features:
        name = anchor.get("name", "anchor")
        role = anchor.get("role", "")
        role_suffix = f" [{role}]" if role else ""
        source = anchor.get("source") or {}
        target = anchor.get("target") or {}
        on_both = _landmark_present_on_both(anchor)
        both_tag = " [shared]" if on_both else " [NOT on both]"
        lines.append(
            f"  - {name}{role_suffix}{both_tag}: "
            f"source=({source.get('horizontal', '?')}, {source.get('vertical', '?')}) "
            f"target=({target.get('horizontal', '?')}, {target.get('vertical', '?')})"
        )
    return "\n".join(lines)


def _background_fill_color(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        [
            arr[0, :, :],
            arr[-1, :, :],
            arr[:, 0, :],
            arr[:, -1, :],
        ],
        axis=0,
    )
    return tuple(int(channel) for channel in np.median(border, axis=0))


def apply_pre_alignment(
    source_img: Image.Image,
    decision: PreAlignDecision,
    min_confidence: float,
    max_rotation: float,
) -> Image.Image:
    aligned = source_img
    if decision.confidence < min_confidence:
        decision.applied_horizontal_flip = False
        decision.applied_vertical_flip = False
        decision.applied_rotation_degrees = 0.0
        return aligned

    if decision.should_horizontal_flip:
        aligned = aligned.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        decision.applied_horizontal_flip = True

    rotation = _clamp(decision.rotation_degrees, -max_rotation, max_rotation)
    if abs(rotation) >= 1.0:
        aligned = aligned.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=_background_fill_color(aligned),
        )
        decision.applied_rotation_degrees = rotation
    else:
        decision.applied_rotation_degrees = 0.0

    return aligned


COARSE_FLIP_MODES: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
COARSE_ROTATIONS_DEG: tuple[int, ...] = (0, 90, 180, 270)


@dataclass(frozen=True)
class CoarseOrientationTransform:
    candidate_id: int
    flip_horizontal: bool
    flip_vertical: bool
    rotation_degrees: int

    @property
    def label(self) -> str:
        flip_parts: list[str] = []
        if self.flip_horizontal:
            flip_parts.append("flip_h")
        if self.flip_vertical:
            flip_parts.append("flip_v")
        flip_text = "+".join(flip_parts) if flip_parts else "no_flip"
        return f"id={self.candidate_id}: {flip_text}, rotate={self.rotation_degrees}°"


def _coarse_transform_output_key(image: Image.Image) -> tuple[tuple[int, int], bytes]:
    return (image.size, image.tobytes())


def enumerate_coarse_orientation_transforms() -> list[CoarseOrientationTransform]:
    """Enumerate all 4×4 flip/rotation parameter combos (16 total, with duplicates)."""
    transforms: list[CoarseOrientationTransform] = []
    candidate_id = 0
    for rotation_degrees in COARSE_ROTATIONS_DEG:
        for flip_horizontal, flip_vertical in COARSE_FLIP_MODES:
            transforms.append(
                CoarseOrientationTransform(
                    candidate_id=candidate_id,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    rotation_degrees=rotation_degrees,
                )
            )
            candidate_id += 1
    return transforms


def apply_coarse_orientation_transform(
    source_img: Image.Image,
    transform: CoarseOrientationTransform,
) -> Image.Image:
    aligned = source_img
    if transform.flip_horizontal:
        aligned = aligned.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform.flip_vertical:
        aligned = aligned.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform.rotation_degrees % 360 != 0:
        aligned = aligned.rotate(
            transform.rotation_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=_background_fill_color(aligned),
        )
    return aligned


def enumerate_unique_coarse_orientation_transforms(
    source_img: Image.Image,
) -> tuple[list[CoarseOrientationTransform], int]:
    """Return deduplicated transforms for this source; typically 8 unique orientations."""
    unique: list[CoarseOrientationTransform] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    raw_count = 0
    next_id = 0
    for rotation_degrees in COARSE_ROTATIONS_DEG:
        for flip_horizontal, flip_vertical in COARSE_FLIP_MODES:
            raw_count += 1
            raw = CoarseOrientationTransform(
                candidate_id=-1,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                rotation_degrees=rotation_degrees,
            )
            output = apply_coarse_orientation_transform(source_img, raw)
            key = _coarse_transform_output_key(output)
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                CoarseOrientationTransform(
                    candidate_id=next_id,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    rotation_degrees=rotation_degrees,
                )
            )
            next_id += 1
    return unique, raw_count


def _thumbnail_on_canvas(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), _background_fill_color(image))
    offset = ((size - thumb.width) // 2, (size - thumb.height) // 2)
    canvas.paste(thumb, offset)
    return canvas


def build_coarse_candidate_grid(
    candidates: list[Image.Image],
    transforms: list[CoarseOrientationTransform],
    thumb_size: int = 256,
    columns: int = 4,
) -> Image.Image:
    if len(candidates) != len(transforms):
        raise ValueError("candidates and transforms must have the same length")
    if not candidates:
        raise ValueError("candidates must not be empty")

    rows = (len(candidates) + columns - 1) // columns
    label_height = 28
    cell_w = thumb_size
    cell_h = thumb_size + label_height
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for index, (candidate, transform) in enumerate(zip(candidates, transforms)):
        row = index // columns
        col = index % columns
        x0 = col * cell_w
        y0 = row * cell_h
        thumb = _thumbnail_on_canvas(candidate, thumb_size)
        grid.paste(thumb, (x0, y0))
        draw.rectangle(
            [x0, y0 + thumb_size, x0 + cell_w - 1, y0 + cell_h - 1],
            fill=(12, 12, 12),
        )
        draw.text((x0 + 6, y0 + thumb_size + 6), str(transform.candidate_id), fill=(255, 255, 255), font=font)
    return grid


def _format_coarse_candidate_table(transforms: list[CoarseOrientationTransform]) -> str:
    return "\n".join(f"- {transform.label}" for transform in transforms)


def bruteforce_pre_align_source(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    output_dir: Path,
) -> tuple[Image.Image, PreAlignDecision, dict[str, Any]]:
    transforms, raw_count = enumerate_unique_coarse_orientation_transforms(source_img)
    if not transforms:
        raise RuntimeError("No coarse orientation candidates generated for bruteforce fallback.")

    log(
        "\n========== Phase -1B: Bruteforce Orientation Fallback "
        f"({len(transforms)} unique / {raw_count} raw combos) =========="
    )

    bruteforce_dir = output_dir / "prealign_bruteforce"
    bruteforce_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Image.Image] = []
    candidate_records: list[dict[str, Any]] = []
    for transform in transforms:
        candidate = apply_coarse_orientation_transform(source_img, transform)
        candidates.append(candidate)
        candidate_path = bruteforce_dir / f"candidate_{transform.candidate_id:02d}.png"
        save_image(candidate, candidate_path)
        candidate_records.append(
            {
                "candidate_id": transform.candidate_id,
                "flip_horizontal": transform.flip_horizontal,
                "flip_vertical": transform.flip_vertical,
                "rotation_degrees": transform.rotation_degrees,
                "image_path": str(candidate_path),
            }
        )

    grid_columns = 4 if len(transforms) >= 4 else len(transforms)
    grid = build_coarse_candidate_grid(candidates, transforms, columns=grid_columns)
    grid_path = bruteforce_dir / "candidate_grid.png"
    save_image(grid, grid_path)

    max_candidate_id = len(transforms) - 1
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PREALIGN_BRUTEFORCE_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": grid},
                {
                    "type": "text",
                    "text": f"(IMAGE 1 = CANDIDATE GRID, IDs 0-{max_candidate_id})",
                },
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = TARGET)"},
                {
                    "type": "text",
                    "text": PREALIGN_BRUTEFORCE_USER.format(
                        num_candidates=len(transforms),
                        max_candidate_id=max_candidate_id,
                        candidate_table=_format_coarse_candidate_table(transforms),
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=768))
    raw_id = parsed.get("best_candidate_id", 0)
    try:
        best_id = int(raw_id)
    except (TypeError, ValueError):
        best_id = 0
    best_id = max(0, min(len(transforms) - 1, best_id))
    chosen_transform = transforms[best_id]
    aligned = candidates[best_id]

    decision = PreAlignDecision(
        should_horizontal_flip=chosen_transform.flip_horizontal,
        rotation_degrees=float(chosen_transform.rotation_degrees),
        confidence=1.0,
        reason=str(parsed.get("reason", "")).strip()
        or f"Bruteforce fallback selected candidate {best_id}.",
        applied_horizontal_flip=chosen_transform.flip_horizontal,
        applied_vertical_flip=chosen_transform.flip_vertical,
        applied_rotation_degrees=float(chosen_transform.rotation_degrees),
    )
    payload = {
        "mode": "bruteforce_fallback",
        "num_raw_candidates": raw_count,
        "num_candidates": len(transforms),
        "best_candidate_id": best_id,
        "chosen_transform": asdict(chosen_transform),
        "candidate_records": candidate_records,
        "grid_path": str(grid_path),
        "vlm_response": parsed,
        "decision": asdict(decision),
    }
    (bruteforce_dir / "bruteforce_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(
        "[prealign] Bruteforce selected "
        f"candidate {best_id}: flip_h={chosen_transform.flip_horizontal}, "
        f"flip_v={chosen_transform.flip_vertical}, rotate={chosen_transform.rotation_degrees}°"
    )
    return aligned, decision, payload


def pre_align_source(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    output_dir: Path,
    min_confidence: float,
    max_rotation: float,
    retry_feedback: str = "",
) -> tuple[Image.Image, PreAlignDecision, dict[str, Any]]:
    log("\n========== Phase -1: Coarse Orientation Pre-Alignment ==========")
    log("[prealign] Selecting a two-landmark axis and comparing source vs target...")
    t0 = time.time()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PREALIGN_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = TARGET)"},
                {
                    "type": "text",
                    "text": PREALIGN_USER.format(
                        min_confidence=min_confidence,
                        max_rotation=max_rotation,
                    )
                    + (
                        "\n\nPrevious pre-align verification failed. Revise the decision.\n"
                        f"Failure reason: {retry_feedback}"
                        if retry_feedback
                        else ""
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=1024))
    decision = build_pre_align_decision(parsed, max_rotation=max_rotation)
    aligned = apply_pre_alignment(
        source_img=source_img,
        decision=decision,
        min_confidence=min_confidence,
        max_rotation=max_rotation,
    )
    parsed.update(asdict(decision))
    prealign_path = output_dir / "pre_alignment.json"
    prealign_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    log(
        "[prealign] Done in "
        f"{time.time() - t0:.1f}s | category={decision.object_category or 'unknown'}, "
        f"axis={decision.primary_landmark or '?'}->{decision.secondary_landmark or '?'}, "
        f"layout={decision.horizontal_layout_match or 'unknown'}, "
        f"anchor_votes flip={decision.flip_votes_flip} no_flip={decision.flip_votes_no_flip}, "
        f"confidence={decision.confidence:.2f}, "
        f"flip={decision.applied_horizontal_flip}, "
        f"rotate={decision.applied_rotation_degrees:.1f}°"
    )
    if decision.anchor_features:
        log("[prealign] Landmark pair comparison:")
        log(
            _format_anchor_summary(
                decision.anchor_features,
                decision.primary_landmark,
                decision.secondary_landmark,
            )
        )
    if decision.rotation_basis:
        log(f"[prealign] Rotation basis: {decision.rotation_basis}")
    if decision.reason:
        log(f"[prealign] Reason: {decision.reason}")
    log(f"[prealign] Wrote {prealign_path}")
    return aligned, decision, parsed


def verify_pre_alignment(
    original_source: Image.Image,
    aligned_source: Image.Image,
    target_img: Image.Image,
    decision: PreAlignDecision,
    planner_vlm: QwenVLMClient,
) -> PreAlignVerifyResult:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PREALIGN_VERIFY_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": original_source},
                {"type": "text", "text": "(IMAGE 1 = ORIGINAL SOURCE)"},
                {"type": "image", "image": aligned_source},
                {"type": "text", "text": "(IMAGE 2 = ALIGNED SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 3 = TARGET)"},
                {
                    "type": "text",
                    "text": PREALIGN_VERIFY_USER.format(
                        decision_json=json.dumps(asdict(decision), indent=2, ensure_ascii=False),
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=512))
    return PreAlignVerifyResult(
        overall_ok=bool(parsed.get("overall_ok", False)),
        recommendation=str(parsed.get("recommendation", "retry")).lower(),
        failure_reason=parsed.get("failure_reason"),
    )


def pre_align_source_until_verified(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    output_dir: Path,
    min_confidence: float,
    max_rotation: float,
    max_attempts: int = 0,
    bruteforce_after_attempts: int = 5,
) -> tuple[Image.Image, PreAlignDecision, dict[str, Any]]:
    attempt = 0
    feedback = ""
    retry_cap = max_attempts if max_attempts > 0 else bruteforce_after_attempts
    while True:
        attempt += 1
        log(f"\n[prealign] Verified attempt {attempt}/{retry_cap}")
        aligned, decision, parsed = pre_align_source(
            source_img=source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            output_dir=output_dir,
            min_confidence=min_confidence,
            max_rotation=max_rotation,
            retry_feedback=feedback,
        )
        if feedback:
            parsed["retry_feedback"] = feedback

        verify = verify_pre_alignment(source_img, aligned, target_img, decision, planner_vlm)
        parsed["verify"] = asdict(verify)
        (output_dir / "pre_alignment.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if verify.overall_ok and verify.recommendation == "apply":
            log("[prealign] Verify PASSED; applying aligned source.")
            return aligned, decision, parsed
        if verify.overall_ok and verify.recommendation == "rollback":
            log("[prealign] Verify accepted rollback; using original source.")
            decision.applied_horizontal_flip = False
            decision.applied_vertical_flip = False
            decision.applied_rotation_degrees = 0.0
            parsed["rolled_back"] = True
            return source_img, decision, parsed

        feedback = verify.failure_reason or "Pre-align verification failed; choose safer landmarks or skip transform."
        log(f"[prealign] Verify FAILED: {feedback}")
        if attempt >= retry_cap:
            log(
                f"[prealign] Landmark-based pre-align failed after {attempt} attempt(s); "
                f"running bruteforce fallback with unique flip/rotate candidates."
            )
            aligned, decision, bruteforce_payload = bruteforce_pre_align_source(
                source_img=source_img,
                target_img=target_img,
                planner_vlm=planner_vlm,
                output_dir=output_dir,
            )
            parsed["bruteforce_fallback"] = bruteforce_payload
            (output_dir / "pre_alignment.json").write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return aligned, decision, parsed


def verify_s_goal_image(
    source_img: Image.Image,
    s_goal_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    identity_scorer: DINOv2IdentityScorer,
    identity_threshold: float = 0.72,
) -> SGoalVerifyResult:
    identity_score = identity_scorer.similarity(source_img, s_goal_img)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SGOAL_VERIFY_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = PRE-ALIGNED SOURCE)"},
                {"type": "image", "image": s_goal_img},
                {"type": "text", "text": "(IMAGE 2 = S_GOAL CANDIDATE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 3 = TARGET GEOMETRY REFERENCE)"},
                {
                    "type": "text",
                    "text": SGOAL_VERIFY_USER.format(
                        identity_score=identity_score,
                        identity_threshold=identity_threshold,
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=512))
    hard_checks = {
        "pose_match": bool(parsed.get("pose_match", False)),
        "camera_match": bool(parsed.get("camera_match", False)),
        "scale_match": bool(parsed.get("scale_match", False)),
        "identity_preserved": bool(parsed.get("identity_preserved", False)),
        "not_target_copy": bool(parsed.get("not_target_copy", False)),
        "target_visible_layout_match": bool(parsed.get("target_visible_layout_match", False)),
        "no_severe_artifacts": bool(parsed.get("no_severe_artifacts", False)),
    }
    overall_ok = (
        bool(parsed.get("overall_ok", False))
        and identity_score >= identity_threshold
        and all(hard_checks.values())
    )
    failure_reason = parsed.get("failure_reason")
    if not overall_ok and not failure_reason:
        failed = [name for name, ok in hard_checks.items() if not ok]
        if identity_score < identity_threshold:
            failed.append(f"identity_score_below_threshold:{identity_score:.3f}")
        failure_reason = ", ".join(failed) or "S_goal verification failed"
    recommendation = str(
        parsed.get("recommendation", "use_as_goal" if overall_ok else "retry")
    ).lower()
    return SGoalVerifyResult(
        overall_ok=overall_ok,
        pose_match=hard_checks["pose_match"],
        camera_match=hard_checks["camera_match"],
        scale_match=hard_checks["scale_match"],
        identity_preserved=hard_checks["identity_preserved"],
        not_target_copy=hard_checks["not_target_copy"],
        target_visible_layout_match=hard_checks["target_visible_layout_match"],
        no_severe_artifacts=hard_checks["no_severe_artifacts"],
        recommendation=recommendation,
        identity_score=float(identity_score),
        identity_threshold=identity_threshold,
        scores=parse_score_dict(parsed.get("scores")),
        failure_reason=failure_reason,
    )


def generate_s_goal_until_verified(
    source_img: Image.Image,
    target_img: Image.Image,
    editor: MotionNFTEditor,
    planner_vlm: QwenVLMClient,
    identity_scorer: DINOv2IdentityScorer,
    output_dir: Path,
    max_retries: int = 2,
    identity_threshold: float = 0.72,
) -> tuple[Optional[Image.Image], dict[str, Any]]:
    log("\n========== Phase -0.5: S_goal One-Shot Generation (Base Qwen Edit Plus) ==========")
    attempts: list[dict[str, Any]] = []
    last_img: Optional[Image.Image] = None
    last_verify: Optional[SGoalVerifyResult] = None
    max_attempts = max(1, max_retries + 1)

    for attempt_idx in range(1, max_attempts + 1):
        log(f"[s_goal] Attempt {attempt_idx}/{max_attempts} with base Qwen Edit Plus")
        t0 = time.time()
        prompt = SGOAL_EDIT_PROMPT
        if last_verify is not None and last_verify.failure_reason:
            prompt += (
                "\n\nPrevious S_goal verification failed. Correct this issue while preserving SOURCE identity:\n"
                f"{last_verify.failure_reason}"
            )
        s_goal_img = editor.edit_s_goal_base(
            source_img=source_img,
            target_img=target_img,
            instruction=prompt,
            step_seed=90000 + attempt_idx,
        )
        last_img = s_goal_img
        candidate_path = output_dir / f"s_goal_attempt_{attempt_idx:02d}.png"
        save_image(s_goal_img, candidate_path)

        verify = verify_s_goal_image(
            source_img=source_img,
            s_goal_img=s_goal_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            identity_scorer=identity_scorer,
            identity_threshold=identity_threshold,
        )
        last_verify = verify
        attempt_record = {
            "attempt": attempt_idx,
            "image_path": str(candidate_path),
            "elapsed_sec": round(time.time() - t0, 2),
            "verify": asdict(verify),
        }
        attempts.append(attempt_record)
        log(
            f"[s_goal] Verify overall_ok={verify.overall_ok}, "
            f"identity={verify.identity_score:.3f}, recommendation={verify.recommendation}"
        )
        if verify.overall_ok and verify.recommendation == "use_as_goal":
            final_path = output_dir / "s_goal.png"
            save_image(s_goal_img, final_path)
            payload = {
                "enabled": True,
                "used_as_planning_target": True,
                "fallback_to_target_path": False,
                "max_retries": max_retries,
                "identity_threshold": identity_threshold,
                "attempts": attempts,
                "selected_attempt": attempt_idx,
                "s_goal_path": str(final_path),
            }
            (output_dir / "s_goal_verify.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log("[s_goal] PASSED; using S_goal as progressive endpoint.")
            return s_goal_img, payload

    payload = {
        "enabled": True,
        "used_as_planning_target": False,
        "fallback_to_target_path": True,
        "max_retries": max_retries,
        "identity_threshold": identity_threshold,
        "attempts": attempts,
        "selected_attempt": None,
        "last_candidate_available": last_img is not None,
    }
    (output_dir / "s_goal_verify.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log("[s_goal] FAILED after retries; falling back to direct S_pre -> TARGET progressive path.")
    return None, payload


def plan(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    n_steps: int,
    output_dir: Optional[Path] = None,
    pre_alignment: Optional[PreAlignDecision] = None,
) -> tuple[Analysis, list[SubInstruction], dict[str, Any]]:
    log(f"[plan] Generating {n_steps}-step editing plan with VLM...")
    t0 = time.time()
    prealign_note = ""
    if pre_alignment is not None and (
        pre_alignment.applied_horizontal_flip
        or pre_alignment.applied_vertical_flip
        or abs(pre_alignment.applied_rotation_degrees) > 0
    ):
        prealign_note = (
            "\n\nPre-alignment already applied to SOURCE before this planning step:\n"
            f"- object_category: {pre_alignment.object_category}\n"
            f"- landmark_axis: {pre_alignment.primary_landmark} -> {pre_alignment.secondary_landmark}\n"
            f"- horizontal_flip: {pre_alignment.applied_horizontal_flip}\n"
            f"- vertical_flip: {pre_alignment.applied_vertical_flip}\n"
            f"- in_plane_rotation_degrees: {pre_alignment.applied_rotation_degrees:.1f}\n"
            f"- anchor_layout: {pre_alignment.horizontal_layout_match}\n"
            "Comparable landmark pairs were aligned (e.g. head->tail, front->rear). "
            "Do not repeat these coarse orientation transforms unless a small residual "
            "correction is still clearly needed. Focus planning on pose, scale, "
            "articulation, and viewpoint changes that flip/rotate cannot solve."
        )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PLANNING_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = ENDPOINT)"},
                {"type": "text", "text": PLANNING_USER.format(n_steps=n_steps) + prealign_note},
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=2048))
    analysis = build_analysis(parsed)
    steps = [build_step(step, idx + 1) for idx, step in enumerate(parsed["steps"])]
    if len(steps) != n_steps:
        raise ValueError(f"Planner returned {len(steps)} steps, expected {n_steps}.")

    elapsed = time.time() - t0
    src_desc = (analysis.source_description or "unknown")[:80]
    tgt_desc = (analysis.target_description or "unknown")[:80]
    log(f"[plan] Done in {elapsed:.1f}s — source: {src_desc}")
    log(f"[plan] Target geometry ref: {tgt_desc}")
    if analysis.shared_parts:
        log(f"[plan] Shared parts (track across steps): {format_shared_parts_list(analysis.shared_parts)}")
    else:
        log("[plan] Shared parts: none identified by planner")
    for step in steps:
        log(
            f"  step {step.step}/{n_steps} [{step.transform_type}] "
            f"progress={step.cumulative_progress:.2f} | {step.instruction}"
        )
    if output_dir is not None:
        plan_path = output_dir / "plan.json"
        plan_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[plan] Wrote {plan_path}")

    return analysis, steps, parsed


def diagnose_stage_plan(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    output_dir: Path,
    max_pose_steps: int,
    revision_hint: str = "",
) -> tuple[Analysis, StagePlan, dict[str, Any]]:
    log("\n========== Phase 0A: Pair Diagnosis ==========")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DIAGNOSIS_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = CURRENT SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = ENDPOINT)"},
                {
                    "type": "text",
                    "text": DIAGNOSIS_USER.format(max_pose_steps=max_pose_steps)
                    + (
                        "\n\nPrevious planning verification failed. Revise diagnosis/stage plan.\n"
                        f"Revision hint: {revision_hint}"
                        if revision_hint
                        else ""
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=2048))
    analysis = build_analysis(parsed)
    stage_plan = build_stage_plan(parsed, max_pose_steps=max_pose_steps)
    diagnosis_path = output_dir / "stage_diagnosis.json"
    diagnosis_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    log(
        "[diagnosis] "
        f"type={stage_plan.object_motion_type}, pose_gap={stage_plan.pose_gap}, "
        f"pose_steps={stage_plan.pose_steps}, angle_stage={stage_plan.run_angle_stage}"
    )
    log(
        "[diagnosis] camera "
        f"{stage_plan.source_camera.prompt} -> {stage_plan.target_camera.prompt}"
    )
    return analysis, stage_plan, parsed


def verify_planning(
    source_img: Image.Image,
    target_img: Image.Image,
    plan_payload: dict[str, Any],
    planner_vlm: QwenVLMClient,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": PLANNING_VERIFY_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = CURRENT SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = ENDPOINT)"},
                {
                    "type": "text",
                    "text": PLANNING_VERIFY_USER.format(
                        plan_json=json.dumps(plan_payload, indent=2, ensure_ascii=False),
                    ),
                },
            ],
        },
    ]
    return parse_json(planner_vlm.chat(messages, max_new_tokens=768))


def plan_pose_stage(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    diagnosis: dict[str, Any],
    pose_steps: int,
    output_dir: Path,
    revision_hint: str = "",
) -> tuple[list[SubInstruction], dict[str, Any]]:
    if pose_steps <= 0:
        return [], {"steps": []}

    prompt = POSE_PLANNING_USER.format(
        diagnosis_json=json.dumps(diagnosis, indent=2, ensure_ascii=False),
        pose_steps=pose_steps,
    )
    if revision_hint:
        prompt += (
            "\n\nPrevious planning verification failed. Revise the pose plan.\n"
            f"Revision hint: {revision_hint}"
        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": POSE_PLANNING_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_img},
                {"type": "text", "text": "(IMAGE 1 = CURRENT SOURCE)"},
                {"type": "image", "image": target_img},
                {"type": "text", "text": "(IMAGE 2 = ENDPOINT)"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=2048))
    raw_steps = parsed.get("steps", [])
    steps = [build_step(step, idx + 1) for idx, step in enumerate(raw_steps)]
    if len(steps) != pose_steps:
        raise ValueError(f"Pose planner returned {len(steps)} steps, expected {pose_steps}.")
    return steps, parsed


def build_stage_plan_payload(
    analysis: Analysis,
    stage_plan: StagePlan,
    pose_steps: list[SubInstruction],
    angle_steps: list[SubInstruction],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": asdict(analysis),
        "stage_plan": asdict(stage_plan),
        "diagnosis": diagnosis,
        "pose_steps": [asdict(step) for step in pose_steps],
        "angle_steps": [asdict(step) for step in angle_steps],
        "steps": [asdict(step) for step in pose_steps + angle_steps],
    }


def plan_staged_until_verified(
    source_img: Image.Image,
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    output_dir: Path,
    max_pose_steps: int,
    max_planning_attempts: int,
) -> tuple[Analysis, StagePlan, list[SubInstruction], dict[str, Any], dict[str, Any]]:
    attempt = 0
    revision_hint = ""
    last_verify: dict[str, Any] = {}
    while True:
        attempt += 1
        log(f"\n[plan] Verified staged planning attempt {attempt}")
        analysis, stage_plan, diagnosis = diagnose_stage_plan(
            source_img=source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            output_dir=output_dir,
            max_pose_steps=max_pose_steps,
            revision_hint=revision_hint,
        )
        pose_steps, pose_raw = plan_pose_stage(
            source_img=source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            diagnosis=diagnosis,
            pose_steps=stage_plan.pose_steps,
            output_dir=output_dir,
            revision_hint=revision_hint,
        )
        angle_steps = generate_angle_steps(stage_plan, start_step=len(pose_steps) + 1)
        all_steps = pose_steps + angle_steps
        for idx, step in enumerate(all_steps, start=1):
            step.step = idx
            if all_steps:
                step.cumulative_progress = max(step.cumulative_progress, idx / len(all_steps))

        plan_payload = build_stage_plan_payload(
            analysis=analysis,
            stage_plan=stage_plan,
            pose_steps=pose_steps,
            angle_steps=angle_steps,
            diagnosis=diagnosis,
        )
        plan_payload["pose_raw"] = pose_raw
        plan_payload["planning_attempt"] = attempt
        last_verify = verify_planning(source_img, target_img, plan_payload, planner_vlm)
        plan_payload["planning_verify"] = last_verify

        plan_path = output_dir / "plan.json"
        plan_path.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[plan] Wrote {plan_path}")

        if bool(last_verify.get("overall_ok", False)):
            log("[plan] Verify PASSED")
            return analysis, stage_plan, all_steps, plan_payload, last_verify

        revision_hint = str(
            last_verify.get("revision_hint")
            or last_verify.get("failure_reason")
            or "Planning verification failed; revise stage ordering and prompt formats."
        )
        log(f"[plan] Verify FAILED: {revision_hint}")
        if max_planning_attempts > 0 and attempt >= max_planning_attempts:
            raise RuntimeError(f"Planning verification failed after {attempt} attempts: {revision_hint}")


def vlm_verify(
    before: Image.Image,
    after: Image.Image,
    target_img: Image.Image,
    step: SubInstruction,
    planner_vlm: QwenVLMClient,
    shared_parts: Optional[list[str]] = None,
) -> VLMVerifyResult:
    shared_parts = shared_parts or []
    num_questions = 6 if shared_parts else 5
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
                {"type": "text", "text": "(IMAGE 3 = ENDPOINT)"},
                {
                    "type": "text",
                    "text": build_verify_prompt(
                        instruction=step.instruction,
                        transform_type=step.transform_type,
                        expected_change=step.expected_change,
                        shared_parts=shared_parts,
                    ),
                },
            ],
        },
    ]
    response = planner_vlm.chat(messages, max_new_tokens=256)
    answers = parse_yes_no(response, num_questions=num_questions)
    shared_parts_visible = answers.get(6, True) if shared_parts else True
    return VLMVerifyResult(
        geometric_change_applied=answers.get(1, False),
        identity_preserved=answers.get(2, False),
        physically_plausible=answers.get(3, False),
        no_artifacts=answers.get(4, False),
        closer_to_target=answers.get(5, False),
        shared_parts_visible=shared_parts_visible,
        scores=parse_scores(response),
    )


def replan(
    before: Image.Image,
    failed_after: Image.Image,
    target_img: Image.Image,
    failed_step: SubInstruction,
    failure_reason: str,
    planner_vlm: QwenVLMClient,
    shared_parts: Optional[list[str]] = None,
) -> SubInstruction:
    log(f"[replan] Asking VLM to revise step {failed_step.step} after failure: {failure_reason}")
    t0 = time.time()
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
                {"type": "text", "text": "(IMAGE 3 = ENDPOINT)"},
                {
                    "type": "text",
                    "text": REPLAN_USER.format(
                        failed_instruction=failed_step.instruction,
                        transform_type=failed_step.transform_type,
                        expected_change=failed_step.expected_change,
                        identity_warning=failed_step.identity_warning,
                        failure_reason=failure_reason,
                        shared_parts_list=format_shared_parts_list(shared_parts or []),
                    ),
                },
            ],
        },
    ]
    parsed = parse_json(planner_vlm.chat(messages, max_new_tokens=512))
    new_step = build_step(parsed, failed_step.step)
    log(f"[replan] Done in {time.time() - t0:.1f}s — new instruction: {new_step.instruction}")
    return new_step


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


def _flow_pair_stats(flow: np.ndarray, flow_threshold: float) -> tuple[float, tuple[float, float]]:
    mag = flow_magnitude(flow)
    motion_mask = mag > flow_threshold
    if not motion_mask.any():
        motion_mask = np.ones(mag.shape, dtype=bool)
    roi_flow = flow[motion_mask]
    mean_vec = roi_flow.mean(axis=0).astype(np.float32)
    mean_mag = float(np.linalg.norm(roi_flow, axis=-1).mean())
    norm = float(np.linalg.norm(mean_vec))
    if norm < 1e-6:
        direction = (0.0, 0.0)
    else:
        direction = (float(mean_vec[0] / norm), float(mean_vec[1] / norm))
    return mean_mag, direction


def compute_adjacent_flow_metrics(
    flow_estimator: UniMatchFlowEstimator,
    trajectory: list[Image.Image],
    flow_threshold: float,
) -> list[AdjacentFlowMetrics]:
    pairs: list[AdjacentFlowMetrics] = []
    for idx in range(len(trajectory) - 1):
        flow = flow_estimator.estimate(trajectory[idx], trajectory[idx + 1])
        mean_mag, direction = _flow_pair_stats(flow, flow_threshold)
        pairs.append(
            AdjacentFlowMetrics(
                from_step=idx,
                to_step=idx + 1,
                mean_magnitude=mean_mag,
                direction=direction,
            )
        )
    return pairs


def verify_trajectory_flow(
    pair_metrics: list[AdjacentFlowMetrics],
    max_magnitude_ratio: float = 4.0,
    direction_flip_cosine: float = -0.15,
) -> TrajectoryFlowVerifyResult:
    issues: list[str] = []
    if len(pair_metrics) < 2:
        return TrajectoryFlowVerifyResult(
            pair_metrics=pair_metrics,
            smooth_ok=True,
            issues=issues,
        )

    magnitudes = [pair.mean_magnitude for pair in pair_metrics]
    median_mag = float(np.median(magnitudes))

    for pair in pair_metrics:
        if median_mag > 0.5 and pair.mean_magnitude > max_magnitude_ratio * median_mag:
            issues.append(
                f"Flow outlier at step {pair.from_step:02d}->{pair.to_step:02d}: "
                f"magnitude={pair.mean_magnitude:.2f} vs median={median_mag:.2f}"
            )

    for idx in range(len(pair_metrics) - 1):
        left = pair_metrics[idx]
        right = pair_metrics[idx + 1]
        if left.mean_magnitude > 1e-3 and right.mean_magnitude > 1e-3:
            ratio = max(left.mean_magnitude, right.mean_magnitude) / min(
                left.mean_magnitude, right.mean_magnitude
            )
            if ratio > max_magnitude_ratio:
                issues.append(
                    f"Flow magnitude jump between step {left.from_step:02d}->{left.to_step:02d} "
                    f"and {right.from_step:02d}->{right.to_step:02d}: ratio={ratio:.2f}"
                )

        if left.direction != (0.0, 0.0) and right.direction != (0.0, 0.0):
            cos = cosine_similarity_np(
                np.asarray(left.direction, dtype=np.float32),
                np.asarray(right.direction, dtype=np.float32),
            )
            if cos < direction_flip_cosine:
                issues.append(
                    f"Flow direction flip between step {left.to_step:02d} and "
                    f"{right.to_step:02d}: cosine={cos:.2f}"
                )

    return TrajectoryFlowVerifyResult(
        pair_metrics=pair_metrics,
        smooth_ok=len(issues) == 0,
        issues=issues,
    )


def identify_suspect_steps(
    flow_result: TrajectoryFlowVerifyResult,
    verify_results: list[VerifyResult],
    vlm_result: Optional[TrajectoryVLMVerifyResult],
) -> list[int]:
    """Rank 1-based editing steps most likely responsible for trajectory issues."""
    scores: dict[int, float] = {}

    for issue in flow_result.issues:
        for match in re.finditer(r"step (\d+)->(\d+)", issue):
            to_step = int(match.group(2))
            from_step = int(match.group(1))
            if to_step > 0:
                scores[to_step] = scores.get(to_step, 0.0) + 2.0
            if from_step > 0:
                scores[from_step] = scores.get(from_step, 0.0) + 1.0
        for match in re.finditer(r"between step (\d+) and (\d+)", issue):
            left = int(match.group(1))
            right = int(match.group(2))
            scores[right] = scores.get(right, 0.0) + 1.5
            scores[left] = scores.get(left, 0.0) + 1.0

    for step_idx, verify in enumerate(verify_results, start=1):
        if not verify.overall_ok:
            scores[step_idx] = scores.get(step_idx, 0.0) + 1.0

    if vlm_result is not None and vlm_result.abrupt_jumps and flow_result.pair_metrics:
        worst = max(flow_result.pair_metrics, key=lambda pair: pair.mean_magnitude)
        if worst.to_step > 0:
            scores[worst.to_step] = scores.get(worst.to_step, 0.0) + 1.5

    if not scores and flow_result.pair_metrics:
        magnitudes = [pair.mean_magnitude for pair in flow_result.pair_metrics]
        median_mag = float(np.median(magnitudes))
        for pair in flow_result.pair_metrics:
            if pair.to_step <= 0:
                continue
            if median_mag > 0.5 and pair.mean_magnitude > 2.0 * median_mag:
                scores[pair.to_step] = scores.get(pair.to_step, 0.0) + 1.0

    return sorted(scores.keys(), key=lambda step: (-scores[step], step))


def vlm_verify_trajectory(
    trajectory: list[Image.Image],
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    shared_parts: list[str],
) -> TrajectoryVLMVerifyResult:
    content: list[dict[str, Any]] = []
    step_notes: list[str] = []
    for idx, image in enumerate(trajectory):
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f"(IMAGE {idx + 1} = step_{idx:02d})"})
        step_notes.append(f"  IMAGE {idx + 1} = step_{idx:02d}")

    content.append({"type": "image", "image": target_img})
    content.append(
        {
            "type": "text",
            "text": f"(IMAGE {len(trajectory) + 1} = ENDPOINT geometry reference)",
        }
    )
    content.append(
        {
            "type": "text",
            "text": TRAJECTORY_VERIFY_USER.format(
                step_image_notes="\n".join(step_notes),
                shared_parts_list=format_shared_parts_list(shared_parts),
            ),
        }
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": TRAJECTORY_VERIFY_SYSTEM}]},
        {"role": "user", "content": content},
    ]
    response = planner_vlm.chat(messages, max_new_tokens=256)
    answers = parse_yes_no(response, num_questions=3)
    shared_parts_visible = answers.get(3, True) if shared_parts else True
    return TrajectoryVLMVerifyResult(
        progressive_toward_target=answers.get(1, False),
        abrupt_jumps=answers.get(2, False),
        shared_parts_visible=shared_parts_visible,
        scores=parse_scores(response),
    )


def verify_trajectory(
    trajectory: list[Image.Image],
    target_img: Image.Image,
    flow_estimator: UniMatchFlowEstimator,
    planner_vlm: Optional[QwenVLMClient],
    shared_parts: list[str],
    flow_threshold: float,
    skip_trajectory_vlm: bool,
    max_flow_magnitude_ratio: float,
    verify_results: Optional[list[VerifyResult]] = None,
) -> TrajectoryVerifyResult:
    log("[trajectory] Computing adjacent-pair optical flow metrics...")
    pair_metrics = compute_adjacent_flow_metrics(flow_estimator, trajectory, flow_threshold)
    for pair in pair_metrics:
        log(
            f"  step {pair.from_step:02d}->{pair.to_step:02d}: "
            f"mag={pair.mean_magnitude:.2f}, "
            f"dir=({pair.direction[0]:+.2f}, {pair.direction[1]:+.2f})"
        )

    flow_result = verify_trajectory_flow(
        pair_metrics,
        max_magnitude_ratio=max_flow_magnitude_ratio,
    )
    if flow_result.smooth_ok:
        log("[trajectory] Flow continuity OK")
    else:
        log("[trajectory] Flow continuity issues:")
        for issue in flow_result.issues:
            log(f"  - {issue}")

    vlm_result: Optional[TrajectoryVLMVerifyResult] = None
    vlm_ok = True
    if skip_trajectory_vlm:
        log("[trajectory] VLM sequence review skipped (--skip_trajectory_vlm)")
    else:
        if planner_vlm is None:
            raise ValueError("planner_vlm is required for trajectory VLM verify.")
        log("[trajectory] Running VLM multi-image sequence review...")
        vlm_result = vlm_verify_trajectory(
            trajectory=trajectory,
            target_img=target_img,
            planner_vlm=planner_vlm,
            shared_parts=shared_parts,
        )
        log(
            "[trajectory] VLM: "
            f"geometry_progressive={vlm_result.progressive_toward_target}, "
            f"abrupt_geometry_jumps={vlm_result.abrupt_jumps}, "
            f"shared_parts={vlm_result.shared_parts_visible}, "
            f"scores={vlm_result.scores}"
        )
        vlm_ok = (
            vlm_result.progressive_toward_target
            and not vlm_result.abrupt_jumps
            and vlm_result.shared_parts_visible
        )

    overall_ok = flow_result.smooth_ok and vlm_ok
    failure_reasons: list[str] = []
    if not flow_result.smooth_ok:
        failure_reasons.extend(flow_result.issues)
    if vlm_result is not None:
        if not vlm_result.progressive_toward_target:
            failure_reasons.append(
                "VLM: SOURCE geometry (pose/size/rotation) not progressively toward ENDPOINT"
            )
        if vlm_result.abrupt_jumps:
            failure_reasons.append("VLM: abrupt geometry jumps, reversals, part swaps, or topology changes detected")
        if not vlm_result.shared_parts_visible:
            failure_reasons.append(
                f"VLM: shared parts not trackable across the trajectory: "
                f"{format_shared_parts_list(shared_parts)}"
            )

    if overall_ok:
        log("[trajectory] Trajectory verify PASSED")
    else:
        log(f"[trajectory] Trajectory verify FAILED: {'; '.join(failure_reasons)}")

    suspect_steps = identify_suspect_steps(
        flow_result,
        verify_results=verify_results or [],
        vlm_result=vlm_result,
    )
    if suspect_steps:
        log(f"[trajectory] Suspect steps (1-based): {suspect_steps}")

    return TrajectoryVerifyResult(
        flow=flow_result,
        vlm=vlm_result,
        overall_ok=overall_ok,
        failure_reason=("; ".join(failure_reasons) if failure_reasons else None),
        suspect_steps=suspect_steps,
    )


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
    shared_parts: Optional[list[str]] = None,
    verbose: bool = True,
) -> VerifyResult:
    shared_parts = shared_parts or []
    identity_threshold = 0.80
    skip_flow_direction = False
    check_flow_radial = False
    check_flow_roi = False
    check_flow_rotation = False

    if step.transform_type in {"viewpoint_azimuth", "viewpoint_elevation", "angle_size"}:
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
    if verbose:
        log(f"    [verify] optical flow computed ({flow.shape[1]}x{flow.shape[0]})")
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
    if verbose:
        log(
            f"    [verify] flow direction={flow_direction_ok}, magnitude={flow_magnitude_ok} | "
            f"DINO={identity_score:.3f} (threshold={identity_threshold:.2f})"
        )
    texture_ok = True
    silhouette_ok = True

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

    if objective_failures:
        if verbose:
            log(f"    [verify] objective check FAILED: {'; '.join(objective_failures)}")
        return VerifyResult(
            flow_direction_ok=flow_direction_ok,
            flow_magnitude_ok=flow_magnitude_ok,
            identity_ok=identity_ok,
            texture_ok=texture_ok,
            silhouette_ok=silhouette_ok,
            semantic_ok=False,
            shared_parts_ok=True,
            overall_ok=False,
            failure_reason="; ".join(objective_failures),
        )

    if skip_vlm_verify:
        if verbose:
            log("    [verify] VLM skipped (--skip_vlm_verify); objective checks passed")
        return VerifyResult(
            flow_direction_ok=flow_direction_ok,
            flow_magnitude_ok=flow_magnitude_ok,
            identity_ok=identity_ok,
            texture_ok=texture_ok,
            silhouette_ok=silhouette_ok,
            semantic_ok=True,
            shared_parts_ok=True,
            overall_ok=True,
            failure_reason=None,
        )

    if planner_vlm is None:
        raise ValueError("planner_vlm is required unless --skip_vlm_verify is set.")

    if verbose:
        log("    [verify] running VLM semantic checklist...")
    vlm_result = vlm_verify(
        before, after, target_img, step, planner_vlm, shared_parts=shared_parts
    )
    if verbose:
        log(
            "    [verify] VLM: "
            f"geometry={vlm_result.geometric_change_applied}, "
            f"identity={vlm_result.identity_preserved}, "
            f"shared_parts={vlm_result.shared_parts_visible}, "
            f"step_size_ok={vlm_result.physically_plausible}, "
            f"trackable_transition={vlm_result.no_artifacts}, "
            f"closer_to_endpoint={vlm_result.closer_to_target}, "
            f"scores={vlm_result.scores}"
        )
    shared_parts_ok = vlm_result.shared_parts_visible if shared_parts else True
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
        and shared_parts_ok
    )

    failure_reason = None
    if not overall:
        reasons = []
        if not vlm_result.geometric_change_applied:
            reasons.append("Geometric change not applied as instructed")
        if not vlm_result.identity_preserved:
            reasons.append("VLM: identity not preserved")
        if not shared_parts_ok:
            reasons.append(
                f"Shared parts are not trackable: {format_shared_parts_list(shared_parts)}"
            )
        if not vlm_result.physically_plausible:
            reasons.append("Step size is not comparable to a trackable short video gap")
        if not vlm_result.no_artifacts:
            reasons.append("Major parts are not trackable or changed identity abruptly")
        if not vlm_result.closer_to_target:
            reasons.append(
                "SOURCE geometry (pose/size/rotation) not closer to ENDPOINT than before"
            )
        failure_reason = "; ".join(reasons)

    return VerifyResult(
        flow_direction_ok=flow_direction_ok,
        flow_magnitude_ok=flow_magnitude_ok,
        identity_ok=identity_ok,
        texture_ok=texture_ok,
        silhouette_ok=silhouette_ok,
        semantic_ok=vlm_result.geometric_change_applied,
        shared_parts_ok=shared_parts_ok,
        overall_ok=overall,
        scores=vlm_result.scores,
        failure_reason=failure_reason,
    )


def execute_one_step(
    step_idx: int,
    current_step: SubInstruction,
    before: Image.Image,
    target_img: Image.Image,
    editor: MotionNFTEditor,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    planner_vlm: Optional[QwenVLMClient],
    max_retries: int,
    flow_threshold: float,
    skip_vlm_verify: bool,
    output_dir: Path,
    shared_parts: list[str],
    attempt_prefix: str = "",
) -> tuple[Image.Image, SubInstruction, VerifyResult]:
    retry_count = 0
    edited: Optional[Image.Image] = None
    verify: Optional[VerifyResult] = None

    while retry_count <= max_retries:
        log(
            f"\n[step {step_idx} | retry {retry_count}/{max_retries}] "
            f"[{current_step.transform_type}] {current_step.instruction}"
        )
        t_edit = time.time()
        log("  editing with MotionNFT...")
        adapter = "angle" if current_step.transform_type == "angle_size" else "motion"
        edited = editor.edit(
            source_img=before,
            instruction=current_step.instruction,
            step_seed=step_idx * 100 + retry_count,
            adapter=adapter,
        )
        log(f"  edit done in {time.time() - t_edit:.1f}s")
        retry_name = f"step_{step_idx:02d}{attempt_prefix}_retry_{retry_count}.png"
        save_image(edited, output_dir / retry_name)

        log("  verifying edit...")
        t_verify = time.time()
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
            shared_parts=shared_parts,
        )
        log(f"  verify done in {time.time() - t_verify:.1f}s")

        if verify.overall_ok:
            log(f"[step {step_idx}] PASSED")
            break

        log(f"[step {step_idx}] FAILED: {verify.failure_reason}")
        if retry_count >= max_retries:
            log(f"[step {step_idx}] max retries reached; keeping last attempt")
            break
        if current_step.transform_type == "angle_size":
            log("[replan] Angle-LoRA step keeps fixed <sks> prompt; retrying with next seed.")
            retry_count += 1
            continue
        if planner_vlm is None:
            raise ValueError("Replanning requires planner_vlm.")
        current_step = replan(
            before=before,
            failed_after=edited,
            target_img=target_img,
            failed_step=current_step,
            failure_reason=verify.failure_reason or "unknown failure",
            planner_vlm=planner_vlm,
            shared_parts=shared_parts,
        )
        retry_count += 1

    if edited is None or verify is None:
        raise RuntimeError(f"Step {step_idx} did not produce an edited image.")
    return edited, current_step, verify


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
    shared_parts: Optional[list[str]] = None,
) -> tuple[list[Image.Image], list[SubInstruction], list[VerifyResult]]:
    shared_parts = shared_parts or []
    trajectory = [source_img]
    verify_results: list[VerifyResult] = []
    final_steps: list[SubInstruction] = []

    log(f"[execute] Starting progressive editing ({len(steps)} steps, max_retries={max_retries})")
    if shared_parts:
        log(f"[execute] Tracking shared parts: {format_shared_parts_list(shared_parts)}")
    step_bar = tqdm(steps, desc="Editing steps", unit="step", disable=not VERBOSE)

    for step_idx, step in enumerate(step_bar, start=1):
        step_bar.set_postfix(step=f"{step_idx}/{len(steps)}", type=step.transform_type)
        before = trajectory[-1]
        edited, current_step, verify = execute_one_step(
            step_idx=step_idx,
            current_step=step,
            before=before,
            target_img=target_img,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            planner_vlm=planner_vlm,
            max_retries=max_retries,
            flow_threshold=flow_threshold,
            skip_vlm_verify=skip_vlm_verify,
            output_dir=output_dir,
            shared_parts=shared_parts,
        )
        trajectory.append(edited)
        final_steps.append(current_step)
        verify_results.append(verify)
        save_image(edited, output_dir / f"step_{step_idx:02d}.png")

    log(f"[execute] Finished all {len(steps)} steps")
    return trajectory, final_steps, verify_results


def re_execute_steps_from(
    start_step: int,
    trajectory: list[Image.Image],
    final_steps: list[SubInstruction],
    verify_results: list[VerifyResult],
    target_img: Image.Image,
    editor: MotionNFTEditor,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    planner_vlm: Optional[QwenVLMClient],
    max_retries: int,
    flow_threshold: float,
    skip_vlm_verify: bool,
    output_dir: Path,
    shared_parts: list[str],
    attempt_prefix: str,
) -> tuple[list[Image.Image], list[SubInstruction], list[VerifyResult]]:
    if start_step < 1 or start_step > len(final_steps):
        raise ValueError(f"start_step must be in [1, {len(final_steps)}], got {start_step}")

    updated_trajectory = trajectory[:start_step]
    updated_steps = list(final_steps)
    updated_verify = list(verify_results)

    for step_idx in range(start_step, len(final_steps) + 1):
        before = updated_trajectory[-1]
        edited, current_step, verify = execute_one_step(
            step_idx=step_idx,
            current_step=updated_steps[step_idx - 1],
            before=before,
            target_img=target_img,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            planner_vlm=planner_vlm,
            max_retries=max_retries,
            flow_threshold=flow_threshold,
            skip_vlm_verify=skip_vlm_verify,
            output_dir=output_dir,
            shared_parts=shared_parts,
            attempt_prefix=attempt_prefix,
        )
        updated_trajectory.append(edited)
        updated_steps[step_idx - 1] = current_step
        updated_verify[step_idx - 1] = verify
        save_image(edited, output_dir / f"step_{step_idx:02d}.png")

    return updated_trajectory, updated_steps, updated_verify


def repair_trajectory(
    trajectory: list[Image.Image],
    final_steps: list[SubInstruction],
    verify_results: list[VerifyResult],
    target_img: Image.Image,
    planner_vlm: QwenVLMClient,
    editor: MotionNFTEditor,
    flow_estimator: UniMatchFlowEstimator,
    identity_scorer: DINOv2IdentityScorer,
    output_dir: Path,
    shared_parts: list[str],
    max_retries: int,
    flow_threshold: float,
    skip_vlm_verify: bool,
    skip_trajectory_vlm: bool,
    trajectory_flow_ratio: float,
    max_trajectory_repairs: int,
) -> tuple[
    list[Image.Image],
    list[SubInstruction],
    list[VerifyResult],
    TrajectoryVerifyResult,
    list[dict[str, Any]],
]:
    log("\n========== Phase 1.6: Trajectory Repair ==========")
    repair_log: list[dict[str, Any]] = []
    current_trajectory = list(trajectory)
    current_steps = list(final_steps)
    current_verify = list(verify_results)

    trajectory_verify = verify_trajectory(
        trajectory=current_trajectory,
        target_img=target_img,
        flow_estimator=flow_estimator,
        planner_vlm=planner_vlm,
        shared_parts=shared_parts,
        flow_threshold=flow_threshold,
        skip_trajectory_vlm=skip_trajectory_vlm,
        max_flow_magnitude_ratio=trajectory_flow_ratio,
        verify_results=current_verify,
    )

    for round_idx in range(max_trajectory_repairs):
        if trajectory_verify.overall_ok:
            log("[trajectory-repair] Trajectory already OK; no repair needed.")
            break

        suspect_steps = trajectory_verify.suspect_steps
        if not suspect_steps:
            log("[trajectory-repair] No suspect steps identified; stopping repair.")
            break

        start_step = min(suspect_steps)
        attempt_prefix = f"_repair{round_idx + 1}"
        failure_reason = trajectory_verify.failure_reason or "Trajectory continuity failed"
        log(
            f"[trajectory-repair] Round {round_idx + 1}/{max_trajectory_repairs}: "
            f"re-executing from step {start_step} (suspects={suspect_steps})"
        )

        if planner_vlm is not None and start_step <= len(current_trajectory) - 1:
            repaired_step = replan(
                before=current_trajectory[start_step - 1],
                failed_after=current_trajectory[start_step],
                target_img=target_img,
                failed_step=current_steps[start_step - 1],
                failure_reason=(
                    f"Trajectory repair round {round_idx + 1}: {failure_reason}. "
                    f"Focus on smoothing the transition into step_{start_step:02d}."
                ),
                planner_vlm=planner_vlm,
                shared_parts=shared_parts,
            )
            current_steps[start_step - 1] = repaired_step

        current_trajectory, current_steps, current_verify = re_execute_steps_from(
            start_step=start_step,
            trajectory=current_trajectory,
            final_steps=current_steps,
            verify_results=current_verify,
            target_img=target_img,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            planner_vlm=planner_vlm,
            max_retries=max_retries,
            flow_threshold=flow_threshold,
            skip_vlm_verify=skip_vlm_verify,
            output_dir=output_dir,
            shared_parts=shared_parts,
            attempt_prefix=attempt_prefix,
        )

        trajectory_verify = verify_trajectory(
            trajectory=current_trajectory,
            target_img=target_img,
            flow_estimator=flow_estimator,
            planner_vlm=planner_vlm,
            shared_parts=shared_parts,
            flow_threshold=flow_threshold,
            skip_trajectory_vlm=skip_trajectory_vlm,
            max_flow_magnitude_ratio=trajectory_flow_ratio,
            verify_results=current_verify,
        )

        round_record = {
            "round": round_idx + 1,
            "start_step": start_step,
            "suspect_steps": suspect_steps,
            "overall_ok_after": trajectory_verify.overall_ok,
            "failure_reason_after": trajectory_verify.failure_reason,
        }
        repair_log.append(round_record)
        log(
            f"[trajectory-repair] Round {round_idx + 1} done | "
            f"overall_ok={trajectory_verify.overall_ok}"
        )
        if trajectory_verify.overall_ok:
            break

    repair_path = output_dir / "trajectory_repair.json"
    repair_path.write_text(json.dumps(repair_log, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[trajectory-repair] Wrote {repair_path}")

    return current_trajectory, current_steps, current_verify, trajectory_verify, repair_log


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
    skip_trajectory_vlm: bool = False,
    trajectory_flow_ratio: float = 4.0,
    pre_alignment: Optional[PreAlignDecision] = None,
    skip_trajectory_repair: bool = False,
    max_trajectory_repairs: int = 2,
    max_pose_steps: int = 6,
    max_planning_attempts: int = 0,
    skip_s_goal: bool = False,
    s_goal_max_retries: int = 2,
    s_goal_identity_threshold: float = 0.72,
) -> PipelineResult:
    s_goal_payload: Optional[dict[str, Any]] = None
    planning_target_img = target_img
    planning_target_name = "target"
    if skip_s_goal:
        log("\n========== Phase -0.5: S_goal One-Shot Generation Skipped ==========")
        s_goal_payload = {
            "enabled": False,
            "used_as_planning_target": False,
            "fallback_to_target_path": True,
            "reason": "skip_s_goal=True",
        }
    else:
        s_goal_img, s_goal_payload = generate_s_goal_until_verified(
            source_img=source_img,
            target_img=target_img,
            editor=editor,
            planner_vlm=planner_vlm,
            identity_scorer=identity_scorer,
            output_dir=output_dir,
            max_retries=s_goal_max_retries,
            identity_threshold=s_goal_identity_threshold,
        )
        if s_goal_img is not None:
            planning_target_img = s_goal_img
            planning_target_name = "s_goal"

    log(f"\n========== Phase 0: Staged Planning (target={planning_target_name}) ==========")
    analysis, stage_plan, steps, raw_plan, planning_verify = plan_staged_until_verified(
        source_img=source_img,
        target_img=planning_target_img,
        planner_vlm=planner_vlm,
        output_dir=output_dir,
        max_pose_steps=max_pose_steps,
        max_planning_attempts=max_planning_attempts,
    )
    raw_plan["planning_target"] = planning_target_name
    raw_plan["s_goal"] = s_goal_payload
    (output_dir / "plan.json").write_text(
        json.dumps(raw_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("\n========== Phase 1: Progressive Execution (Pose -> Angle/Size) ==========")
    if not steps:
        log("[execute] No pose or angle/size edits needed; keeping source as final.")
        trajectory = [source_img]
        final_steps = []
        verify_results = []
    else:
        for step in steps:
            log(f"  staged step {step.step}/{len(steps)} [{step.transform_type}] {step.instruction}")
        trajectory, final_steps, verify_results = execute_progressive(
            source_img=source_img,
            target_img=planning_target_img,
            steps=steps,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            planner_vlm=planner_vlm,
            max_retries=max_retries,
            flow_threshold=flow_threshold,
            skip_vlm_verify=skip_vlm_verify,
            output_dir=output_dir,
            shared_parts=analysis.shared_parts,
        )

    log("\n========== Phase 1.5: Trajectory Verify ==========")
    trajectory_verify = verify_trajectory(
        trajectory=trajectory,
        target_img=planning_target_img,
        flow_estimator=flow_estimator,
        planner_vlm=planner_vlm,
        shared_parts=analysis.shared_parts,
        flow_threshold=flow_threshold,
        skip_trajectory_vlm=skip_trajectory_vlm,
        max_flow_magnitude_ratio=trajectory_flow_ratio,
        verify_results=verify_results,
    )

    trajectory_repair: Optional[list[dict[str, Any]]] = None
    if not skip_trajectory_repair and max_trajectory_repairs > 0 and not trajectory_verify.overall_ok:
        (
            trajectory,
            final_steps,
            verify_results,
            trajectory_verify,
            trajectory_repair,
        ) = repair_trajectory(
            trajectory=trajectory,
            final_steps=final_steps,
            verify_results=verify_results,
            target_img=planning_target_img,
            planner_vlm=planner_vlm,
            editor=editor,
            flow_estimator=flow_estimator,
            identity_scorer=identity_scorer,
            output_dir=output_dir,
            shared_parts=analysis.shared_parts,
            max_retries=max_retries,
            flow_threshold=flow_threshold,
            skip_vlm_verify=skip_vlm_verify,
            skip_trajectory_vlm=skip_trajectory_vlm,
            trajectory_flow_ratio=trajectory_flow_ratio,
            max_trajectory_repairs=max_trajectory_repairs,
        )
    elif skip_trajectory_repair:
        log("[trajectory-repair] Skipped (--skip_trajectory_repair)")

    trajectory_verify_path = output_dir / "trajectory_verify.json"
    trajectory_verify_path.write_text(
        json.dumps(trajectory_verify_to_dict(trajectory_verify), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"[trajectory] Wrote {trajectory_verify_path}")
    return PipelineResult(
        source_img=source_img,
        target_img=target_img,
        analysis=analysis,
        trajectory=trajectory,
        instructions=final_steps,
        verify_results=verify_results,
        final_img=trajectory[-1],
        pre_alignment=pre_alignment,
        s_goal=s_goal_payload,
        stage_plan=stage_plan,
        planning_verify=planning_verify,
        trajectory_verify=trajectory_verify,
        trajectory_repair=trajectory_repair,
    )


# ============================================================
# CLI
# ============================================================


def validate_pretrained_paths(
    editor_base_model: str,
    motionedit_lora_path: str,
    qwen_angles_lora_path: str,
    planner_vlm: str,
    dinov2_model: str,
    unimatch_ckpt: str,
) -> None:
    """Fail fast with clear errors if shared model paths are missing."""
    checks: list[tuple[str, Path, bool]] = [
        ("editor_base_model", Path(editor_base_model), True),
        ("planner_vlm", Path(planner_vlm), True),
        ("dinov2_model", Path(dinov2_model), True),
        ("unimatch_ckpt", Path(unimatch_ckpt), False),
        (
            "motionedit_lora",
            Path(motionedit_lora_path) / MOTIONEDIT_LORA_WEIGHT,
            False,
        ),
        (
            "qwen_angles_lora",
            Path(qwen_angles_lora_path) / ANGLE_LORA_WEIGHT,
            False,
        ),
    ]
    missing: list[str] = []
    for name, path, is_dir in checks:
        if is_dir and not path.is_dir():
            missing.append(f"  - {name}: {path} (directory not found)")
        elif not is_dir and not path.is_file():
            missing.append(f"  - {name}: {path} (file not found)")
    if missing:
        raise FileNotFoundError(
            "Missing pretrained model paths:\n"
            + "\n".join(missing)
            + "\n\nRun: python tools/download_progressive_pose_models.py"
        )


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

    parser.add_argument(
        "--editor_base_model",
        default=str(DEFAULT_EDITOR_BASE_MODEL),
        help="Editor base model path (default: .../Qwen-Image-Edit-2511).",
    )
    parser.add_argument(
        "--motionedit_lora_path",
        default=str(DEFAULT_MOTIONEDIT_LORA_PATH),
        help="MotionEdit LoRA directory (default: .../motionedit_vlm/motionedit-lora).",
    )
    parser.add_argument(
        "--qwen_angles_lora_path",
        default=str(DEFAULT_QWEN_ANGLES_LORA_PATH),
        help="Qwen Image Edit multi-angle LoRA directory.",
    )
    parser.add_argument(
        "--planner_vlm",
        default=str(DEFAULT_PLANNER_VLM),
        help="Planner/verifier VLM path (default: .../Qwen3-VL-8B-Instruct).",
    )
    parser.add_argument(
        "--dinov2_model",
        default=str(DEFAULT_DINOV2_MODEL),
        help="DINOv2 path (default: .../motionedit_vlm/dinov2-base).",
    )
    parser.add_argument(
        "--unimatch_ckpt",
        default=str(DEFAULT_UNIMATCH_CKPT),
        help="UniMatch checkpoint (default: .../motionedit_vlm/unimatch/pretrained/...).",
    )
    parser.add_argument(
        "--skip_path_check",
        action="store_true",
        help="Skip local pretrained path existence checks.",
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
    parser.add_argument(
        "--skip_trajectory_vlm",
        action="store_true",
        help="Skip VLM multi-image trajectory review (flow continuity still runs).",
    )
    parser.add_argument(
        "--trajectory_flow_ratio",
        type=float,
        default=4.0,
        help="Max allowed adjacent-pair flow magnitude ratio for trajectory continuity.",
    )
    parser.add_argument(
        "--max_trajectory_repairs",
        type=int,
        default=2,
        help="After trajectory verify fails, re-execute suspect steps up to this many rounds.",
    )
    parser.add_argument(
        "--skip_trajectory_repair",
        action="store_true",
        help="Disable Phase 1.6 cascade re-edit of suspect trajectory steps.",
    )
    parser.add_argument(
        "--skip_pre_align",
        action="store_true",
        help="Disable VLM-guided deterministic flip/rotate before planning.",
    )
    parser.add_argument(
        "--pre_align_min_confidence",
        type=float,
        default=0.60,
        help="Minimum VLM confidence required to apply pre-alignment.",
    )
    parser.add_argument(
        "--max_pre_align_rotation",
        type=float,
        default=30.0,
        help="Maximum absolute in-plane pre-alignment rotation in degrees.",
    )
    parser.add_argument(
        "--max_prealign_verify_attempts",
        type=int,
        default=0,
        help=(
            "Max landmark-based pre-align verify retries before bruteforce fallback; "
            "0 uses --prealign_bruteforce_after_attempts (default 5)."
        ),
    )
    parser.add_argument(
        "--prealign_bruteforce_after_attempts",
        type=int,
        default=5,
        help=(
            "After this many failed pre-align verify attempts, enumerate unique flip/rotate "
            "candidates (typically 8) and let the VLM pick the closest to TARGET."
        ),
    )
    parser.add_argument(
        "--max_pose_steps",
        type=int,
        default=6,
        help="Maximum MotionEdit pose/deformation steps from staged diagnosis.",
    )
    parser.add_argument(
        "--max_planning_attempts",
        type=int,
        default=0,
        help="Retry planning until verified; 0 means unlimited.",
    )
    parser.add_argument(
        "--skip_s_goal",
        action="store_true",
        help="Disable S_goal one-shot generation and use direct S_pre -> TARGET progressive editing.",
    )
    parser.add_argument(
        "--s_goal_max_retries",
        type=int,
        default=2,
        help="Retry S_goal one-shot generation this many times after the first failed attempt.",
    )
    parser.add_argument(
        "--s_goal_identity_threshold",
        type=float,
        default=0.72,
        help="Minimum DINO similarity between S_pre and S_goal for accepting S_goal.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose progress logs (tqdm bar still shown).",
    )
    return parser.parse_args()


def trajectory_verify_to_dict(result: Optional[TrajectoryVerifyResult]) -> Optional[dict[str, Any]]:
    if result is None:
        return None
    return {
        "overall_ok": result.overall_ok,
        "failure_reason": result.failure_reason,
        "flow": {
            "smooth_ok": result.flow.smooth_ok,
            "issues": result.flow.issues,
            "pair_metrics": [asdict(pair) for pair in result.flow.pair_metrics],
        },
        "suspect_steps": result.suspect_steps,
        "vlm": asdict(result.vlm) if result.vlm is not None else None,
    }


def average_score_dicts(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    valid = [scores for scores in score_dicts if scores]
    if not valid:
        return {}
    averaged: dict[str, float] = {}
    for key in SCORE_KEYS:
        values = [float(scores[key]) for scores in valid if key in scores]
        if values:
            averaged[key] = round(sum(values) / len(values), 4)
    return averaged


def summarize_scores(result: PipelineResult) -> dict[str, Any]:
    step_score_dicts = [verify.scores for verify in result.verify_results if verify.scores]
    trajectory_scores = (
        result.trajectory_verify.vlm.scores
        if result.trajectory_verify is not None and result.trajectory_verify.vlm is not None
        else {}
    )

    s_goal_scores: dict[str, float] = {}
    if result.s_goal:
        attempts = result.s_goal.get("attempts", [])
        selected_attempt = result.s_goal.get("selected_attempt")
        for attempt in attempts:
            if attempt.get("attempt") == selected_attempt:
                verify = attempt.get("verify", {})
                s_goal_scores = parse_score_dict(verify.get("scores"))
                break

    combined_score_dicts = list(step_score_dicts)
    if trajectory_scores:
        combined_score_dicts.append(trajectory_scores)
    if s_goal_scores:
        combined_score_dicts.append(s_goal_scores)

    return {
        "scale": "0.0-5.0",
        "per_step_mean": average_score_dicts(step_score_dicts),
        "trajectory": trajectory_scores,
        "s_goal": s_goal_scores,
        "overall_mean": average_score_dicts(combined_score_dicts),
        "num_scored_steps": len(step_score_dicts),
    }


def export_final_folder(output_dir: Path, result: PipelineResult) -> dict[str, Any]:
    """Save source, interpolation frames, and final image under output_dir/final/."""
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trajectory = result.trajectory
    if not trajectory:
        return {
            "source": None,
            "interpolation": [],
            "final": None,
            "num_interpolation_frames": 0,
        }

    save_image(trajectory[0], final_dir / "source.png")

    interpolation_paths: list[str] = []
    for idx, image in enumerate(trajectory[1:-1], start=1):
        rel_name = f"step_{idx:02d}.png"
        save_image(image, final_dir / rel_name)
        interpolation_paths.append(f"final/{rel_name}")

    save_image(trajectory[-1], final_dir / "final.png")

    return {
        "source": "final/source.png",
        "interpolation": interpolation_paths,
        "final": "final/final.png",
        "num_interpolation_frames": len(interpolation_paths),
    }


def serializable_result(
    result: PipelineResult,
    final_folder: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "pre_alignment": asdict(result.pre_alignment) if result.pre_alignment is not None else None,
        "s_goal": result.s_goal,
        "stage_plan": asdict(result.stage_plan) if result.stage_plan is not None else None,
        "planning_verify": result.planning_verify,
        "analysis": asdict(result.analysis),
        "instructions": [asdict(step) for step in result.instructions],
        "verify_results": [asdict(verify) for verify in result.verify_results],
        "trajectory_verify": trajectory_verify_to_dict(result.trajectory_verify),
        "trajectory_repair": result.trajectory_repair,
        "score_summary": summarize_scores(result),
        "final_image": "final.png",
        "trajectory": [f"step_{idx:02d}.png" for idx in range(len(result.trajectory))],
    }
    if final_folder is not None:
        payload["final_folder"] = final_folder
    return payload


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = not args.quiet
    torch.manual_seed(args.seed)

    log("========== Progressive Pose Edit ==========")
    log(f"source: {args.source_image}")
    log(f"target: {args.target_image}")
    log(f"output: {args.output_dir}")
    log(f"n_steps={args.n_steps}, max_retries={args.max_retries}, seed={args.seed}")

    log("[deps] Checking runtime dependency versions...")
    _check_runtime_dependencies()
    log("[deps] OK")

    if not args.skip_path_check:
        validate_pretrained_paths(
            editor_base_model=args.editor_base_model,
            motionedit_lora_path=args.motionedit_lora_path,
            qwen_angles_lora_path=args.qwen_angles_lora_path,
            planner_vlm=args.planner_vlm,
            dinov2_model=args.dinov2_model,
            unimatch_ckpt=args.unimatch_ckpt,
        )

    log("\n[models] Using pretrained paths:")
    log(f"  editor_base_model     = {args.editor_base_model}")
    log(f"  motionedit_lora_path  = {args.motionedit_lora_path}")
    log(f"  qwen_angles_lora_path = {args.qwen_angles_lora_path}")
    log(f"  planner_vlm           = {args.planner_vlm}")
    log(f"  dinov2_model          = {args.dinov2_model}")
    log(f"  unimatch_ckpt         = {args.unimatch_ckpt}")

    source_img = Image.open(args.source_image).convert("RGB")
    target_img = Image.open(args.target_image).convert("RGB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log("\n[init] Saving input images")
    save_image(source_img, output_dir / "source.png")
    save_image(target_img, output_dir / "target.png")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    log("\n[load] Planner VLM (Qwen3-VL)...")
    t0 = time.time()
    planner_vlm = QwenVLMClient(
        model_id=args.planner_vlm,
        device_map=args.vlm_device_map,
        torch_dtype="auto",
    )
    log(f"[load] Planner VLM ready ({time.time() - t0:.1f}s)")

    aligned_source_img = source_img
    pre_alignment: Optional[PreAlignDecision] = None
    if args.skip_pre_align:
        log("\n[prealign] Skipped (--skip_pre_align)")
        save_image(aligned_source_img, output_dir / "step_00.png")
    else:
        aligned_source_img, pre_alignment, _ = pre_align_source_until_verified(
            source_img=source_img,
            target_img=target_img,
            planner_vlm=planner_vlm,
            output_dir=output_dir,
            min_confidence=args.pre_align_min_confidence,
            max_rotation=args.max_pre_align_rotation,
            max_attempts=args.max_prealign_verify_attempts,
            bruteforce_after_attempts=args.prealign_bruteforce_after_attempts,
        )
        save_image(aligned_source_img, output_dir / "aligned_source.png")
        save_image(aligned_source_img, output_dir / "step_00.png")

    log("[load] MotionNFT editor (Qwen Image Edit + LoRA)...")
    t0 = time.time()
    editor = MotionNFTEditor(
        base_model=args.editor_base_model,
        lora_path=args.motionedit_lora_path,
        angle_lora_path=args.qwen_angles_lora_path,
        device=args.device,
        device_map=args.editor_device_map,
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    log(f"[load] MotionNFT editor ready ({time.time() - t0:.1f}s)")

    log("[load] UniMatch optical flow...")
    t0 = time.time()
    flow_estimator = UniMatchFlowEstimator(
        ckpt_path=Path(args.unimatch_ckpt),
        device=args.device,
        resize_to=args.flow_resize_to,
    )
    log(f"[load] UniMatch ready ({time.time() - t0:.1f}s)")

    log("[load] DINOv2 identity scorer...")
    t0 = time.time()
    identity_scorer = DINOv2IdentityScorer(args.dinov2_model, args.device)
    log(f"[load] DINOv2 ready ({time.time() - t0:.1f}s)")

    pipeline_t0 = time.time()
    result = progressive_pose_edit(
        source_img=aligned_source_img,
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
        skip_trajectory_vlm=args.skip_trajectory_vlm,
        trajectory_flow_ratio=args.trajectory_flow_ratio,
        pre_alignment=pre_alignment,
        skip_trajectory_repair=args.skip_trajectory_repair,
        max_trajectory_repairs=args.max_trajectory_repairs,
        max_pose_steps=args.max_pose_steps,
        max_planning_attempts=args.max_planning_attempts,
        skip_s_goal=args.skip_s_goal,
        s_goal_max_retries=args.s_goal_max_retries,
        s_goal_identity_threshold=args.s_goal_identity_threshold,
    )
    log(f"\n[pipeline] Total runtime: {time.time() - pipeline_t0:.1f}s")

    log("\n[output] Writing final artifacts")
    save_image(result.final_img, output_dir / "final.png")
    final_folder = export_final_folder(output_dir, result)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(serializable_result(result, final_folder=final_folder), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"  saved {result_path}")
    log(f"\nDone. Outputs in {output_dir}")
    log("  step_00.png .. step_NN.png  — full editing trajectory (debug/log)")
    log("  final/                      — source + interpolation + final only")
    log("  plan.json                   — VLM editing plan")
    log("  trajectory_verify.json      — flow + VLM trajectory continuity")
    log("  trajectory_repair.json      — suspect-step cascade repair log (if run)")
    log("  final.png                   — final result (same as final/final.png)")
    log("  result.json                 — analysis + verify summary + scores")


if __name__ == "__main__":
    main()
