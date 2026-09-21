import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    video_sample_fps: float = 10.0
    max_video_pixels: int = 40_000_000
    max_video_dimension: int = 8192
    max_video_duration_seconds: float = 14_400.0
    analysis_width: int = 768
    stable_frames: int = 5
    motion_threshold: float = 0.012
    turn_threshold: float = 0.025
    candidates_per_spread: int = 7
    candidate_selection_mode: str = "spread"
    output_layout: str = "spread"
    sharpness_weight: float = 1.0
    motion_weight: float = 2.0
    hand_overlap_weight: float = 8.0
    glare_overlap_weight: float = 6.0
    distortion_weight: float = 0.3
    flatness_weight: float = 0.2
    clipping_weight: float = 0.2
    exposure_weight: float = 0.3
    hand_backend: str = "mediapipe"
    hand_model: str = "models/hand_landmarker.task"
    hand_padding: float = 0.015
    finger_repair: bool = True
    glare_repair: bool = True
    finger_repair_min_coverage: float = 0.9
    finger_repair_fallback: str = "paper"
    page_background_fill: str = "paper"
    duplicate_hash_distance: int = 4
    duplicate_ssim: float = 0.985
    duplicate_suspect_ssim: float = 0.94
    dedupe_window: int = 3
    refine_quad: bool = True
    quad_max_shift: float = 0.025
    perspective_mode: str = "spread"
    page_contour_min_confidence: float = 0.55
    split_mode: str = "auto"
    spine_ratio: float = 0.5
    gutter_fraction: float = 0.0
    reading_order: str = "rtl"
    image_format: str = "png"
    max_image_pixels: int = 60_000_000
    max_image_dimension: int = 12_000
    max_image_files: int = 5_000
    jpeg_quality: int = 92
    grayscale: bool = False
    contrast: float = 1.0
    illumination_correction: bool = True
    illumination_strength: float = 0.7
    white_normalization: bool = True
    white_target: int = 245
    white_strength: float = 0.6
    rotation: int = 0
    auto_rotation: bool = True
    dewarp_mode: str = "auto"
    dewarp_strength: float = 0.0
    dewarp_max_strength: float = 0.25
    dewarp_min_confidence: float = 0.6
    pdf_dpi: int = 300
    hwaccel: str = "none"
    save_lowres: bool = False
    suspect_sharpness: float = 60.0
    suspect_hand_overlap: float = 0.015
    suspect_glare_overlap: float = 0.01
    interval_gap_factor: float = 3.0

    def validate(self):
        integer_fields = {
            "analysis_width",
            "max_video_pixels",
            "max_video_dimension",
            "max_image_pixels",
            "max_image_dimension",
            "max_image_files",
            "stable_frames",
            "candidates_per_spread",
            "dedupe_window",
            "jpeg_quality",
            "pdf_dpi",
            "rotation",
            "white_target",
            "duplicate_hash_distance",
        }
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in integer_fields and (type(v) is not int):
                raise ValueError(f"{f.name} must be an integer")
            if f.type is bool and type(v) is not bool:
                raise ValueError(f"{f.name} must be a boolean")
            if f.type is str and not isinstance(v, str):
                raise ValueError(f"{f.name} must be a string")
            if f.type is float and (type(v) not in (int, float) or not math.isfinite(v)):
                raise ValueError(f"{f.name} must be a finite number")
        limits = {
            "video_sample_fps": (1, 30),
            "max_video_pixels": (1_000_000, 40_000_000),
            "max_video_dimension": (1024, 8192),
            "max_video_duration_seconds": (60, 14_400),
            "max_image_pixels": (1_000_000, 60_000_000),
            "max_image_dimension": (1024, 12_000),
            "max_image_files": (1, 5_000),
            "analysis_width": (128, 1920),
            "stable_frames": (2, 100),
            "candidates_per_spread": (2, 30),
            "motion_threshold": (0.00001, 1),
            "turn_threshold": (0.00001, 1),
            "jpeg_quality": (1, 100),
            "pdf_dpi": (36, 1200),
            "spine_ratio": (0.25, 0.75),
            "gutter_fraction": (0, 0.05),
            "duplicate_hash_distance": (0, 64),
            "duplicate_ssim": (0, 1),
            "duplicate_suspect_ssim": (0, 1),
            "dedupe_window": (1, 50),
            "quad_max_shift": (0, 0.1),
            "page_contour_min_confidence": (0, 1),
            "hand_padding": (0, 0.1),
            "finger_repair_min_coverage": (0, 1),
            "suspect_glare_overlap": (0, 1),
            "dewarp_strength": (0, 0.6),
            "dewarp_max_strength": (0, 0.35),
            "dewarp_min_confidence": (0, 1),
            "illumination_strength": (0, 1),
            "contrast": (0.5, 2),
            "white_target": (200, 255),
            "white_strength": (0, 1),
            "suspect_sharpness": (0, 100000),
            "suspect_hand_overlap": (0, 1),
            "interval_gap_factor": (1, 20),
        }
        for name, (lo, hi) in limits.items():
            if not lo <= getattr(self, name) <= hi:
                raise ValueError(f"{name}: expected {lo}..{hi}")
        for f in fields(self):
            if f.name.endswith("_weight") and getattr(self, f.name) < 0:
                raise ValueError(f"{f.name} must be nonnegative")
        for name, choices in {
            "hand_backend": ("mediapipe", "none"),
            "image_format": ("png", "jpeg"),
            "split_mode": ("center", "auto"),
            "candidate_selection_mode": ("spread", "per_page"),
            "output_layout": ("spread", "split"),
            "perspective_mode": ("spread", "per_page"),
            "dewarp_mode": ("off", "manual", "auto"),
            "reading_order": ("rtl", "ltr"),
            "finger_repair_fallback": ("preserve", "paper", "white"),
            "page_background_fill": ("preserve", "paper", "white"),
            "hwaccel": ("none", "videotoolbox"),
            "rotation": (0, 90, 180, 270),
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name}: expected one of {choices}")
        if self.finger_repair and self.hand_backend != "mediapipe":
            raise ValueError("finger_repair requires hand_backend='mediapipe'")
        if self.turn_threshold < self.motion_threshold:
            raise ValueError("turn_threshold must be >= motion_threshold")
        if self.duplicate_suspect_ssim > self.duplicate_ssim:
            raise ValueError("duplicate_suspect_ssim must be <= duplicate_ssim")
        return self

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.get("hand_backend") == "none" and "finger_repair" not in data:
            data["finger_repair"] = False
        if "page_background_fill" not in data:
            data["page_background_fill"] = "preserve"
        # Numeric rotation existed before auto detection. Keep old configs and
        # manifests manual unless they explicitly opt into the new behavior.
        if data.get("rotation") in (90, 180, 270) and "auto_rotation" not in data:
            data["auto_rotation"] = False
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls().validate()
        path = Path(path).resolve()
        with path.open("rb") as f:
            cfg = cls.from_dict(tomllib.load(f))
        model = Path(cfg.hand_model).expanduser()
        cfg.hand_model = str((path.parent / model).resolve())
        return cfg

    def to_dict(self):
        return asdict(self)
