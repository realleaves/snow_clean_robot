"""Deterministic synthetic world used by the virtual acceptance suite.

The mock camera is a *fixed-viewpoint* synthetic scene. For V1.0 we model the
scene as a small set of ground targets whose camera-relative position is
``(forward_m, left_m)`` at a constant ground distance ``depth_m``. That is the
exact geometry the pipeline inverts:

    pixel  --(pixel_to_robot_ground)-->  (left, depth)  --(robot_to_world)-->  world

Targets are therefore declared from the calibration viewpoint (robot at
``(0, 0)`` with ``yaw = 0``) and reach the world frame through the very same
``camera_to_robot`` extrinsics that perception uses. This keeps the mock scene,
the coordinate transform and the resulting task coordinates mutually
consistent, so a full mock closed loop is self-verifying.
"""
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

# Optical-to-robot transform, identical to the one used by perception.
CAMERA_TO_ROBOT = np.array([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]], dtype=float)

DEFAULT_INTRINSICS = (500.0, 500.0, 320.0, 240.0)

#: Supported synthetic appearances. ``blob`` is a dark wet patch, ``bright_spot``
#: models a specular sheen, ``dark_patch`` models a soaked matte patch.
DECALS = ('blob', 'bright_spot', 'dark_patch')


@dataclass
class VirtualTarget:
    """One ground pollution patch placed in front of the calibration viewpoint."""

    target_id: int
    forward_m: float
    left_m: float
    radius_m: float
    pollution_fraction: float = 1.0
    base_color: tuple[int, int, int] = (45, 65, 75)
    depth_m: float = 0.8
    decal: str = 'blob'  # blob | bright_spot | dark_patch
    blend: float = 1.0
    brightness_override: float | None = None
    cleaning_effectiveness: float = 1.0
    frozen: bool = False
    freeze_after_pass: bool = False
    detection_hint: tuple[float, float] | None = None
    cleaned: float = field(default=0.0, init=False)

    @property
    def label(self) -> str:
        return f'target-{self.target_id}'

    def pixel_center(self, intrinsics=DEFAULT_INTRINSICS) -> tuple[int, int]:
        """Project the target onto the image plane using the mock geometry.

        The mock depth model is a single metric plane: the target sits at
        ``depth_m`` along the optical axis, so ``pixel_to_robot_ground`` inverts
        this projection back to ``(forward=depth_m, left=left_m)`` exactly.
        """
        fx, fy, cx, cy = intrinsics
        u = cx - self.left_m * fx / self.depth_m
        v = cy + 1.0
        return int(round(u)), int(round(v))

    def pixel_radius(self, intrinsics=DEFAULT_INTRINSICS) -> int:
        fx = intrinsics[0]
        return max(1, int(round(self.radius_m * fx / self.depth_m)))

    def effect_pixel_radius(self, intrinsics=DEFAULT_INTRINSICS) -> int:
        if self.decal == 'blob':
            return max(1, int(round(self.pixel_radius(intrinsics) * self.pollution_fraction ** 0.5)))
        if self.decal == 'dark_patch':
            return max(1, int(round(self.pixel_radius(intrinsics) * self.pollution_fraction)))
        return self.pixel_radius(intrinsics)


class VirtualWorld:
    """Collection of virtual ground targets with deterministic cleaning physics."""

    def __init__(self, width: int = 640, height: int = 480,
                 intrinsics=DEFAULT_INTRINSICS, background: tuple[int, int, int] = (125, 145, 155),
                 noise_sigma: float = 0.0, spike_probability: float = 0.0,
                 seed: int = 20260101):
        self.width, self.height = width, height
        self.intrinsics = tuple(float(v) for v in intrinsics)
        self.background = background
        self.noise_sigma = noise_sigma
        self.spike_probability = spike_probability
        self.rng = np.random.default_rng(seed)
        self.targets: list[VirtualTarget] = []
        self.next_target_id = 1
        self.clean_reference_mode = True
        #: Optional ``(visual_score, area_score)`` pair the *scenario layer* may use
        #: to drive classification deterministically. Perception itself never reads
        #: this; the system applies it after detection to exercise the full decision
        #: chain for a chosen pollution level (see scenario_runner.py).
        self.detection_hint: tuple[float, float] | None = None
        #: Fraction of the pollution that survives one cleaning traverse. The mock
        #: cleaner is deliberately imperfect so the compensation path is exercised.
        self.cleaning_retain = 0.45
        self._hint_scale = 'sqrt'

    # ------------------------------------------------------------------ setup
    def add_target(self, forward_m: float, left_m: float, radius_m: float,
                   pollution_fraction: float = 1.0, decal: str = 'blob',
                   base_color: tuple[int, int, int] = (45, 65, 75),
                   depth_m: float = 0.8, blend: float = 1.0,
                   brightness_override: float | None = None,
                   cleaning_effectiveness: float = 1.0,
                   frozen: bool = False,
                   freeze_after_pass: bool = False) -> VirtualTarget:
        if radius_m <= 0 or depth_m <= 0:
            raise ValueError('radius_m and depth_m must be positive')
        if decal not in DECALS:
            raise ValueError(f'unknown decal {decal!r}; expected one of {DECALS}')
        if not 0 < blend <= 1:
            raise ValueError('blend must be in (0, 1]')
        if brightness_override is not None and not 0 <= brightness_override <= 1:
            raise ValueError('brightness_override must be in [0, 1]')
        if not 0 <= cleaning_effectiveness <= 1:
            raise ValueError('cleaning_effectiveness must be in [0, 1]')
        target = VirtualTarget(self.next_target_id, float(forward_m), float(left_m),
                               float(radius_m), float(pollution_fraction),
                               base_color, float(depth_m), decal, float(blend),
                               brightness_override, float(cleaning_effectiveness),
                               bool(frozen), bool(freeze_after_pass),
                               self.detection_hint)
        self.targets.append(target)
        self.next_target_id += 1
        return target

    def set_detection_hint(self, hint: tuple[float, float] | None) -> None:
        """Apply a scenario score hint to the world and to every existing target."""
        self.detection_hint = hint
        for target in self.targets:
            target.detection_hint = hint

    def hint_for(self, target: VirtualTarget) -> tuple[float, float] | None:
        """Current score hint for *target*, scaled down as the stain is cleaned."""
        hint = target.detection_hint or self.detection_hint
        if hint is None:
            return None
        visual, area = hint
        fraction = max(0.0, min(1.0, target.pollution_fraction))
        # The stain keeps its intensity while it covers less ground, so only the
        # area component shrinks (area is roughly proportional to radius squared).
        if self._hint_scale == 'linear':
            scale = fraction
        elif self._hint_scale == 'none':
            scale = 1.0
        else:
            scale = fraction ** 0.5
        return float(visual), float(area * scale)

    def clear_targets(self) -> None:
        self.targets.clear()

    def active_targets(self) -> list[VirtualTarget]:
        return [t for t in self.targets if t.pollution_fraction > 0]

    def target_by_id(self, target_id: int) -> VirtualTarget:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise KeyError(target_id)

    @property
    def pollution_fraction(self) -> float:
        """Aggregate remaining pollution, used by legacy single-stain callers."""
        return max((t.pollution_fraction for t in self.targets), default=0.0)

    @pollution_fraction.setter
    def pollution_fraction(self, value: float) -> None:
        """Scale every target, so ``world.pollution_fraction = 0`` clears the scene."""
        value = max(0.0, min(1.0, float(value)))
        for target in self.targets:
            target.pollution_fraction = value

    def clean(self, passes: int, target_id: int | None = None, retain: float | None = None) -> None:
        retain = self.cleaning_retain if retain is None else retain
        """Apply *passes* cleaning traverses; exposes deterministic success or failure."""
        if passes < 0:
            raise ValueError('passes must be non-negative')
        if not 0 <= retain < 1:
            raise ValueError('retain must be in [0, 1)')
        selected = self.targets if target_id is None else [self.target_by_id(target_id)]
        for target in selected:
            if target.pollution_fraction <= 0 or target.frozen:
                continue
            if target.cleaned > 0 and target.freeze_after_pass:
                continue  # modelled as un-cleanable by the mock cleaning mechanism
            effectiveness = target.cleaning_effectiveness
            target.pollution_fraction = float(
                target.pollution_fraction * retain ** (passes * effectiveness))
            target.cleaned += passes

    # --------------------------------------------------------------- rendering
    def render(self) -> np.ndarray:
        """Render the current scene as a BGR image.

        While ``clean_reference_mode`` is set the scene renders as a clean floor:
        that is what calibration is supposed to observe before any pollution is
        allowed to appear.
        """
        image = np.full((self.height, self.width, 3), self.background, np.uint8)
        mask = np.zeros((self.height, self.width), np.uint8)
        for target in self.active_targets():
            if self.clean_reference_mode:
                continue
            center = target.pixel_center(self.intrinsics)
            radius = target.effect_pixel_radius(self.intrinsics)
            local = np.zeros((self.height, self.width, 3), np.uint8)
            cv2.circle(local, center, radius, target.base_color, -1)
            region = local.any(axis=2)
            if target.blend < 1.0:
                blended = cv2.addWeighted(image, 1.0 - target.blend, local, target.blend, 0)
                image[region] = blended[region]
            else:
                image[region] = local[region]
            if target.brightness_override is not None:
                gray = np.full((self.height, self.width), target.brightness_override, np.float32)
                target_gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                scale = np.where(target_gray > 0, gray / np.maximum(target_gray, 1e-6), 1.0)
                tinted = np.clip(image.astype(np.float32) * scale[:, :, None], 0, 255)
                image[region] = tinted[region].astype(np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
        self._apply_texture(image, mask)
        return image

    def _apply_texture(self, image: np.ndarray, mask: np.ndarray) -> None:
        if self.noise_sigma <= 0 and self.spike_probability <= 0:
            return
        noise = np.zeros((self.height, self.width, 3), np.float32)
        if self.noise_sigma > 0:
            noise += self.rng.normal(0.0, self.noise_sigma, (self.height, self.width, 3))
        if self.spike_probability > 0:
            spikes = self.rng.random((self.height, self.width, 3)) < self.spike_probability
            noise += spikes * self.rng.normal(0.0, 60.0, (self.height, self.width, 3))
        textured = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        image[:, :, :] = textured

    # ------------------------------------------------------------ depth plane
    def render_depth(self, depth_m: float = 0.8, invalid_mask: np.ndarray | None = None,
                     background_depth: float = 0.8) -> np.ndarray:
        depth = np.full((self.height, self.width), float(background_depth), np.float32)
        for target in self.active_targets():
            center = target.pixel_center(self.intrinsics)
            radius = max(target.effect_pixel_radius(self.intrinsics), target.pixel_radius(self.intrinsics))
            cv2.circle(depth, center, radius, float(target.depth_m), -1)
        if invalid_mask is not None:
            depth = np.where(invalid_mask, 0.0, depth).astype(np.float32)
        return depth


def yaw_between(x0: float, y0: float, x1: float, y1: float) -> float:
    """Heading in radians from ``(x0, y0)`` towards ``(x1, y1)``."""
    return math.atan2(y1 - y0, x1 - x0)
