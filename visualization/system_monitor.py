"""Lightweight OpenCV monitoring panel (debugging, demos and screenshot material).

Layout::

    +--------------------+--------------------+
    | RGB + bounding box | Detection mask     |
    +--------------------+--------------------+
    | State / FPS / VisualScore / HumidityScore / AreaScore / PollutionScore
    | Level / Task ID / Retry / Path length
    +-----------------------------------------+
"""
import cv2
import numpy as np

from visualization.camera_view import annotate

WINDOW = 'Snow Clean Robot'


class SystemMonitor:
    """Never crashes a headless run: display errors degrade to no-op."""

    def __init__(self, headless: bool = True, window: str = WINDOW,
                 show_scores: bool = True):
        self.headless = bool(headless)
        self.window = window
        self.show_scores = show_scores
        self.frames_rendered = 0
        self.last_panel: np.ndarray | None = None
        self.display_error: str | None = None

    # ------------------------------------------------------------------ rendering
    def render(self, frame, mask, regions, state: str, task=None, fps: float = 0.0,
               region=None, path_length: float | None = None) -> np.ndarray:
        """Build the panel image; also used by tests in headless mode."""
        image = annotate(frame, regions)
        if mask is None:
            mask = np.zeros(image.shape[:2], np.uint8)
        mask = cv2.resize(np.asarray(mask), (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack((image, panel))
        self._draw_overlay(display, image, state, task, fps, region, path_length)
        self.frames_rendered += 1
        return display

    def _draw_overlay(self, display, image, state, task, fps, region, path_length) -> None:
        lines = [f'State: {state}', f'FPS: {fps:.1f}']
        if region is not None:
            lines += [f'VisualScore: {region.visual_score:.3f}',
                      f'AreaScore: {region.area_score:.3f}']
            if region.pollution_score is not None:
                lines.append(f'PollutionScore: {region.pollution_score:.3f}')
            if region.pollution_level is not None:
                lines.append(f'Level: {region.pollution_level}')
        if task is not None:
            lines.append(f'Task ID: {task.task_id}  Retry: {task.retry_count}')
            lines.append(f'Level: {task.pollution_level}  Score: {task.pollution_score:.3f}')
        if path_length is not None:
            lines.append(f'Path: {path_length:.2f} m')
        for index, text in enumerate(lines):
            origin = (10, 25 + index * 20)
            cv2.putText(display, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(display, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ------------------------------------------------------------------ display
    def show(self, frame, mask, regions, state: str, task=None, fps: float = 0.0,
             region=None, path_length: float | None = None) -> None:
        if self.headless:
            return
        try:
            display = self.render(frame, mask, regions, state, task, fps, region, path_length)
            self.last_panel = display
            cv2.imshow(self.window, display)
            cv2.waitKey(1)
        except cv2.error as exc:  # no display server / window backend missing
            self.display_error = str(exc)
            self.headless = True

    def close(self) -> None:
        if self.headless:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:  # pragma: no cover - defensive
            pass
