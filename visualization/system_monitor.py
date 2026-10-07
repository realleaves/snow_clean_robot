import cv2
import numpy as np
from visualization.camera_view import annotate


class SystemMonitor:
    def __init__(self, headless: bool):
        self.headless = headless

    def show(self, frame, mask, regions, state: str, task=None, fps: float = 0) -> None:
        if self.headless:
            return
        image = annotate(frame, regions)
        if mask is None:
            mask = np.zeros(image.shape[:2], np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack((image, panel))
        details = f'State: {state}  FPS: {fps:.1f}'
        if task:
            details += f'  Task: {task.task_id}  Score: {task.pollution_score:.2f}  Level: {task.pollution_level}  Retry: {task.retry_count}'
        cv2.putText(display, details, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        cv2.imshow('Snow Clean Robot', display)
        cv2.waitKey(1)

    def close(self) -> None:
        if not self.headless:
            cv2.destroyAllWindows()
