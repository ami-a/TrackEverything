"""A runnable demo that needs no model, no video file and no download.

It synthesises a short clip of moving shapes, runs the full TrackEverything
pipeline over it with a toy detector and classifier, and writes an annotated
video plus a still frame.

    python examples/demo.py

The point is to show the two things the library adds to a plain per-frame
detector: ids that persist while an object moves, and classification scores
that are smoothed over time instead of jumping frame to frame.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from TrackEverything import (
    ClassificationVars,
    Detector,
    DetectionVars,
    InspectorVars,
    StatisticalCalculator,
    StatMethods,
    VisualizationVars,
)

WIDTH, HEIGHT = 640, 360
FRAMES = 120
LABELS = ["cool", "warm"]


class Scene:
    """Three shapes, each oscillating within its own lane.

    The lanes never overlap, so any identity switch in the output is the
    library's doing rather than two objects genuinely occluding each other.
    """

    def __init__(self):
        self.objects = [
            {"lane": 0, "phase": 0.0, "speed": 1.0, "size": (64, 78), "warm": True},
            {"lane": 1, "phase": 2.1, "speed": 0.7, "size": (58, 66), "warm": False},
            {"lane": 2, "phase": 4.2, "speed": 1.3, "size": (52, 74), "warm": True},
        ]

    def boxes_at(self, t: float) -> list[tuple[int, int, int, int]]:
        boxes = []
        lane_height = HEIGHT / len(self.objects)
        for obj in self.objects:
            box_w, box_h = obj["size"]
            # horizontal sweep, kept clear of the frame edges
            travel = WIDTH - box_w - 40
            angle = obj["phase"] + t * 2 * math.pi * obj["speed"]
            x = 20 + travel * (0.5 + 0.5 * math.sin(angle))
            # vertical centre of this object's lane
            y = obj["lane"] * lane_height + (lane_height - box_h) / 2
            boxes.append((int(x), int(y), box_w, box_h))
        return boxes

    def render(self, t: float) -> np.ndarray:
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        # a faint gradient so the trackers have texture to lock onto
        gradient = np.linspace(0, 45, WIDTH, dtype=np.uint8)
        frame += gradient[None, :, None]
        for obj, (x, y, box_w, box_h) in zip(self.objects, self.boxes_at(t)):
            color = (60, 90, 220) if obj["warm"] else (210, 160, 60)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, -1)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (250, 250, 250), 2)
            cv2.circle(frame, (x + box_w // 2, y + box_h // 3), 9, (250, 250, 250), -1)
        return frame


class ToyDetector:
    """Stands in for a real detection model.

    Finds the bright-outlined shapes by thresholding, and jitters the boxes a
    little so the matching step has something non-trivial to do.
    """

    def __init__(self, rng):
        self.rng = rng

    def predict(self, batch):
        frame = batch[0] if batch.ndim == 4 else batch
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rows = []
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w * box_h < 900:
                continue
            jitter = self.rng.integers(-3, 4, size=2)
            confidence = float(self.rng.uniform(0.75, 0.99))
            rows.append(
                [confidence, x + jitter[0], y + jitter[1],
                 x + box_w + jitter[0], y + box_h + jitter[1]]
            )
        return rows


class ToyClassifier:
    """Stands in for a real classifier, and is deliberately unreliable.

    It reads the dominant hue of the crop, which is the right answer, but 25%
    of the time it returns noise. A per-frame pipeline would flicker; the
    temporal smoothing is what holds the label steady.
    """

    def __init__(self, rng, noise=0.25):
        self.rng = rng
        self.noise = noise

    def predict(self, images):
        out = []
        for img in images:
            if img.size == 0:
                out.append([0.5, 0.5])
                continue
            if self.rng.random() < self.noise:
                warm = self.rng.random()
                out.append([1 - warm, warm])
                continue
            blue, _green, red = (float(img[..., i].mean()) for i in range(3))
            warm = 0.88 if red > blue else 0.12
            out.append([1 - warm, warm])
        return np.asarray(out, dtype=float)


def build_detector(rng, smoothing: StatMethods) -> Detector:
    return Detector(
        det_vars=DetectionVars(
            detection_model=ToyDetector(rng),
            detection_threshold=0.5,
            non_max_sup_threshold=0.3,
        ),
        class_vars=ClassificationVars(class_model=ToyClassifier(rng)),
        inspector_vars=InspectorVars(
            trck_type="CSRT",
            max_trck_fails=12,
            stat_calc=StatisticalCalculator(method=smoothing, class_num=2),
        ),
        visualization_vars=VisualizationVars(
            labels=LABELS,
            show_ids=True,
            uncertainty_threshold=0.15,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_output"),
        help="directory for the annotated video and still (default: demo_output)",
    )
    parser.add_argument(
        "--smoothing",
        choices=[m.name for m in StatMethods],
        default="EMA",
        help="temporal smoothing method (default: EMA)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    scene = Scene()
    detector = build_detector(rng, StatMethods[args.smoothing])

    video_path = args.out / "demo.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (WIDTH, HEIGHT)
    )

    seen_ids: set[float] = set()
    for i in range(FRAMES):
        frame = scene.render(i / FRAMES)
        detector.update(frame)
        detector.draw_visualization(frame)
        seen_ids.update(det.id_num for det in detector.detections)
        writer.write(frame)

        if i == FRAMES - 1:
            cv2.imwrite(str(args.out / "demo.png"), frame)

        if i % 30 == 0:
            summary = detector.get_current_class_summary()
            readable = {LABELS[k]: v for k, v in summary.items() if k < len(LABELS)}
            print(
                f"frame {i:3d} | tracked {len(detector.detections)} "
                f"| ids {sorted(d.id_num for d in detector.detections)} | {readable}"
            )

    writer.release()

    print(f"\n3 objects in the scene, {len(seen_ids)} distinct ids issued overall.")
    print("An id count close to 3 means identities survived the whole clip.")
    print(f"\nWrote {video_path} and {args.out / 'demo.png'}")


if __name__ == "__main__":
    main()
