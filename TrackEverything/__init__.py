"""TrackEverything - tracking and temporal smoothing for any detection model.

Add OpenCV trackers, Hungarian detection-to-track matching and statistical score
smoothing on top of any detection or classification model that exposes a
``.predict()`` method, from TensorFlow, PyTorch or anywhere else.

Typical use::

    from TrackEverything import Detector, DetectionVars

    detector = Detector(det_vars=DetectionVars(detection_model=my_model))
    detector.update(frame)
    detector.draw_visualization(frame)
"""
# Keep this a plain string literal on its own line: the build backend reads it
# statically from the AST, and only falls back to importing the package (which
# would drag cv2 and numpy into the build) if it cannot.
__version__ = "2.0.0"

# Import order is alphabetical (enforced by the linter) and safe: importing
# `.detector` first pulls the rest of the chain in dependency order anyway.
from .detector import Detector
from .inspector import (
    DetectedObj,
    TrackerObj,
    assign_detections_to_trackers,
    update_trackers,
)
from .statistical_methods import (
    StatisticalCalculator,
    StatMethods,
    StatParams,
    cumulative_moving_average,
    exponential_moving_average,
    finite_moving_average,
    no_average,
)
from .tool_box import (
    ClassificationVars,
    DetectionVars,
    Ids,
    InspectorVars,
    available_trackers,
    box_iou,
    classify_detection,
    crop_np_image,
    cv2_bbox_reshape,
    cv2_bbox_to_tf_bbox,
    get_classified_detection_array,
    get_detection_array,
    get_tracker,
    load_tf_model,
    non_max_suppressions,
    resolve_tracker_factory,
)
from .visualization_utils import STANDARD_COLORS, VisualizationVars, draw_boxes

__all__ = [  # noqa: RUF022 - grouped by role, which reads better than alphabetical
    "__version__",
    # main entry point
    "Detector",
    # configuration
    "DetectionVars",
    "ClassificationVars",
    "InspectorVars",
    "VisualizationVars",
    # statistics
    "StatisticalCalculator",
    "StatParams",
    "StatMethods",
    "no_average",
    "cumulative_moving_average",
    "finite_moving_average",
    "exponential_moving_average",
    # detection and tracker objects
    "DetectedObj",
    "TrackerObj",
    "Ids",
    # tracker resolution
    "get_tracker",
    "resolve_tracker_factory",
    "available_trackers",
    # model helpers
    "get_detection_array",
    "get_classified_detection_array",
    "classify_detection",
    "load_tf_model",
    # geometry helpers
    "box_iou",
    "non_max_suppressions",
    "crop_np_image",
    "cv2_bbox_reshape",
    "cv2_bbox_to_tf_bbox",
    # inspection and drawing
    "update_trackers",
    "assign_detections_to_trackers",
    "draw_boxes",
    "STANDARD_COLORS",
]
