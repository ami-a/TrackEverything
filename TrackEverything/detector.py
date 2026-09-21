"""The main entry point: a detector that is updated frame by frame."""
from typing import Optional

import numpy as np

from . import inspector
from . import tool_box as tlbx
from . import visualization_utils as visu


class Detector:
    """Perform detection and classification, taking previous frames into account.

    Each detected object is matched to a persistent tracker, so its identity and its
    classification score survive across frames even when a single frame is ambiguous.
    """
    def __init__(
            self,
            det_vars: Optional[tlbx.DetectionVars] = None,
            class_vars: Optional[tlbx.ClassificationVars] = None,
            inspector_vars: Optional[tlbx.InspectorVars] = None,
            visualization_vars: Optional[visu.VisualizationVars] = None,
        ):
        """Build a detector.

        Args:
            det_vars (DetectionVars): the detection model and its pre/post-processing.
            class_vars (ClassificationVars): the classification model and its
                pre/post-processing. Omit it to run detection only.
            inspector_vars (InspectorVars): tracking and statistics configuration.
            visualization_vars (VisualizationVars): drawing configuration.

        Note:
            Each argument defaults to a freshly constructed object. They are not shared
            between ``Detector`` instances, so two detectors never share tracker ids or
            accumulated statistics.
        """
        self.det_vars=tlbx.DetectionVars() if det_vars is None else det_vars
        self.class_vars=tlbx.ClassificationVars() if class_vars is None else class_vars
        #load models
        self.det_vars.load_model()
        self.class_vars.load_model()
        #inspector parameters
        self.ins_vars=tlbx.InspectorVars() if inspector_vars is None else inspector_vars
        #set arrays
        self.trackers: list[inspector.TrackerObj]=[]
        self.detections: list[inspector.DetectedObj]=[]
        #visualization parameters
        self.vis_var=(visu.VisualizationVars() if visualization_vars is None
                      else visualization_vars)

    def update(self, img: np.ndarray) -> None:
        """Find new detections and update old ones using statistical configuration
        and data from previous frames.

        Args:
            img (np.ndarray): current frame.
        """
        self.detections=[]#clear detection from last frame
        #Get detections that are over the threshold
        detection_arr=self.det_vars.detection_proccessing(
            self.det_vars,
            img,
        )
        #if detection failed
        if not detection_arr:
            #update trackers
            self.trackers =inspector.update_trackers(
                img,
                self.trackers,
                self.ins_vars.penaltie(),
                mark_new=True,
            )
            return
        #classify each detection
        classified_det_arr=tlbx.get_classified_detection_array(
            self.class_vars.class_model,
            img,
            detection_arr,
            self.class_vars.class_proccessing,
            )

        #Put classified detection in detections as DetectedObj
        for ind in range(len(classified_det_arr['det'])):
            self.detections.append(
                inspector.DetectedObj(
                classified_det_arr['det'][ind][0],
                classified_det_arr['class_res'][ind],
                classified_det_arr['det'][ind][1],
                )
            )

        #Update detection and trackers using saved trackers
        self.detections,self.trackers =inspector.assign_detections_to_trackers(
            self.trackers,
            self.detections,
            img,
            self.ins_vars,
            iou_overlapping_threshold=self.det_vars.non_max_sup_threshold,
            )

    def draw_visualization(
            self,
            img: np.ndarray,
            original_size: Optional[tuple[int, int]] = None,
        ) -> None:
        """Draw bounding boxes and labels around targets, in place.

        Args:
            img (np.ndarray): frame to draw on.
            original_size (width, height): size of the original image the bounding boxes
                were created against.
        """
        visu.draw_boxes(img,self.detections,self.trackers,self.vis_var,org_img_size=original_size)

    def get_current_class_summary(self) -> dict[int, int]:
        """A dictionary containing the total number of current detections by class.

        Returns:
            Dict[int, int]: number of current detections keyed by class number.
        """
        class_summary: dict[int, int] ={}
        for detection in self.detections:
            classification=int(np.argmax(detection.class_score))
            class_summary[classification]=class_summary.get(classification,0)+1
        return class_summary
