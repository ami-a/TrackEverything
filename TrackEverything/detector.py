"""Main model for creating a detector that can be updated
frame by frame
"""
import numpy as np
from . import inspector
from . import tool_box as tlbx
from . import visualization_utils as visu

class Detector:
    """A class that can be used to perform a detection and classification
    taking into account previous detection and classification.
    """
    def __init__(
            self,
            det_vars:tlbx.DetectionVars=tlbx.DetectionVars(),
            class_vars:tlbx.ClassificationVars=tlbx.ClassificationVars(),
            inspector_vars:tlbx.InspectorVars=tlbx.InspectorVars(),
            visualization_vars:visu.VisualizationVars=visu.VisualizationVars(),
        ):
        self.det_vars=det_vars
        self.class_vars=class_vars
        #load models
        self.det_vars.load_model()
        self.class_vars.load_model()
        #inspector parameters
        self.ins_vars=inspector_vars
        #set arrays
        self.trackers=[]
        self.detections=[]
        #visualization parameters
        self.vis_var=visualization_vars

    def update(self,img):
        """Find new detections and update old detection using statistical
        configurations and data from previous frames

        Args:
            img (np.ndarray): current frame
        """
        self.detections=[]#clear detection from last frame
        #Get detections that are over the threshold
        detection_arr=self.det_vars.detection_proccessing(
            img,
        )
        #if detection faild
        if not detection_arr:
            #update trakers
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

    def draw_visualization(self,img,original_size=None):
        """Draw bounding boxes and lables around targets using visualization
        varaubles as settings

        Args:
            img (np.array): frame to draw on
            original_size (width,height): of the original image the bounding box where created on
        """
        visu.draw_boxes(img,self.detections,self.trackers,self.vis_var,org_img_size=original_size)

    def get_current_class_summary(self):
        """A dictionary containig the total number of current detections by class
        (only classes with more than one detection exist)

        Returns:
            Dictionary: dictionary containig the total number of current detections by class
        """
        class_summary ={}
        for detection in self.detections:
            classification=np.argmax(detection.class_score)
            class_summary[classification]=class_summary.get(classification,0)+1
        return class_summary
