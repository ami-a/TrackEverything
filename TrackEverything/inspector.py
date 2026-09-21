"""Matching between trackers and detections.

This is where the tracking half and the detection half of the pipeline meet:
trackers are advanced with the new frame, detections are matched to them by
solving the assignment problem over an IOU matrix, and the surviving pairs
exchange identity and statistics.
"""
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import tool_box as tlbx


def _cv2_bbox(bounding_box: Sequence[float]) -> tuple[int, int, int, int]:
    """Coerce a bounding box to the plain integer tuple OpenCV trackers require.

    Detection models routinely emit floats or numpy scalars, which OpenCV rejects
    with an unhelpful "Can't parse 'boundingBox'" overload error.

    Args:
        bounding_box ((xmin,ymin,width,height)): the box to convert.

    Returns:
        Tuple[int, int, int, int]: the box as plain Python ints.
    """
    xmin,ymin,box_w,box_h=bounding_box
    return (round(float(xmin)), round(float(ymin)),
            max(1, round(float(box_w))), max(1, round(float(box_h))))

class DetectedObj:
    """A class that manages the detection data."""
    def __init__(self, det_score, class_res, bounding_box: Sequence[float]):
        self.det_score = det_score
        self.class_score=class_res
        self.bounding_box=bounding_box
        self.id_num=-1

    def load_tracker_info(self, tracker: "TrackerObj") -> None:
        """Transfer current data from the tracker to the detection.

        Args:
            tracker (TrackerObj): The tracker holding the current data.
        """
        self.class_score=tracker.get_class()
        self.id_num=tracker.id_num

    def get_current_class(self) -> tuple[int, float]:
        """Get the current classification class of the detected object.

        Returns:
            Tuple[int, float]: The class number and its score.
        """
        class_num=int(np.argmax(self.class_score))
        return class_num, self.class_score[class_num]

class TrackerObj:
    """A class that manages the tracker data."""
    def __init__(
            self,
            id_num: float,
            frame: np.ndarray,
            bounding_box: Sequence[float],
            inspector_vars: tlbx.InspectorVars,
        ):
        self.inspector_vars=inspector_vars
        #check if there is statistical data in the inspector_vars
        #or create a new statistical calculator.
        if inspector_vars.saved_stat_calc_holder is None:
            self.statistical_calc=inspector_vars.stat_calc.__copy__()
        else:
            self.statistical_calc=inspector_vars.saved_stat_calc_holder
        self.id_num = id_num
        self.bounding_box=bounding_box
        self.tracker = self.inspector_vars.get_tracker_factory()()
        self.fails=0#failures counting for the tracker
        self.new=True#is the tracker new

        try:
            _ok = self.tracker.init(frame,_cv2_bbox(bounding_box))
            #OpenCV 4.5.1+ returns None from init() instead of a bool, and
            #`not None` would mark every new tracker as failed
            _ok = True if _ok is None else _ok
        except cv2.error as exception:
            print(f"Tracker {id_num} failed to initialise: {exception}")
            _ok=False
        if not _ok:
            #tracker init not ok!
            #adding enough points to be removed
            self.fails+=self.inspector_vars.max_trck_fails+1

    def update(self, frame: np.ndarray) -> None:
        """Update the tracker position using the new frame.

        Args:
            frame (np.ndarray): new frame.
        """
        _ok, self.bounding_box = self.tracker.update(frame)
        if not _ok:
            #penalties for a failed tracker
            self.fails+=self.inspector_vars.trck_failure_pt
        else:
            #rewards for a successful tracker
            self.fails=max(0,self.fails-self.inspector_vars.trck_reward_pt)

    def update_stats(self, class_scor: np.ndarray, detection_scor: float) -> None:
        """Update the class score using previous statistics.

        Args:
            class_scor (np.ndarray): the class vector score.
            detection_scor (float): the confidence score of the detection.
        """
        score=detection_scor*class_scor
        self.statistical_calc.update(score)

    def get_class(self) -> np.ndarray:
        """Return the current smoothed class score vector.

        Returns:
            np.ndarray: the current statistical score for each class.
        """
        return self.statistical_calc.get_score()

    def destroy(self) -> bool:
        """Whether or not to destroy this tracker.

        Returns:
            bool: True for destroy, False for keep.
        """
        return self.fails>self.inspector_vars.max_trck_fails

def update_trackers(
        frame: np.ndarray,
        trackers: list[TrackerObj],
        penalties: float = 0,
        mark_new: bool = True,
    ) -> list[TrackerObj]:
    """Update all the trackers using the new frame.

    Args:
        frame (np.ndarray): new frame.
        trackers (List[TrackerObj]): List of trackers to update.
        penalties (float, optional): Amount of penalty. Defaults to 0.
        mark_new (bool, optional): Mark the tracker as new or old; if it is old, the
            bounding box will later be reset to be more accurate using the detection
            box. Defaults to True.

    Returns:
        List[TrackerObj]: The updated list without destroyed trackers.
    """
    for trk in trackers:
        trk.update(frame)
        trk.new=mark_new
        trk.fails+=penalties
    return [tr for tr in trackers if not tr.destroy()]

def assign_detections_to_trackers(
        trackers: list[TrackerObj],
        detections: list[DetectedObj],
        frame: np.ndarray,
        inspector_vars: tlbx.InspectorVars,
        iou_overlapping_threshold: float = -1,
    ) -> tuple[list[DetectedObj], list[TrackerObj]]:
    """Match detections to trackers by solving the assignment problem.

    Uses the intersection over union (IOU) of a tracker bounding box and a detection
    bounding box as a metric. We solve the linear sum assignment problem (also known
    as minimum weight matching in bipartite graphs) for the IOU matrix using the
    Hungarian algorithm (also known as the Munkres algorithm). SciPy has a built-in
    utility function that implements it.

    Args:
        trackers (List[TrackerObj]): List of TrackerObj.
        detections (List[DetectedObj]): List of DetectedObj.
        frame (np.ndarray): current frame.
        inspector_vars (InspectorVars): current inspector_vars.
        iou_overlapping_threshold (float, optional): for the non_max_suppressions; this is
            the max IOU between detections, above which the lowest scored detection gets
            removed. Defaults to -1, which skips the process.

    Returns:
        Tuple[List[DetectedObj], List[TrackerObj]]: The updated detection and tracker lists.

    TODO:optimize reforming of trackers and detections try np.fromiter((f(xi) for xi in x),x.dtype)
    """

    #mark all existing trackers and update them
    #add penalties to all trackers if there are no detections
    trackers=update_trackers(
        frame,
        trackers,
        penalties=inspector_vars.penaltie() if len(detections)<1 else 0,
        mark_new=False,
        )

    #if there are no detections in frame all trackers failed so penalize and return
    if len(detections)<1:
        return [],trackers

    #remove overlapping detections
    if iou_overlapping_threshold>=0:
        detections=tlbx.non_max_suppressions(
            detections,
            threshold_iou=iou_overlapping_threshold,
            )

    matches=[]
    matched_idx=[]
    unmatched_trackers=set()
    unmatched_detections=set()
    #if there are no trackers existing all detections are unmatched
    if len(trackers)<1:
        unmatched_detections=set(range(len(detections)))
    else:
        #create the IOU matrix with trackers and detections
        iou_matrix= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t_index,trk in enumerate(trackers):
            for d_index,det in enumerate(detections):
                iou_matrix[t_index,d_index] = tlbx.box_iou(trk.bounding_box,det.bounding_box)

        # Produces matches
        # Solve the maximizing of the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as the Munkres algorithm)
        matched_idx = linear_sum_assignment(-iou_matrix)
        unmatched_trackers=set(range(len(trackers)))-set(matched_idx[0])
        unmatched_detections=set(range(len(detections)))-set(matched_idx[1])

        # For creating trackers we consider any detection with an
        # overlap less than inspector_vars.iou_paring_threshold to signify the existence of
        # an untracked object
        for i in range(len(matched_idx[0])):
            t_index=matched_idx[0][i]
            d_index=matched_idx[1][i]
            if iou_matrix[t_index,d_index]<inspector_vars.iou_paring_threshold:
                unmatched_trackers.add(t_index)
                unmatched_detections.add(d_index)
            else:
                matches.append([d_index,t_index])#add the matched pairs detections first

    #important to be last so the indices wont change
    #Creates new trackers for the unmatched detections
    for d_index in unmatched_detections:
        trackers+=[TrackerObj(
            inspector_vars.trck_id_generator.get_next_id(),# gives new id to the detection
            frame,
            detections[d_index].bounding_box,
            inspector_vars,
            )]
        matches+=[[d_index,len(trackers)-1]]#add the new pair to the matches

    transfer_matches(detections,trackers,matches,frame,inspector_vars)

    #update failed trackers and delete destroyable ones
    for t_index in unmatched_trackers:
        trackers[t_index].fails+=inspector_vars.penaltie()
    trackers = [trk for trk in trackers if not trk.destroy()]

    return detections,trackers

def transfer_matches(
        detections: list[DetectedObj],
        trackers: list[TrackerObj],
        matches: list[list[int]],
        frame: np.ndarray,
        inspector_vars: tlbx.InspectorVars,
    ) -> None:
    """Add the current data to the tracker statistics, then transfer the newly
    calculated statistics from the matched trackers back to their detections.

    Args:
        detections (List[DetectedObj]): List of DetectedObj.
        trackers (List[TrackerObj]): List of TrackerObj.
        matches (List[[det_id,trck_id]]): List of index pairs matching trackers and
            detections.
        frame (np.ndarray): The current frame, used for updating old trackers by creating
            new ones, only if ``inspector_vars.trck_resizing=True``.
        inspector_vars (InspectorVars): Used for creating new trackers if
            ``inspector_vars.trck_resizing=True``.
    """
    for i in matches:
        det=detections[i[0]]
        trk=trackers[i[1]]
        trk.update_stats(det.class_score,det.det_score)
        if inspector_vars.trck_resizing and not trk.new:
            #resize the tracker if it is an old one
            #for passing the saved statistics and methods param
            inspector_vars.saved_stat_calc_holder=trk.statistical_calc
            #create new resized tracker with old stats and new detection data
            trackers[i[1]]=TrackerObj(
                trk.id_num,
                frame,
                det.bounding_box,
                inspector_vars,
            )
            #reset the saved statistics holder to None
            inspector_vars.saved_stat_calc_holder=None
            trk=trackers[i[1]]

        det.load_tracker_info(trk)
