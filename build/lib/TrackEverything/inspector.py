"""A module for managing the relationship between the trackers and detections
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import tool_box as tlbx

class DetectedObj:
    """A class that mange the detection data"""
    def __init__(self,det_score,class_res,bounding_box):
        self.det_score = det_score
        self.class_score=class_res
        self.bounding_box=bounding_box
        self.id_num=-1

    def load_tracker_info(self,tracker):
        """Transfer current data from the tracker to the detector

        Args:
            tracker (TrackerObj): The tracker holding the current data
        """
        self.class_score=tracker.get_class()
        self.id_num=tracker.id_num

    def get_current_class(self):
        """get the current classification class of the detected object

        Returns:
            [int,flota]: The class number and score
        """
        class_num=np.argmax(self.class_score)
        return class_num, self.class_score[class_num]

class TrackerObj:
    """A class that mange the trackers data"""
    def __init__(
            self,
            id_num,
            frame,
            bounding_box,
            inspector_vars:tlbx.InspectorVars
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
        self.tracker = self.inspector_vars.trck_type()
        self.fails=0#failures counting for the tracker
        self.new=True#is the tracker new

        ####
        _ok = self.tracker.init(frame,bounding_box)
        if not _ok:
            #tracker init not ok!
            #adding enough points to be removed
            self.fails+=self.inspector_vars.max_trck_fails+1
            #exit constructor early since the tracker faild
            return

    def update(self,frame):
        """Update tracker position using the new frame

        Args:
            frame (np.ndarray): new frame
        """
        _ok, self.bounding_box = self.tracker.update(frame)
        if not _ok:
            #penalties for faild tracker
            self.fails+=self.inspector_vars.trck_failure_pt
        else:
            #rewards for successful tracker
            self.fails=max(0,self.fails-self.inspector_vars.trck_reward_pt)

    def update_stats(self,class_scor:np.array,detection_scor:float):
        """Update the class score using previous statistics

        Args:
            class_scor (np.array): the class vector score
            detection_scor (float): the confidence score of the detection
        """
        score=detection_scor*class_scor
        self.statistical_calc.update(score)

    def get_class(self):
        """Return the current highest class number(0 based) and it's score

        Returns:
            [class_num,calss_score]:Return the highest class number(0 based) and it's score
        """
        score=self.statistical_calc.get_score()
        return score

    def destroy(self):
        """Bool method to say whethere or not to destroy this tracker

        Returns:
            [bool]: True for destroy False for keeps
        """
        return self.fails>self.inspector_vars.max_trck_fails

def update_trackers(frame,trackers,penalties=0,mark_new=True):
    """Update all the trackers using the new freame

    Args:
        frame ([type]): new frame
        trackers (List[TrackerObj]): List of trakers to update
        penalties (int, optional): Amount of penaltie. Defaults to 0.
        mark_new (bool, optional): Mark tracker as new or old,
        if it's old, later the bounding box will be reset to be more accurate
        using the detection box. Defaults to True.

    Returns:
        List[TrackerObj]: The updated list without destroyed trackers
    """
    for trk in trackers:
        trk.update(frame)
        trk.new=mark_new
        trk.fails+=penalties
    return [tr for tr in trackers if not tr.destroy()]

def assign_detections_to_trackers(
        trackers,
        detections,
        frame,
        inspector_vars,
        iou_overlapping_threshold=-1
    ):
    """Using intersection over union (IOU) of a tracker bounding box and detection bounding box
    as a metric. We solve the linear sum assignment problem (also known as minimum weight
    matching in bipartite graphs) for the IOU matrix using the Hungarian algorithm (also
    known as Munkres algorithm). The machine learning package scipy has a build-in utility
    function that implements the Hungarian algorithm.

    Args:
        trackers (List[TrackerObj]): List of TrackerObj
        detections (List[DetectedObj]): List of DetectedObj
        frame (np.ndarray): current frame
        inspector_vars (InspectorVars): current inspector_vars
        iou_overlapping_threshold (int, optional): for the non_max_suppressions,this is the max
        IOU between detections, and lower than that, the lowest scored detection gets removed.
        Defaults to -1 will skeep the proccess.

    Returns:
        [List[DetectedObj],List[TrackerObj]]: The updated TrackerObj and DetectedObj lists.
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

    #if there are no detections in frame all trackers faild so panilize and return
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
    unmatched_trackers=[]
    unmatched_detections=[]
    #if there are no trackers existing all detection are unmatched
    if len(trackers)<1:
        unmatched_detections=range(len(detections))
    else:
        #create the IOU matrix with trackers and detections
        iou_matrix= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t_index,trk in enumerate(trackers):
            for d_index,det in enumerate(detections):
                iou_matrix[t_index,d_index] = tlbx.box_iou(trk.bounding_box,det.bounding_box)

        # Produces matches
        # Solve the maximizing of the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)
        matched_idx = linear_sum_assignment(-iou_matrix)
        unmatched_trackers=set(range(len(trackers)))-set(matched_idx[0])
        unmatched_detections=set(range(len(detections)))-set(matched_idx[1])

        # For creating trackers we consider any detection with an
        # overlap less than inspector_vars.iou_paring_threshold to signifiy the existence of
        # an untracked object
        for i in range(len(matched_idx[0])):
            t_index=matched_idx[0][i]
            d_index=matched_idx[1][i]
            if iou_matrix[t_index,d_index]<inspector_vars.iou_paring_threshold:
                unmatched_trackers.add(t_index)
                unmatched_detections.add(d_index)
            else:
                matches.append([d_index,t_index])#add the matched pairs detections first

    #importent to be last so the indecise wont change
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

    #update faild trakers and delete destroyable ones
    for t_index in unmatched_trackers:
        trackers[t_index].fails+=inspector_vars.penaltie()
    trackers = [trk for trk in trackers if not trk.destroy()]

    return detections,trackers

def transfer_matches(detections,trackers,matches,frame,inspector_vars):
    """adding the current data to the stats of the tracker
    and then transfering the new calculated stats from matched trackers to their detectors

    Args:
        detections (List[DetectedObj]): List of DetectedObj
        trackers (List[TrackerObj]): List of TrackerObj
        matches (List[[det_id,trck_id]]): List of pairs indecise matching the trackers and detectors
        frame (np.ndarray): The current frame (used for updating old trackers by creating new ones)
        only if inspector_vars.trck_resizing=True
        inspector_vars (InspectorVars): Used for creating new trackers
        if inspector_vars.trck_resizing=True
    """
    for i in matches:
        det=detections[i[0]]
        trk=trackers[i[1]]
        trk.update_stats(det.class_score,det.det_score)
        if inspector_vars.trck_resizing:
            #resize the tracker if it is an old one
            if not trk.new:
                #for passing the saved statistics and methods param
                inspector_vars.saved_stat_calc_holder=trk.statistical_calc
                #create new resized tracker with old stats and new detection data
                trackers[i[1]]=TrackerObj(
                trk.id_num,
                frame,
                det.bounding_box,
                inspector_vars,
                )
                #reset the global statistics to None
                inspector_vars.saved_stat_calc_holder=None

        det.load_tracker_info(trk)
