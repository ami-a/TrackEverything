"""A module that handels the models and offers some helpfull methods"""
from dataclasses import dataclass, field
from typing import Iterable, Any, Callable
import cv2
import numpy as np
from . import statistical_methods as stat_m

#region methods
def load_tf_model(path):
    """Load tensorflow model from path

    Args:
        path (str): path to the model

    Returns:
        [tf model]: return a tf model object
    """
    #import inside method to remove the requierment of tf
    import tensorflow as tf# pylint: disable=import-outside-toplevel
    print(f"Loading model from {path}...")
    model=tf.keras.models.load_model(path)
    print("Model loaded!")
    return model

def get_detection_array(detection_var,image):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    In this specific function we assume that the detection model return prediction as
    an List if lists in the format of [conf xmin ymin xmax ymax]
    Args:
        detection_var (DetectionVars): Class for defining the detection variables
        image (np.ndarray): current image
    """
    height,width, = image.shape[:2]
    detections=detection_var.detection_model.predict(image[np.newaxis, ...])
    return [
            [x[0],(max(x[1],0),max(x[2],0),min(x[3]-x[0],width),min(x[4]-x[1],height))]
            for x in detections if
            x[0]>detection_var.detection_threshold and x[3]>x[0] and x[4]>x[1]
        ]

def get_classified_detection_array(model,image,detection_array,classiffing_method):
    """For each detection in detection_array it returns dic with
    a detection from detection array and a class result with scores
        {
        'det':detection array,
        'class_res':numpy NxM vector where N num of images, M num of classes and filled with scores
        }

    If there is no classification model it returns all detection with class 0 and class score 1
    Args:
        model (tensorflow model obj): classification model
        image (np.ndarray): current image
        detection_array (List[[confidence,(xmin,ymin,width,height)]]): detections array
    Returns: dictionary
    {'det':detection array,'class_res':numpy NxM vector where N num of images, M num of classes
    and filled with scores}
    """
    dic={'det':detection_array}
    if model is None:
        dic['class_res']=np.ones(len(detection_array))
    else:
        det_images=map(lambda x:crop_np_image(x[1],image),detection_array)
        dic['class_res']=classiffing_method(model,list(det_images))
    return dic

def crop_np_image(coordinates,img):
    """Copying a crop box portion from image

    Args:
        coordinates ((xmin,ymin,width,height)): coordinates for box
        img (np.ndarray): image
    Returns:
        cropt np image
    """
    xmin,ymin,width,height=coordinates
    return img[ymin:ymin+height,xmin:xmin+width,:]

def classify_detection(model,det_images,size=None,interpolation = cv2.INTER_LINEAR):
    """Classify a batch of images

    Args:
        model (tensorflow model): classification model
        det_images (list[np.array]): list of images in numpy array format to classify
        size (tuple, optional): size to resize to, 1-D int32 Tensor of 2 elements:
            new_height, new_width (if None then no resizing). Defaults is None.
            (In custome function you can use model.inputs[0].shape.as_list()
            and set size to default)
    Returns:
        Numpy NxM vector where N num of images, M num of classes and filled with scores.

        For example two images (car,plan) with three possible classes (car,plan,lion)
        that are identify currectly with 90% in the currect category and the rest is devided equally
        will return [[0.9,0.05,0.05],[0.05,0.9,0.05]].
    """
    #resize bounding box capture to fit classification model
    if size is not None:
        det_images=np.asarray(
            [
                cv2.resize(img, size, interpolation = interpolation) for img in det_images
            ]
        )
    predictions=model.predict(det_images)#make sure image at currect format like /255.0
    #if class is binary make sure size is 2
    if len(predictions)>0 and len(predictions[0])<2:
        reshaped_pred=np.ones((len(predictions),2))
        #size of classification list is 1 so turn it to 2
        for ind,pred in enumerate(predictions):
            reshaped_pred[ind,:]=1-pred,pred
        predictions=reshaped_pred
    return predictions

def get_tracker(trck_type):
    """Get Tracker types in OpenCV"""
    trck_method=None
    if trck_type=="CSRT":
        trck_method=cv2.TrackerCSRT_create
    if trck_type=="kcf":
        trck_method=cv2.TrackerKCF_create
    if trck_type=="boosting":
        trck_method=cv2.TrackerBoosting_create
    if trck_type=="mil":
        trck_method=cv2.TrackerMIL_create
    if trck_type=="tld":
        trck_method=cv2.TrackerTLD_create
    if trck_type=="medianflow":
        trck_method=cv2.TrackerMedianFlow_create
    if trck_type=="mosse":
        trck_method=cv2.TrackerMOSSE_create
    return trck_method

def non_max_suppressions(detections,threshold_iou=0.3):
    """removes overlapping detections using the non max suppressions IOU method

    Args:
        detections (List[DetectedObj]): List of DetectedObj

    Returns:
        List[DetectedObj]: clean List of DetectedObj
    TODO:try with tf.image.non_max_suppression
    """
    #checking if detection score is a vector or float
    if len(detections)>0:
        # pylint: disable=isinstance-second-argument-not-valid-type
        eval_func=max if isinstance(detections[0].det_score, Iterable) else abs
    #removing the one with lowest score
    i=0
    while i<len(detections):
        j=i+1
        while j<len(detections):
            iou=box_iou(detections[i].bounding_box,detections[j].bounding_box)
            if iou>threshold_iou:
                if eval_func(detections[i].det_score)>=eval_func(detections[j].det_score):
                    del detections[j]
                    continue
                del detections[i]
                i-=1
                break
            j+=1
        i+=1
    return detections

def cv2_bbox_to_tf_bbox(cv2_bbox,width,height):
    """tensorflow bounding box format is [y_min, x_min, y_max, x_max]
    with bounding box coordinates as floats in [0.0, 1.0] relative to the width and the height
    of the underlying image.

    Args:
        cv2_bbox (List[xmin,ymin,box_width,box_height]): Bounding box.
        width (int): width of the image.
        height (int): height of the image.

    Returns:
        List[y_min, x_min, y_max, x_max]: tensorflow bounding box coordinates.
    """
    return np.array([
    cv2_bbox[1]/height,
    cv2_bbox[0]/width,
    min((cv2_bbox[1]+cv2_bbox[3])/height,1),
    min((cv2_bbox[0]+cv2_bbox[2])/width,1),
    ])

def cv2_bbox_reshape(box):
    """Converts bounding box format of [xmin,ymin,box_width,box_height] to
    [x_min,y_min,x_max,y_max]

    Args:
        box (List[xmin,ymin,box_width,box_height]): Bounding box.

    Returns:
        List[x_min,y_min,x_max,y_max]: Bounding box.
    """
    return (box[0],box[1],box[0]+box[2],box[1]+box[3])

def box_iou(box_a, box_b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    box_a[0], box_a[1], box_a[2], box_a[3] <-> xmin, ymin, width, height
    '''
    w_intsec=np.maximum(
        0,
        (np.minimum(box_a[0]+box_a[2],box_b[0]+box_b[2])-np.maximum(box_a[0],box_b[0]))
    )
    h_intsec=np.maximum(
        0,
        (np.minimum(box_a[1]+box_a[3],box_b[1]+box_b[3])-np.maximum(box_a[1],box_b[1]))
    )
    s_intsec=w_intsec * h_intsec
    s_a = box_a[2]*box_a[3]
    s_b = box_b[2]*box_b[3]

    return float(s_intsec)/(s_a + s_b -s_intsec)
#endregion

@dataclass
class Ids:
    """A class for managing ids for the tracket objects
    """
    def __init__(self,seed):
        self.current = seed

    def get_next_id(self):
        """Gets the next id(+1) and update the current one

        Returns:
            [float/int]: The previuse id+1
        """
        self.current +=1
        return self.current

@dataclass
class InspectorVars:
    """Class for deffining the inspector variables
    Args:
        max_trck_fails (float): Positive float representing the maximum number of failures for
            a tracker.def 20
        trck_failure_pt (float): Positive float representing the failure penaltie val. def 1.0
        trck_reward_pt (float): Positive float representing the reward val. 0.5
        trck_type (Trackers): Type of tracker to use get from Trackers enume. def CSRT
        trck_id_generator (Ids): Entity to generate unique ids for trackers. def Ids(0)
        trck_resizing (bool): whether or not creat new resized tracker in each detection match.
            def True
        iou_paring_threshold (float): The max IOU score for paring tracker and detection. def 0.05
        stat_calc (stat_m.StatisticalCalculator): StatisticalCalculator object for calculating
            statistical data. def stat_m.StatisticalCalculator()
        saved_stat_calc_holder (bool): Whether or not to creat a new statistical calculator for
            the tracker, if not None then the saved_stat_calc_holder reference will be used.
    """
    # pylint: disable=too-many-instance-attributes
    max_trck_fails: float=10.0
    trck_failure_pt:float=1.0
    trck_reward_pt:float=0.5
    trck_type:Callable[[str],None]=get_tracker("CSRT")
    trck_id_generator:Ids=Ids(0)
    trck_resizing:bool=True

    iou_paring_threshold:float = 0.05

    stat_calc:stat_m.StatisticalCalculator=stat_m.StatisticalCalculator()
    saved_stat_calc_holder:stat_m.StatisticalCalculator=None

    def penaltie(self):
        """return the penaltie add with the reward for cases where
        the reward is subtracted later

        Returns:
            [float]: The penaltie add with the reward
        """
        return self.trck_failure_pt+self.trck_reward_pt

@dataclass
class DetectionVars:
    """Class for defining the detection variables
    Args:
        detection_model_path (str): The path to the detection model it will be loaded using
            tf.keras.models.load_model method.
        detection_model (tf model): If detection_model_path is not defined
        detection_proccessing (method): A method that utilize the detection model and return
            an np array of detections. Each detection in
            the format [confidence,[xmin,ymin,xmax,ymax]]
        detection_threshold (float): The minimum score for the detections to exist. Def 0.5
        non_max_sup_threshold (float): Non max suppressions threshold if<0 disabled. Def 0.3
    """
    detection_model_path:str=field(default=None)
    detection_model:Any=field(default=None)
    detection_proccessing:type(get_detection_array)=get_detection_array
    detection_threshold:float=0.5
    non_max_sup_threshold:float=0.3

    def load_model(self):
        """loading the model first from path else from model var

        Raises:
            Exception: Must supply detection_model_path or detection_model
        """
        if self.detection_model_path is not None:
            self.detection_model= load_tf_model(self.detection_model_path)
        elif self.detection_model is None:
            raise Exception("Must supply detection_model_path or detection_model")

@dataclass
class ClassificationVars:
    """Class for defining the  classification variables
    Args:
        class_model_path (str): The path to the classification model. it will be loaded using
            tf.keras.models.load_model method.
        class_model (tf model): If class_model_path is not defined
        class_proccessing (method): A method that utilize the detection model
    like classify_detection(model,det_images,size=None) and return an
    numpy NxM vector where N num of images, M num of classes and filled with scores.
    """
    class_model_path:str=field(default=None)
    class_model:Any=field(default=None)
    class_proccessing:type(classify_detection)=classify_detection

    def load_model(self):
        """loading the model first from path else from model var

        Raises:
            Exception: Must supply class_model_path or class_model
        """
        if self.class_model_path is not None:
            self.class_model=load_tf_model(self.class_model_path)
        elif self.class_model is None:
            print("Attention:When class_model_path and class_model are not supplied " \
            "the detector will act as if the model has one class and the class will always " \
            "be chosen with 100% score (not the final score).")
