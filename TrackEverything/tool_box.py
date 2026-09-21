"""Model handling, bounding-box geometry and configuration containers.

This module holds the configuration dataclasses that drive a
:class:`~TrackEverything.detector.Detector` (:class:`DetectionVars`,
:class:`ClassificationVars`, :class:`InspectorVars`) together with the
framework-agnostic helpers they rely on.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np

from . import statistical_methods as stat_m


#region methods
def load_tf_model(path: str) -> Any:
    """Load a TensorFlow/Keras model from disk.

    TensorFlow is imported lazily so that it never becomes a hard requirement
    of this package.

    Args:
        path (str): path to the model.

    Returns:
        The loaded ``tf.keras`` model object.
    """
    #import inside method to remove the requierment of tf
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    print(f"Loading model from {path}...")
    model=tf.keras.models.load_model(path)
    print("Model loaded!")
    return model

def get_detection_array(detection_var: "DetectionVars", image: np.ndarray) -> list[list]:
    """Run the detection model and return an array of detections.

    Each detection is in the format ``[confidence,(xmin,ymin,width,height)]``.
    This default implementation assumes the detection model returns predictions
    as a list of lists in the format ``[conf xmin ymin xmax ymax]``.

    Args:
        detection_var (DetectionVars): Class for defining the detection variables.
        image (np.ndarray): current image.

    Returns:
        List[list]: detections as ``[confidence,(xmin,ymin,width,height)]``.
    """
    height,width, = image.shape[:2]
    detections=detection_var.detection_model.predict(image[np.newaxis, ...])
    #x is [conf, xmin, ymin, xmax, ymax]; the box width is xmax-xmin (indices 3-1)
    #and the height is ymax-ymin (indices 4-2)
    return [
            [x[0],(max(x[1],0),max(x[2],0),min(x[3]-x[1],width),min(x[4]-x[2],height))]
            for x in detections if
            x[0]>detection_var.detection_threshold and x[3]>x[1] and x[4]>x[2]
        ]

def get_classified_detection_array(
        model: Any,
        image: np.ndarray,
        detection_array: list[list],
        classiffing_method: Callable[..., np.ndarray],
    ) -> dict[str, Any]:
    """Attach classification scores to each detection.

    Returns a dictionary in the format::

        {
        'det':detection array,
        'class_res':numpy NxM vector where N num of images, M num of classes
        and filled with scores
        }

    If there is no classification model, every detection is reported as class 0
    with a score of 1, shaped ``(N,1)`` so that downstream ``argmax`` and
    indexing behave consistently with the multi-class case.

    Args:
        model (Any): classification model, or ``None``.
        image (np.ndarray): current image.
        detection_array (List[list]): detections as ``[confidence,(xmin,ymin,width,height)]``.
        classiffing_method (Callable): method used to classify the cropped detections.

    Returns:
        Dict[str, Any]: ``{'det': detection array, 'class_res': NxM score matrix}``.
    """
    dic: dict[str, Any]={'det':detection_array}
    if model is None:
        #shape (N,1): one class, score 1. A 1-D array here would make each
        #detection class_score a scalar float and break argmax/indexing.
        dic['class_res']=np.ones((len(detection_array),1))
    else:
        det_images=(crop_np_image(x[1],image) for x in detection_array)
        dic['class_res']=classiffing_method(model,list(det_images))
    return dic

def crop_np_image(coordinates: Sequence[int], img: np.ndarray) -> np.ndarray:
    """Copy a crop box portion from an image.

    Args:
        coordinates ((xmin,ymin,width,height)): coordinates for box.
        img (np.ndarray): image.

    Returns:
        np.ndarray: the cropped image.
    """
    xmin,ymin,width,height=coordinates
    return img[ymin:ymin+height,xmin:xmin+width,:]

def classify_detection(
        model: Any,
        det_images: list[np.ndarray],
        size: Optional[tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
    """Classify a batch of images.

    Args:
        model (Any): classification model.
        det_images (list[np.ndarray]): list of images in numpy array format to classify.
        size (tuple, optional): size to resize to, 1-D int32 Tensor of 2 elements:
            new_height, new_width (if None then no resizing). Defaults is None.
            (In a custom function you can use ``model.inputs[0].shape.as_list()``
            and set size to default.)
        interpolation (int, optional): OpenCV interpolation flag used when resizing.

    Returns:
        np.ndarray: NxM vector where N num of images, M num of classes and filled with scores.

        For example two images (car,plane) with three possible classes (car,plane,lion)
        that are identified correctly with 90% in the correct category and the rest
        divided equally will return ``[[0.9,0.05,0.05],[0.05,0.9,0.05]]``.
    """
    #resize bounding box capture to fit classification model
    if size is not None:
        det_images=np.asarray(
            [
                cv2.resize(img, size, interpolation = interpolation) for img in det_images
            ]
        )
    predictions=model.predict(det_images)#make sure image at correct format like /255.0
    #if class is binary make sure size is 2
    if len(predictions)>0 and len(predictions[0])<2:
        reshaped_pred=np.ones((len(predictions),2))
        #size of classification list is 1 so turn it to 2
        for ind,pred in enumerate(predictions):
            reshaped_pred[ind,:]=1-pred,pred
        predictions=reshaped_pred
    return predictions

#: Tracker name -> OpenCV factory attribute. The four legacy entries moved from
#: the ``cv2`` namespace to ``cv2.legacy`` in OpenCV 4.5.1.
_TRACKER_FACTORIES: dict[str, str] = {
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
    "mil": "TrackerMIL_create",
    "boosting": "TrackerBoosting_create",
    "tld": "TrackerTLD_create",
    "medianflow": "TrackerMedianFlow_create",
    "mosse": "TrackerMOSSE_create",
    "goturn": "TrackerGOTURN_create",
}

def available_trackers() -> list[str]:
    """Return the tracker names usable with this OpenCV build, sorted.

    Returns:
        List[str]: names accepted by :func:`get_tracker` that actually resolve.
    """
    found=[]
    for name,attr in _TRACKER_FACTORIES.items():
        for namespace in (cv2, getattr(cv2, "legacy", None)):
            if namespace is not None and getattr(namespace, attr, None) is not None:
                found.append(name)
                break
    return sorted(found)

def get_tracker(trck_type: str) -> Callable[[], Any]:
    """Resolve an OpenCV tracker factory by name.

    Looks in ``cv2`` first and then ``cv2.legacy``, where BOOSTING, TLD,
    MedianFlow and MOSSE moved in OpenCV 4.5.1. Matching is case-insensitive.

    Args:
        trck_type (str): tracker name, e.g. ``"CSRT"``, ``"kcf"``, ``"mosse"``.

    Returns:
        Callable[[], Any]: a zero-argument factory producing a tracker object.

    Raises:
        ValueError: the name is not a known tracker.
        RuntimeError: the name is known but unavailable in this OpenCV build.
    """
    key=str(trck_type).strip().lower()
    if key not in _TRACKER_FACTORIES:
        raise ValueError(
            f"Unknown tracker type {trck_type!r}. "
            f"Available names: {', '.join(sorted(_TRACKER_FACTORIES))}."
        )
    attr=_TRACKER_FACTORIES[key]
    for namespace in (cv2, getattr(cv2, "legacy", None)):
        factory=getattr(namespace, attr, None) if namespace is not None else None
        if factory is not None:
            return factory
    raise RuntimeError(
        f"Tracker {trck_type!r} ({attr}) is not available in this OpenCV build "
        f"(cv2 {cv2.__version__}). BOOSTING, TLD, MedianFlow and MOSSE moved to "
        f"cv2.legacy in OpenCV 4.5.1 and ship only in opencv-contrib-python. "
        f"Install or upgrade with 'pip install --upgrade opencv-contrib-python', or "
        f"pick one of: {', '.join(available_trackers())}."
    )

def resolve_tracker_factory(trck_type: Union[str, Callable[[], Any]]) -> Callable[[], Any]:
    """Accept either a tracker name or an already-resolved factory.

    Args:
        trck_type (Union[str, Callable]): tracker name or zero-argument factory.

    Returns:
        Callable[[], Any]: a zero-argument factory producing a tracker object.
    """
    return trck_type if callable(trck_type) else get_tracker(trck_type)

def non_max_suppressions(detections: list, threshold_iou: float = 0.3) -> list:
    """Remove overlapping detections using the non-max suppression IOU method.

    .. note:: ``detections`` is modified in place and also returned.

    Args:
        detections (List[DetectedObj]): List of DetectedObj.
        threshold_iou (float): IOU above which the lower-scored detection is dropped.

    Returns:
        List[DetectedObj]: clean List of DetectedObj.

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

def cv2_bbox_to_tf_bbox(cv2_bbox: Sequence[float], width: int, height: int) -> np.ndarray:
    """Convert an OpenCV bounding box to the TensorFlow format.

    TensorFlow bounding box format is ``[y_min, x_min, y_max, x_max]`` with
    coordinates as floats in ``[0.0, 1.0]`` relative to the width and the height
    of the underlying image.

    Args:
        cv2_bbox (List[xmin,ymin,box_width,box_height]): Bounding box.
        width (int): width of the image.
        height (int): height of the image.

    Returns:
        np.ndarray: ``[y_min, x_min, y_max, x_max]`` tensorflow bounding box coordinates.
    """
    return np.array([
    cv2_bbox[1]/height,
    cv2_bbox[0]/width,
    min((cv2_bbox[1]+cv2_bbox[3])/height,1),
    min((cv2_bbox[0]+cv2_bbox[2])/width,1),
    ])

def cv2_bbox_reshape(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert ``[xmin,ymin,box_width,box_height]`` to ``[x_min,y_min,x_max,y_max]``.

    Args:
        box (List[xmin,ymin,box_width,box_height]): Bounding box.

    Returns:
        Tuple[float, float, float, float]: ``[x_min,y_min,x_max,y_max]`` bounding box.
    """
    return (box[0],box[1],box[0]+box[2],box[1]+box[3])

def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate the ratio between the intersection and the union of two boxes.

    ``box[0], box[1], box[2], box[3]`` correspond to ``xmin, ymin, width, height``.

    Args:
        box_a (Sequence[float]): first bounding box.
        box_b (Sequence[float]): second bounding box.

    Returns:
        float: the IOU score, or ``0.0`` when both boxes have zero area.
    """
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

    union=s_a + s_b -s_intsec
    if union<=0:
        #two degenerate (zero-area) boxes have no meaningful overlap
        return 0.0
    return float(s_intsec)/union
#endregion

class Ids:
    """A class for managing ids for the tracker objects.

    .. note:: This is deliberately a plain class, not a dataclass: it defines
       its own ``__init__`` and has no fields, so ``@dataclass`` would only add
       an ``__eq__``/``__hash__ = None`` pair, which Python 3.11+ rejects when
       an instance is used as a dataclass field default.
    """
    def __init__(self, seed: float = 0):
        self.current = seed

    def get_next_id(self) -> float:
        """Get the next id (+1) and update the current one.

        Returns:
            float: The previous id + 1.
        """
        self.current +=1
        return self.current

    def __repr__(self) -> str:
        return f"Ids(current={self.current})"

@dataclass
class InspectorVars:
    """Class for defining the inspector variables.

    Args:
        max_trck_fails (float): Positive float representing the maximum number of failures for
            a tracker. def 10.0
        trck_failure_pt (float): Positive float representing the failure penalty value. def 1.0
        trck_reward_pt (float): Positive float representing the reward value. def 0.5
        trck_type (Union[str, Callable]): Tracker name (case-insensitive) such as ``"CSRT"``,
            ``"kcf"`` or ``"mosse"``, or a zero-argument factory returning a tracker.
            Resolved and validated when this object is constructed. def "CSRT"
        trck_id_generator (Ids): Entity to generate unique ids for trackers. Each
            InspectorVars gets its own generator. def Ids(0)
        trck_resizing (bool): whether or not to create a new resized tracker on each
            detection match. def True
        iou_paring_threshold (float): The max IOU score for pairing tracker and detection.
            def 0.05
        stat_calc (stat_m.StatisticalCalculator): StatisticalCalculator object for calculating
            statistical data. Each InspectorVars gets its own calculator.
        saved_stat_calc_holder (Optional[stat_m.StatisticalCalculator]): Whether or not to
            create a new statistical calculator for the tracker; if not None then the
            saved_stat_calc_holder reference will be used.
    """
    # pylint: disable=too-many-instance-attributes
    max_trck_fails: float=10.0
    trck_failure_pt:float=1.0
    trck_reward_pt:float=0.5
    trck_type:Union[str, Callable[[], Any]]="CSRT"
    trck_id_generator:Ids=field(default_factory=lambda: Ids(0))
    trck_resizing:bool=True

    iou_paring_threshold:float = 0.05

    stat_calc:stat_m.StatisticalCalculator=field(
        default_factory=stat_m.StatisticalCalculator)
    saved_stat_calc_holder:Optional[stat_m.StatisticalCalculator]=None

    def __post_init__(self) -> None:
        #resolve eagerly so a bad tracker name fails here, at configuration time,
        #rather than deep inside TrackerObj construction on some later frame
        self._tracker_source = self.trck_type
        self._tracker_factory = resolve_tracker_factory(self.trck_type)

    def get_tracker_factory(self) -> Callable[[], Any]:
        """Return the cached zero-argument tracker factory.

        Re-resolves automatically if ``trck_type`` was reassigned after construction.

        Returns:
            Callable[[], Any]: factory producing a new tracker object.
        """
        if getattr(self, "_tracker_factory", None) is None or \
                getattr(self, "_tracker_source", None) is not self.trck_type:
            self._tracker_source = self.trck_type
            self._tracker_factory = resolve_tracker_factory(self.trck_type)
        return self._tracker_factory

    def penaltie(self) -> float:
        """Return the penalty added with the reward, for cases where
        the reward is subtracted later.

        Returns:
            float: The penalty added with the reward.
        """
        return self.trck_failure_pt+self.trck_reward_pt

@dataclass
class DetectionVars:
    """Class for defining the detection variables.

    Args:
        detection_model_path (str): The path to the detection model; it will be loaded using
            the ``tf.keras.models.load_model`` method.
        detection_model (Any): If detection_model_path is not defined.
        detection_proccessing (Callable): A method that utilizes the detection model and returns
            an array of detections, each in the format
            ``[confidence,(xmin,ymin,width,height)]``.
        detection_threshold (float): The minimum score for the detections to exist. Def 0.5
        non_max_sup_threshold (float): Non max suppression threshold; if <0 disabled. Def 0.3
    """
    detection_model_path:Optional[str]=field(default=None)
    detection_model:Any=field(default=None)
    detection_proccessing:Callable[..., list[list]]=get_detection_array
    detection_threshold:float=0.5
    non_max_sup_threshold:float=0.3

    def load_model(self) -> None:
        """Load the model, first from path and otherwise from the model variable.

        Raises:
            ValueError: Must supply detection_model_path or detection_model.
        """
        if self.detection_model_path is not None:
            self.detection_model= load_tf_model(self.detection_model_path)
        elif self.detection_model is None:
            raise ValueError("Must supply detection_model_path or detection_model")

@dataclass
class ClassificationVars:
    """Class for defining the classification variables.

    Args:
        class_model_path (str): The path to the classification model; it will be loaded using
            the ``tf.keras.models.load_model`` method.
        class_model (Any): If class_model_path is not defined.
        class_proccessing (Callable): A method that utilizes the classification model,
            like ``classify_detection(model,det_images,size=None)``, and returns an
            NxM array where N is the number of images and M the number of classes,
            filled with scores.
    """
    class_model_path:Optional[str]=field(default=None)
    class_model:Any=field(default=None)
    class_proccessing:Callable[..., np.ndarray]=classify_detection

    def load_model(self) -> None:
        """Load the model, first from path and otherwise from the model variable."""
        if self.class_model_path is not None:
            self.class_model=load_tf_model(self.class_model_path)
        elif self.class_model is None:
            print("Attention:When class_model_path and class_model are not supplied " \
            "the detector will act as if the model has one class and the class will always " \
            "be chosen with 100% score (not the final score).")
