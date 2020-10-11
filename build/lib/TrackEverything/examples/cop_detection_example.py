"""An example for using the TackEverything
This example uses an Object Detection model from the TensorFlow git
for detecting humans. Using a simple citizen/cop classification model
I've created using TF, it can now easly detect and track cops in a video using
a few lines of code.
"""
import os
import numpy as np
import cv2
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
import tensorflow as tf
from TrackEverything.detector import Detector
from TrackEverything.tool_box import DetectionVars,ClassificationVars,InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars
# pylint: enable=wrong-import-position

#custome loading the detection model and only providing the model to the DetectionVars
print("loading detection model...")
DET_MODEL_PATH="detection_models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model"
det_model=tf.saved_model.load(DET_MODEL_PATH)
det_model = det_model.signatures['serving_default']
print("detection model loaded!")

#custome detection model interpolation
DETECTION_THRESHOLD=0.5
def custome_get_detection_array(
        image,
        detection_model=det_model,
        detection_threshold=DETECTION_THRESHOLD,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
        model (tensorflow model obj): classification model
    """
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    detections=detection_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in detections.items()}
    output_dict['num_detections'] = num_detections
    #convert cordinates format to (xmin,ymin,width,height)
    output_dict['new_cordinates']=list(
        map(lambda x:get_box_cordinates(x,image.shape),output_dict['detection_boxes'])
    )
    #build the detection_array
    output= [
        [output_dict['detection_scores'][i],output_dict['new_cordinates'][i]]
            for i in range(num_detections) if
            output_dict['detection_scores'][i]>detection_threshold #filter low detectin score
            and output_dict['detection_classes'][i]==1 #detect only humans
            and output_dict['new_cordinates'][i][2]>0 #make sure width>0
            and output_dict['new_cordinates'][i][3]>0 #make sure height>0
    ]
    #print(output)
    return output

def get_box_cordinates(box,img_shape):
    """#convert model cordinates format to (xmin,ymin,width,height)

    Args:
        box ((xmin,xmax,ymin,ymax)): the cordinates are relative [0,1]
        img_shape ((height,width,channels)): the frame size

    Returns:
        (xmin,ymin,width,height): (xmin,ymin,width,height): converted cordinates
    """
    height,width, = img_shape[:2]
    xmin=max(int(box[1]*width),0)
    ymin=max(0,int(box[0]*height))
    xmax=min(int(box[3]*width),width-1)
    ymax=min(int(box[2]*height),height-1)
    return (
        xmin,#xmin
        ymin,#ymin
        xmax-xmin,#box width
        ymax-ymin#box height
    )

#providing only the classification model path for ClassificationVars since the default loding method
#tf.keras.models.load_model(path) will work
CLASS_MODEL_PATH="classification_models/cops_" \
"0.92770_L_0.35915_opt_RMSprop_loss_b_crossentropy_lr_0.0005_baches_20_shape_[165, 90]_loss.hdf5"
#custome classification model interpolation
def custome_classify_detection(model,det_images,size=(90,165)):
    """Classify a batch of images

    Args:
        model (tensorflow model): classification model
        det_images (np.array): batch of images in numpy array to classify
        size (tuple, optional): size to resize to, 1-D int32 Tensor of 2 elements:
            new_height, new_width (if None then no resizing).
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
            list(
                map(
                    lambda img: cv2.resize(img, size, interpolation = cv2.INTER_LINEAR),
                    det_images
                )
            )
        )
    # for i,imgs in enumerate(det_images):
    #     cv2.imshow(f"crop{i}",imgs)
    predictions=model.predict(det_images/255.)
    #print(f"dections:{len(det_images)},pred:{predictions}")
    #if class is binary make sure size is 2
    if len(predictions)>0:
        reshaped_pred=np.ones((len(predictions),2))
        #size of classification list is 1 so turn it to 2
        for ind,pred in enumerate(predictions):
            reshaped_pred[ind,:]=pred,1-pred
        #print(reshaped_pred)
        predictions=reshaped_pred
    return predictions


#set the detector
detector_1=Detector(
    det_vars=DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    ),
    class_vars=ClassificationVars(
        class_model_path=CLASS_MODEL_PATH,
        class_proccessing=custome_classify_detection
    ),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.EMA)
    ),
    visualization_vars=VisualizationVars(
        labels=["Citizen","Cop"],
        show_trackers=True,
        uncertainty_threshold=0.5
    )
)

#Test it on a video
VIDEO_PATH="screens/024.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps=(cap.get(cv2.CAP_PROP_FPS))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"h:{h} w:{w} fps:{fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

TRIM=True#whether or not to resize frame to max 480px width
if TRIM and w>480:
    dst_size=(480,int(480*h/w))

FRAME_NUMBER = -1
while cap.isOpened():
    FRAME_NUMBER += 1
    ret, frame = cap.read()
    if not ret:
        break
    new_frm=frame
    if TRIM:
        #resize frame
        new_frm=cv2.resize(new_frm,dst_size,fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
    #fix channel order since openCV flips them
    new_frm=cv2.cvtColor(new_frm, cv2.COLOR_BGR2RGB)

    #update the detector using the current frame
    detector_1.update(new_frm)
    #add the bounding boxes to the frame
    detector_1.draw_visualization(new_frm)

    #flip the channel order back
    new_frm=cv2.cvtColor(new_frm, cv2.COLOR_RGB2BGR)
    #show frame
    cv2.imshow('frame',new_frm)
    #get a small summary of the number of object of each class
    summ=detector_1.get_current_class_summary()
    print(f"frame:{FRAME_NUMBER}, summary:{summ}")
    #quite using the q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
