"""A module for managing the visualization of the trackers and detections
"""
from dataclasses import dataclass
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Orange', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Plum', 'PowderBlue',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'Pink', 'Purple'
]

@dataclass
class VisualizationVars:
    """Class for deffining the visualization on the frame
    Args:
        labels (list[str]): list of lables for the classes by order of the class vector.
        colors (list[str]): list of colors strings for the classes by order of the class vector.
            def=STANDARD_COLORS(126)
        show_ids (bool): whether to show the detection id. def=True
        show_trackers (bool): whether to show the trackers bounding box (if trck_resizing in
            the InspectorVars is true the bounding box of the tracker on detected object will
            be hidden by the detection bounding box) def=False.
        uncertainty_threshold (float): a threshold for the final score (including
            classification and statistics) where if not met will be marked with
            uncertainty_label tag in uncertainty_color color. def=0
        uncertainty_color (str): the color for uncertain final score. def="Orange"
        uncertainty_label (str): the label for uncertain final score. def="Unknown"
    """
    labels:'list[str]'=None
    colors:'list[str]'=None
    show_ids:bool=True
    show_trackers:bool=False
    uncertainty_threshold:float=0
    uncertainty_color:str="Orange"
    uncertainty_label:str="Unknown"

def draw_boxes(
        image,
        detections,
        trackers,
        v_vars:VisualizationVars=VisualizationVars(),
        org_img_size=None
    ):
    """A method for drawing boxes and labels on the image (it will replace the
    image with the new one)

    Args:
        image (np.array): The image to draw on
        detections (List[DetectedObj]): A list of detected objects to draw boxes around
        trackers (List[TrackerObj]): A list of trackers objects to draw boxes around.
            (only if v_vars.show_trackers).
        v_vars (VisualizationVars): Extra parameters for the drwing style.
        org_img_size (width, height):The original image size for bounding boxes
    """
    #set the dif value for colors array
    v_vars.colors=v_vars.colors if v_vars.colors is not None else STANDARD_COLORS
    #calculate the factor for bounding box with different sized images
    if org_img_size is None:
        factors=(1,1)
    else:
        factors=(image.shape[1]/org_img_size[0],image.shape[0]/org_img_size[1])
    #creats the ImageDraw object
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    #draw trackers
    if v_vars.show_trackers:
        for trck in trackers:
            draw_box_and_text(draw,trck.bounding_box,color=v_vars.colors[-1],factors=factors)
    #draw detections
    for det in detections:
        text =[]
        #get the detection class and score
        class_num,score=det.get_current_class()
        #whether to mark detection as uncertain
        if score<v_vars.uncertainty_threshold:
            color=v_vars.uncertainty_color
            text.append(f"{v_vars.uncertainty_label}\n")#add uncertainty label
        else:
            color=v_vars.colors[class_num]
            #if labels are provided write them
            if v_vars.labels is not None:
                text.append(f"{v_vars.labels[class_num]}\n")
        if v_vars.show_ids:
            text.append(f"Id:{str(det.id_num)}\n")
        #add the final score to the tag
        text.append(f"{100*det.class_score[class_num]:.0f}%")
        draw_box_and_text(draw,det.bounding_box,color=color,text=''.join(text),factors=factors)
    #replace the old image with new
    np.copyto(image, np.array(image_pil))

def draw_box_and_text(draw,bounding_box,color="Red",thickness=2,text="",factors=(1,1)):
    """This method draws a box with a tag using the draw object

    Args:
        draw (ImageDraw): the ImageDraw object for drawing in.
        bounding_box ((xmin,ymin,width,height)): the box coordinates
        color (str, optional): color for box and text. Defaults to "Red".
        thickness (int, optional): thickness of box lines. Defaults to 2.
        text (str, optional): the text to draw with the box. Defaults to "".
        factors (width_ratio, height_ratio):The ratio size for bounding boxes
    """
    left=bounding_box[0]*factors[0]
    top=bounding_box[1]*factors[1]
    right=left+bounding_box[2]*factors[0]
    bottom=top+bounding_box[3]*factors[1]
    #draw box
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top),(left, top)],
        width=thickness,
        fill=color
        )
    try:
        font = ImageFont.truetype('arial.ttf', 22)
        #font = ImageFont.truetype('arial.ttf', int(rs[0]*(72/288)))
    except IOError:
        font = ImageFont.load_default()
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_height = font.getsize(text)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * display_str_height

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    text_height = font.getsize(text)[1]
    margin = np.ceil(0.05 * text_height)
    #draw text
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        text,
        fill=color,
        font=font
    )
