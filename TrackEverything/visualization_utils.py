"""Drawing of tracker and detection overlays onto frames."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

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
    """Class for defining the visualization on the frame.

    Args:
        labels (list[str]): list of labels for the classes, in the order of the class vector.
        colors (list[str]): list of color strings for the classes, in the order of the class
            vector. def=STANDARD_COLORS(126)
        show_ids (bool): whether to show the detection id. def=True
        show_trackers (bool): whether to show the trackers bounding box (if trck_resizing in
            the InspectorVars is true, the bounding box of the tracker on a detected object
            will be hidden by the detection bounding box). def=False
        uncertainty_threshold (float): a threshold for the final score (including
            classification and statistics) where, if not met, the detection will be marked
            with the uncertainty_label tag in the uncertainty_color color. def=0
        uncertainty_color (str): the color for an uncertain final score. def="Orange"
        uncertainty_label (str): the label for an uncertain final score. def="Unknown"
    """
    labels:Optional[list[str]]=None
    #copy, so that a caller mutating one instance's palette cannot affect
    #every other instance or the module-level list itself
    colors:list[str]=field(default_factory=lambda: list(STANDARD_COLORS))
    show_ids:bool=True
    show_trackers:bool=False
    uncertainty_threshold:float=0
    uncertainty_color:str="Orange"
    uncertainty_label:str="Unknown"

def _measure_text(draw: ImageDraw.ImageDraw, font, text: str) -> tuple[float, float]:
    """Return the ``(width, height)`` of ``text`` as rendered with ``font``.

    ``ImageFont.getsize`` was removed in Pillow 10. ``ImageDraw.multiline_textbbox``
    has existed since Pillow 8.0 and, unlike ``font.getbbox``, is available for both
    ``FreeTypeFont`` and the bitmap font returned by ``ImageFont.load_default()`` on
    every supported version. It also measures every line of a multi-line label,
    which the labels here always are.

    Args:
        draw (ImageDraw.ImageDraw): the draw object the text will be rendered with.
        font: the font the text will be rendered with.
        text (str): the text to measure, possibly containing newlines.

    Returns:
        Tuple[float, float]: the width and height of the rendered text.
    """
    if hasattr(draw, "multiline_textbbox"):# Pillow >= 8.0
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return font.getsize(text)# Pillow < 8.0

def draw_boxes(
        image: np.ndarray,
        detections: list,
        trackers: list,
        v_vars: Optional[VisualizationVars] = None,
        org_img_size: Optional[tuple[int, int]] = None,
    ) -> None:
    """Draw boxes and labels on the image, replacing the image contents in place.

    Args:
        image (np.ndarray): The image to draw on.
        detections (List[DetectedObj]): A list of detected objects to draw boxes around.
        trackers (List[TrackerObj]): A list of tracker objects to draw boxes around
            (only if ``v_vars.show_trackers``).
        v_vars (VisualizationVars): Extra parameters for the drawing style.
        org_img_size (width, height): The original image size the bounding boxes were
            created against.
    """
    if v_vars is None:
        v_vars=VisualizationVars()
    #calculate the factor for bounding box with different sized images
    if org_img_size is None:
        factors=(1,1)
    else:
        factors=(image.shape[1]/org_img_size[0],image.shape[0]/org_img_size[1])
    #creates the ImageDraw object
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
            color=v_vars.colors[class_num%len(v_vars.colors)]
            #if labels are provided write them
            if v_vars.labels is not None:
                text.append(f"{v_vars.labels[class_num]}\n")
        if v_vars.show_ids:
            text.append(f"Id:{det.id_num!s}\n")
        #add the final score to the tag
        text.append(f"{100*det.class_score[class_num]:.0f}%")
        draw_box_and_text(draw,det.bounding_box,color=color,text=''.join(text),factors=factors)
    #replace the old image with new
    np.copyto(image, np.array(image_pil))

def draw_box_and_text(
        draw: ImageDraw.ImageDraw,
        bounding_box: Sequence[float],
        color: str = "Red",
        thickness: int = 2,
        text: str = "",
        factors: tuple[float, float] = (1, 1),
    ) -> None:
    """Draw a box with a tag using the draw object.

    Args:
        draw (ImageDraw.ImageDraw): the ImageDraw object to draw in.
        bounding_box ((xmin,ymin,width,height)): the box coordinates.
        color (str, optional): color for box and text. Defaults to "Red".
        thickness (int, optional): thickness of box lines. Defaults to 2.
        text (str, optional): the text to draw with the box. Defaults to "".
        factors (width_ratio, height_ratio): The size ratio for bounding boxes.
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
    except OSError:
        font = ImageFont.load_default()
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    _text_width, text_height = _measure_text(draw, font, text)
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * text_height

    text_bottom = top if top > total_display_str_height else bottom + total_display_str_height

    margin = np.ceil(0.05 * text_height)
    #draw text
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        text,
        fill=color,
        font=font
    )
