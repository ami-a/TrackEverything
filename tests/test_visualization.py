"""Drawing overlays, including the Pillow 10 font-measurement break.

``ImageFont.getsize`` was removed in Pillow 10, which made every call to
``draw_visualization`` raise AttributeError.
"""
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import pytest

from TrackEverything import DetectedObj, VisualizationVars, draw_boxes
from TrackEverything.visualization_utils import _measure_text, draw_box_and_text


def det(bbox=(40, 40, 100, 160), classes=(0.2, 0.8), id_num=1):
    obj = DetectedObj(0.9, np.array(classes, dtype=float), bbox)
    obj.id_num = id_num
    return obj


class TestMeasureText:
    def test_returns_positive_dimensions(self):
        image = ImageDraw.Draw(_blank())
        width, height = _measure_text(image, ImageFont.load_default(), "hello")
        assert width > 0
        assert height > 0

    def test_multiline_is_taller_than_single_line(self):
        """Measurement used to be single-line while rendering was multi-line."""
        draw = ImageDraw.Draw(_blank())
        font = ImageFont.load_default()
        _, single = _measure_text(draw, font, "mask")
        _, triple = _measure_text(draw, font, "mask\nId:3\n90%")
        assert triple > single

    def test_legacy_pillow_branch_still_works(self, monkeypatch):
        """Force the Pillow<8 path by hiding multiline_textbbox."""
        monkeypatch.delattr(ImageDraw.ImageDraw, "multiline_textbbox", raising=False)
        draw = ImageDraw.Draw(_blank())
        font = ImageFont.load_default()
        if not hasattr(font, "getsize"):
            pytest.skip("installed Pillow has no getsize to fall back to")
        width, height = _measure_text(draw, font, "hello")
        assert width > 0 and height > 0


def _blank():
    from PIL import Image

    return Image.new("RGB", (320, 240))


class TestDrawBoxes:
    def test_draws_onto_a_blank_frame(self, frame):
        blank = np.zeros_like(frame)
        draw_boxes(blank, [det()], [], VisualizationVars())
        assert blank.any(), "nothing was drawn"

    def test_works_with_default_visualization_vars(self, frame):
        blank = np.zeros_like(frame)
        draw_boxes(blank, [det()], [])
        assert blank.any()

    def test_labels_are_optional(self, frame):
        blank = np.zeros_like(frame)
        draw_boxes(blank, [det()], [], VisualizationVars(labels=None))
        assert blank.any()

    def test_named_labels_are_accepted(self, frame):
        blank = np.zeros_like(frame)
        draw_boxes(blank, [det()], [], VisualizationVars(labels=["no_mask", "mask"]))
        assert blank.any()

    def test_uncertainty_threshold_changes_output(self, frame):
        certain = np.zeros_like(frame)
        uncertain = np.zeros_like(frame)
        draw_boxes(certain, [det()], [], VisualizationVars(uncertainty_threshold=0.0))
        draw_boxes(uncertain, [det()], [], VisualizationVars(uncertainty_threshold=1.0))
        assert not np.array_equal(certain, uncertain)

    def test_org_img_size_scales_the_boxes(self, frame):
        unscaled = np.zeros_like(frame)
        scaled = np.zeros_like(frame)
        draw_boxes(unscaled, [det()], [], VisualizationVars())
        draw_boxes(scaled, [det()], [], VisualizationVars(), org_img_size=(720, 480))
        assert not np.array_equal(unscaled, scaled)

    def test_many_classes_do_not_overflow_the_palette(self, frame):
        blank = np.zeros_like(frame)
        classes = [0.0] * 400
        classes[399] = 1.0
        draw_boxes(blank, [det(classes=classes)], [], VisualizationVars())
        assert blank.any()

    def test_frame_shape_is_preserved(self, frame):
        blank = np.zeros_like(frame)
        draw_boxes(blank, [det()], [], VisualizationVars())
        assert blank.shape == frame.shape


class TestDrawBoxAndText:
    def test_draws_without_text(self):
        image = _blank()
        draw = ImageDraw.Draw(image)
        draw_box_and_text(draw, (10, 10, 50, 50))
        assert np.array(image).any()

    def test_box_near_the_top_edge_is_handled(self):
        """Labels stack below the box when there is no room above."""
        image = _blank()
        draw = ImageDraw.Draw(image)
        draw_box_and_text(draw, (10, 0, 50, 50), text="a\nb\nc")
        assert np.array(image).any()
