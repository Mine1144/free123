from PySide6.QtCore import QRectF

from screen_thai.capture import region_bounds
from screen_thai.pipeline import extract_blocks


def test_negative_monitor_and_region_scaling():
    monitor = dict(left=-2560, top=-200, width=2560, height=1440)
    bounds = region_bounds(monitor, QRectF(.25, .5, .5, .25))
    assert bounds == dict(left=-1920, top=520, width=1280, height=360)


def test_ocr_filter_geometry_and_limit():
    points = [[-10, 10], [120, 10], [120, 40], [-10, 40]]
    raw = [[points, "Hello", .95], [points, "123", .99], [points, "ไทย", .99],
           [points, "noise", .1], [points, "world", .9]]
    blocks = extract_blocks(raw, 100, 100, .5, 1)
    assert len(blocks) == 1 and blocks[0].text == "Hello"
    assert blocks[0].box == (0, 10, 100, 30)
