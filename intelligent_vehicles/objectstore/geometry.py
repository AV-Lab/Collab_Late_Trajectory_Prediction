"""Rotated BEV overlap for boxes [x, y, z, length, width, height, yaw].

Yaw is in radians. Z and height do not affect the two-dimensional overlap.
"""

import math


def _rectangle(box, origin):
    """Return counterclockwise BEV corners and area, or None for invalid boxes."""
    try:
        x, y, length, width, yaw = (float(box[index]) for index in (0, 1, 3, 4, 6))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, length, width, yaw)):
        return None
    if length <= 0 or width <= 0:
        return None
    cosine, sine = math.cos(yaw), math.sin(yaw)
    corners = []
    for dx, dy in ((-length / 2, -width / 2), (length / 2, -width / 2),
                   (length / 2, width / 2), (-length / 2, width / 2)):
        corners.append((x - origin[0] + cosine * dx - sine * dy,
                        y - origin[1] + sine * dx + cosine * dy))
    area = length * width
    if not math.isfinite(area) or not all(math.isfinite(v) for corner in corners for v in corner):
        return None
    return corners, area


def _side(start, end, point):
    return ((end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0]))


def point_in_rotated_bev_box(point, box):
    """Whether a finite XY point lies inside or on a valid rotated box."""
    try:
        origin = (float(point[0]), float(point[1]))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in origin):
        return False
    rectangle = _rectangle(box, origin)
    if rectangle is None:
        return False
    corners, _ = rectangle
    return all(_side(start, end, (0., 0.)) >= -1e-10
               for start, end in zip(corners, corners[1:] + corners[:1]))


def _intersection(subject, clip):
    """Sutherland-Hodgman intersection of counterclockwise convex polygons."""
    output = list(subject)
    for start, end in zip(clip, clip[1:] + clip[:1]):
        incoming, output = output, []
        if not incoming:
            break
        previous = incoming[-1]
        previous_side = _side(start, end, previous)
        for current in incoming:
            current_side = _side(start, end, current)
            if (current_side >= 0) != (previous_side >= 0):
                weight = previous_side / (previous_side - current_side)
                output.append((previous[0] + weight * (current[0] - previous[0]),
                               previous[1] + weight * (current[1] - previous[1])))
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
    return output


def rotated_bev_iou(box_a, box_b):
    """Return oriented rectangle IoU; invalid/missing BEV geometry yields zero.

    Z and height are deliberately irrelevant. Shift both boxes to box A's
    center before clipping to avoid cancellation at large world coordinates.
    """
    try:
        origin = (float(box_a[0]), float(box_a[1]))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return 0.0
    rectangle_a, rectangle_b = _rectangle(box_a, origin), _rectangle(box_b, origin)
    if rectangle_a is None or rectangle_b is None:
        return 0.0
    polygon_a, area_a = rectangle_a
    polygon_b, area_b = rectangle_b
    for axis in (0, 1):
        if (max(point[axis] for point in polygon_a) <= min(point[axis] for point in polygon_b)
                or max(point[axis] for point in polygon_b) <= min(point[axis] for point in polygon_a)):
            return 0.0
    intersection = _intersection(polygon_a, polygon_b)
    if len(intersection) < 3:
        return 0.0
    overlap = abs(math.fsum(a[0] * b[1] - a[1] * b[0]
                            for a, b in zip(intersection, intersection[1:] + intersection[:1]))) / 2
    overlap = min(overlap, area_a, area_b)
    union = area_a + area_b - overlap
    return max(0.0, min(1.0, overlap / union)) if union > 0 and math.isfinite(union) else 0.0
