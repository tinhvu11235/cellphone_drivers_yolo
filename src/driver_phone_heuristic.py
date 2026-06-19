"""Rule-based spatial reasoning for driver cellphone-use alerts.

This module does not train or run a behavior model. It consumes object
detections from a detector such as YOLO and infers the final behavior from the
spatial relationship between a person/driver, a phone, and a steering wheel.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot
from typing import Any


# Tune these first when moving between datasets/camera viewpoints.
DEFAULT_CONFIG: dict[str, float] = {
    "person_conf": 0.40,
    "phone_conf": 0.30,
    "wheel_conf": 0.30,
    "driver_expand_ratio": 0.12,
    "zone_expand_ratio": 0.08,
    "phone_driver_iou": 0.01,
    "phone_driver_overlap": 0.10,
    "phone_driver_distance_ratio": 0.10,
    "side_phone_min_conf": 0.45,
    "side_phone_distance_ratio": 0.20,
    "side_phone_overlap": 0.03,
    "side_phone_min_x_ratio": -0.30,
    "side_phone_max_x_ratio": 1.35,
    "side_phone_min_y_ratio": 0.30,
    "side_phone_max_y_ratio": 0.95,
    "low_phone_min_conf": 0.60,
    "low_phone_min_overlap": 0.50,
    "low_phone_min_y_ratio": 0.85,
    "low_phone_max_y_ratio": 1.05,
    "high_phone_conf": 0.60,
    "alert_score": 0.50,
}

PERSON_CLASS_NAMES = {"person", "driver"}
PHONE_CLASS_NAMES = {"cellphone", "cell_phone", "cell phone", "mobile_phone", "mobile phone", "phone"}
WHEEL_CLASS_NAMES = {"wheel", "steering_wheel", "steering wheel"}


Box = list[float]
Point = tuple[float, float]
Detection = dict[str, Any]


def center(box: Sequence[float] | Detection) -> Point:
    """Return the center point of an xyxy box."""
    x1, y1, x2, y2 = _box_xyxy(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def area(box: Sequence[float] | Detection) -> float:
    """Return the non-negative area of an xyxy box."""
    x1, y1, x2, y2 = _box_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box1: Sequence[float] | Detection, box2: Sequence[float] | Detection) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    x1, y1, x2, y2 = _box_xyxy(box1)
    xx1, yy1, xx2, yy2 = _box_xyxy(box2)

    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = area([x1, y1, x2, y2]) + area([xx1, yy1, xx2, yy2]) - intersection
    return 0.0 if union <= 0 else intersection / union


def expand_box(
    box: Sequence[float] | Detection,
    ratio: float,
    image_width: int | float,
    image_height: int | float,
) -> Box:
    """Expand an xyxy box by a ratio of its width/height and clamp to image bounds."""
    x1, y1, x2, y2 = _box_xyxy(box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    pad_x = width * ratio
    pad_y = height * ratio
    return [
        _clamp(x1 - pad_x, 0.0, float(image_width)),
        _clamp(y1 - pad_y, 0.0, float(image_height)),
        _clamp(x2 + pad_x, 0.0, float(image_width)),
        _clamp(y2 + pad_y, 0.0, float(image_height)),
    ]


def point_inside_box(point: Point, box: Sequence[float] | Detection) -> bool:
    """Return True when a point lies inside an xyxy box."""
    x, y = point
    x1, y1, x2, y2 = _box_xyxy(box)
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_point_to_box(point: Point, box: Sequence[float] | Detection) -> float:
    """Return Euclidean distance from a point to a box, or 0 when inside."""
    x, y = point
    x1, y1, x2, y2 = _box_xyxy(box)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)


def select_driver(
    persons: list[Detection],
    wheels: list[Detection],
    image_shape: Sequence[int],
) -> tuple[Detection | None, Detection | None, bool]:
    """Select the most likely driver.

    Returns ``(driver_detection, wheel_detection, selected_with_good_wheel)``.
    If no wheel is available, the largest person is used as a lower-confidence
    fallback.
    """
    persons = [_normalize_detection(person) for person in persons]
    wheels = [_normalize_detection(wheel) for wheel in wheels]

    if not persons:
        return None, None, False

    if not wheels:
        largest_person = max(persons, key=lambda person: area(person["bbox"]))
        return largest_person, None, False

    best: tuple[float, Detection, Detection, bool] | None = None
    for person in persons:
        person_box = person["bbox"]
        person_center = center(person_box)
        person_size = max(1.0, _box_width(person_box), _box_height(person_box))

        for wheel in wheels:
            wheel_box = wheel["bbox"]
            wheel_center = center(wheel_box)
            wheel_inside = point_inside_box(wheel_center, person_box)
            overlap = iou(person_box, wheel_box)
            distance = hypot(person_center[0] - wheel_center[0], person_center[1] - wheel_center[1])
            proximity = max(0.0, 1.0 - min(distance / person_size, 1.0))

            # Priority order: wheel center inside driver, overlap, then closest center.
            score = proximity
            if overlap > 0:
                score += 1.0 + min(overlap * 4.0, 1.0)
            if wheel_inside:
                score += 3.0

            good_wheel_match = wheel_inside or overlap > 0 or distance <= person_size * 0.75
            candidate = (score, person, wheel, good_wheel_match)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        largest_person = max(persons, key=lambda person: area(person["bbox"]))
        return largest_person, None, False
    _, driver, wheel, selected_with_wheel = best
    return driver, wheel, selected_with_wheel


def phone_belongs_to_driver(
    phone: Detection,
    driver_box: Sequence[float],
    image_shape: Sequence[int],
) -> bool:
    """Return True when a phone is spatially associated with the selected driver."""
    image_height, image_width = _image_hw(image_shape)
    phone = _normalize_detection(phone)
    relation = _phone_driver_relation(
        phone,
        driver_box,
        image_width=image_width,
        image_height=image_height,
        config=DEFAULT_CONFIG,
    )
    return relation["belongs"]


def infer_driver_phone_usage(
    detections: list[Detection],
    image_shape: Sequence[int],
) -> dict[str, Any]:
    """Infer whether the driver is using/listening to a phone for one frame."""
    image_height, image_width = _image_hw(image_shape)
    normalized = [_normalize_detection(detection) for detection in detections]

    persons = _filter_by_class_and_conf(
        normalized,
        PERSON_CLASS_NAMES,
        DEFAULT_CONFIG["person_conf"],
    )
    phones = _filter_by_class_and_conf(
        normalized,
        PHONE_CLASS_NAMES,
        DEFAULT_CONFIG["phone_conf"],
    )
    wheels = _filter_by_class_and_conf(
        normalized,
        WHEEL_CLASS_NAMES,
        DEFAULT_CONFIG["wheel_conf"],
    )

    if not persons:
        return _empty_prediction("missing_person")

    driver, wheel, selected_with_wheel = select_driver(persons, wheels, image_shape)
    if driver is None:
        return _empty_prediction("missing_driver")

    if not phones:
        return {
            **_empty_prediction("missing_phone"),
            "driver_bbox": _round_box(driver["bbox"]),
            "wheel_bbox": _round_box(wheel["bbox"]) if wheel else None,
        }

    driver_box = driver["bbox"]
    best_prediction: dict[str, Any] | None = None
    for phone in phones:
        prediction = _score_phone_candidate(
            phone=phone,
            driver_box=driver_box,
            wheel=wheel,
            selected_with_wheel=selected_with_wheel,
            image_width=image_width,
            image_height=image_height,
        )
        if (
            best_prediction is None
            or prediction["risk_score"] > best_prediction["risk_score"]
            or (
                prediction["phone_belongs_to_driver"]
                and not best_prediction["phone_belongs_to_driver"]
            )
        ):
            best_prediction = prediction

    if best_prediction is None or not best_prediction["phone_belongs_to_driver"]:
        return {
            "driver_using_phone": False,
            "risk_score": 0.0,
            "reason": "phone_not_related_to_driver",
            "driver_bbox": _round_box(driver_box),
            "phone_bbox": None,
            "wheel_bbox": _round_box(wheel["bbox"]) if wheel else None,
        }

    risk_score = best_prediction["risk_score"]
    reason = best_prediction["reason"]
    alert_reasons = {"phone_near_head", "phone_near_upper_body", "phone_low_inside_driver"}
    driver_using_phone = reason in alert_reasons and risk_score >= DEFAULT_CONFIG["alert_score"]

    return {
        "driver_using_phone": bool(driver_using_phone),
        "risk_score": round(risk_score, 4),
        "reason": reason if driver_using_phone else _negative_reason(reason),
        "driver_bbox": _round_box(driver_box),
        "phone_bbox": _round_box(best_prediction["phone_bbox"]),
        "wheel_bbox": _round_box(wheel["bbox"]) if wheel else None,
    }


def smooth_predictions(
    predictions: list[dict[str, Any] | bool],
    window_size: int = 10,
    threshold: int = 5,
) -> bool:
    """Return a video-level alert from the last N frame predictions.

    The default means: alert only when at least 5 of the last 10 frames are
    positive. For a streaming pipeline, keep appending per-frame dictionaries
    returned by ``infer_driver_phone_usage`` and pass the history here.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0.")

    window = predictions[-window_size:]
    positives = sum(1 for prediction in window if _prediction_is_positive(prediction))
    return positives >= threshold


def _score_phone_candidate(
    phone: Detection,
    driver_box: Sequence[float],
    wheel: Detection | None,
    selected_with_wheel: bool,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Score one phone-driver pair and keep the strongest rule reason."""
    phone_box = phone["bbox"]
    relation = _phone_driver_relation(
        phone,
        driver_box,
        image_width=image_width,
        image_height=image_height,
        config=DEFAULT_CONFIG,
    )
    if not relation["belongs"]:
        return {
            "phone_belongs_to_driver": False,
            "risk_score": 0.0,
            "reason": "phone_not_related_to_driver",
            "phone_bbox": phone_box,
        }

    phone_center = center(phone_box)
    zones = _driver_body_zones(driver_box)
    head_zone = expand_box(zones["head_zone"], DEFAULT_CONFIG["zone_expand_ratio"], image_width, image_height)
    upper_body_zone = expand_box(
        zones["upper_body_zone"],
        DEFAULT_CONFIG["zone_expand_ratio"],
        image_width,
        image_height,
    )

    risk_score = 0.0
    reason = "phone_low_or_dashboard"
    if _phone_in_zone(phone_box, phone_center, head_zone):
        risk_score += 0.50
        reason = "phone_near_head"
    elif _phone_in_zone(phone_box, phone_center, upper_body_zone):
        risk_score += 0.30
        reason = "phone_near_upper_body"
        if relation["side_upper_close"] and not point_inside_box(phone_center, driver_box):
            risk_score += 0.20
    elif _phone_is_low_inside_driver(phone, driver_box, relation):
        risk_score += 0.30
        reason = "phone_low_inside_driver"

    if point_inside_box(phone_center, driver_box):
        risk_score += 0.20

    if selected_with_wheel:
        risk_score += 0.20
    elif wheel is None:
        risk_score -= 0.20

    if float(phone.get("confidence", 0.0)) >= DEFAULT_CONFIG["high_phone_conf"]:
        risk_score += 0.10

    return {
        "phone_belongs_to_driver": True,
        "risk_score": _clamp(risk_score, 0.0, 1.0),
        "reason": reason,
        "phone_bbox": phone_box,
    }


def _phone_driver_relation(
    phone: Detection,
    driver_box: Sequence[float],
    image_width: int,
    image_height: int,
    config: dict[str, float],
) -> dict[str, Any]:
    """Compute low-level phone-driver association signals."""
    phone_box = phone["bbox"]
    phone_center = center(phone_box)
    expanded_driver = expand_box(driver_box, config["driver_expand_ratio"], image_width, image_height)
    intersection = _intersection_area(phone_box, driver_box)
    phone_area = max(area(phone_box), 1.0)
    overlap_ratio = intersection / phone_area
    box_iou = iou(phone_box, driver_box)
    distance = distance_point_to_box(phone_center, driver_box)
    driver_size = max(1.0, _box_width(driver_box), _box_height(driver_box))
    distance_ratio = distance / driver_size
    close_enough = distance_ratio <= config["phone_driver_distance_ratio"]
    phone_x_ratio, phone_y_ratio = _point_relative_to_box(phone_center, driver_box)
    side_upper_close = _phone_is_side_upper_close(
        phone,
        phone_x_ratio=phone_x_ratio,
        phone_y_ratio=phone_y_ratio,
        distance_ratio=distance_ratio,
        overlap_ratio=overlap_ratio,
        config=config,
    )

    belongs = (
        point_inside_box(phone_center, expanded_driver)
        or box_iou >= config["phone_driver_iou"]
        or overlap_ratio >= config["phone_driver_overlap"]
        or close_enough
        or side_upper_close
    )
    return {
        "belongs": belongs,
        "expanded_driver": expanded_driver,
        "iou": box_iou,
        "overlap_ratio": overlap_ratio,
        "distance": distance,
        "distance_ratio": distance_ratio,
        "phone_x_ratio": phone_x_ratio,
        "phone_y_ratio": phone_y_ratio,
        "side_upper_close": side_upper_close,
    }


def _driver_body_zones(driver_box: Sequence[float]) -> dict[str, Box]:
    """Split a driver bbox into head, upper-body, and lower-body zones."""
    px1, py1, px2, py2 = _box_xyxy(driver_box)
    height = max(0.0, py2 - py1)
    head_bottom = py1 + 0.35 * height
    upper_bottom = py1 + 0.65 * height
    return {
        "head_zone": [px1, py1, px2, head_bottom],
        "upper_body_zone": [px1, py1, px2, upper_bottom],
        "lower_body_zone": [px1, upper_bottom, px2, py2],
    }


def _phone_in_zone(phone_box: Sequence[float], phone_center: Point, zone: Sequence[float]) -> bool:
    """Return True when a phone center or part of its box is in a body zone."""
    return point_inside_box(phone_center, zone) or _intersection_area(phone_box, zone) > 0


def _phone_is_side_upper_close(
    phone: Detection,
    phone_x_ratio: float,
    phone_y_ratio: float,
    distance_ratio: float,
    overlap_ratio: float,
    config: dict[str, float],
) -> bool:
    """Return True for phones held just outside the driver's upper-body box."""
    if float(phone.get("confidence", 0.0)) < config["side_phone_min_conf"]:
        return False
    if not config["side_phone_min_x_ratio"] <= phone_x_ratio <= config["side_phone_max_x_ratio"]:
        return False
    if not config["side_phone_min_y_ratio"] <= phone_y_ratio <= config["side_phone_max_y_ratio"]:
        return False
    return (
        distance_ratio <= config["side_phone_distance_ratio"]
        or overlap_ratio >= config["side_phone_overlap"]
    )


def _phone_is_low_inside_driver(
    phone: Detection,
    driver_box: Sequence[float],
    relation: dict[str, Any],
) -> bool:
    """Return True for confident low phones that are still clearly on the driver."""
    phone_center = center(phone["bbox"])
    _, phone_y_ratio = _point_relative_to_box(phone_center, driver_box)
    return (
        float(phone.get("confidence", 0.0)) >= DEFAULT_CONFIG["low_phone_min_conf"]
        and point_inside_box(phone_center, driver_box)
        and relation["overlap_ratio"] >= DEFAULT_CONFIG["low_phone_min_overlap"]
        and DEFAULT_CONFIG["low_phone_min_y_ratio"] <= phone_y_ratio <= DEFAULT_CONFIG["low_phone_max_y_ratio"]
    )


def _filter_by_class_and_conf(
    detections: list[Detection],
    class_names: set[str],
    min_confidence: float,
) -> list[Detection]:
    """Keep detections whose class alias and confidence pass the configured threshold."""
    return [
        detection
        for detection in detections
        if _normalize_class_name(detection.get("class_name")) in class_names
        and float(detection.get("confidence", 0.0)) >= min_confidence
        and area(detection["bbox"]) > 0
    ]


def _normalize_detection(detection: Detection) -> Detection:
    """Return a normalized detection with class_name, confidence, and bbox keys."""
    if "bbox" in detection:
        bbox = detection["bbox"]
    else:
        bbox = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
    return {
        **detection,
        "class_name": _normalize_class_name(detection.get("class_name")),
        "confidence": float(detection.get("confidence", detection.get("conf", 0.0))),
        "bbox": _round_box(bbox),
    }


def _box_xyxy(box: Sequence[float] | Detection) -> tuple[float, float, float, float]:
    """Read xyxy coordinates from a list/tuple or a detection-like dict."""
    if isinstance(box, dict):
        if "bbox" in box:
            box = box["bbox"]
        else:
            return (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
    if len(box) != 4:
        raise ValueError(f"Expected xyxy box with 4 values, got {box!r}")
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1, y1, x2, y2


def _image_hw(image_shape: Sequence[int]) -> tuple[int, int]:
    """Return image height and width from common HWC or HW shapes."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width.")
    return int(image_shape[0]), int(image_shape[1])


def _normalize_class_name(value: Any) -> str:
    """Normalize class names while preserving semantic aliases."""
    return str(value or "").strip().lower().replace("-", "_")


def _round_box(box: Sequence[float] | Detection) -> Box:
    """Return xyxy coordinates as rounded floats for stable output."""
    return [round(value, 2) for value in _box_xyxy(box)]


def _box_width(box: Sequence[float]) -> float:
    x1, _, x2, _ = _box_xyxy(box)
    return max(0.0, x2 - x1)


def _box_height(box: Sequence[float]) -> float:
    _, y1, _, y2 = _box_xyxy(box)
    return max(0.0, y2 - y1)


def _point_relative_to_box(point: Point, box: Sequence[float]) -> tuple[float, float]:
    """Return point coordinates normalized by a box's width and height."""
    x1, y1, _, _ = _box_xyxy(box)
    return (
        (point[0] - x1) / max(1.0, _box_width(box)),
        (point[1] - y1) / max(1.0, _box_height(box)),
    )


def _intersection_area(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1, y1, x2, y2 = _box_xyxy(box1)
    xx1, yy1, xx2, yy2 = _box_xyxy(box2)
    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)
    return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _empty_prediction(reason: str) -> dict[str, Any]:
    return {
        "driver_using_phone": False,
        "risk_score": 0.0,
        "reason": reason,
        "driver_bbox": None,
        "phone_bbox": None,
        "wheel_bbox": None,
    }


def _negative_reason(reason: str) -> str:
    if reason == "phone_near_upper_body":
        return "phone_near_upper_body_low_score"
    if reason == "phone_near_head":
        return "phone_near_head_low_score"
    return reason


def _prediction_is_positive(prediction: dict[str, Any] | bool) -> bool:
    if isinstance(prediction, dict):
        return bool(prediction.get("driver_using_phone"))
    return bool(prediction)
