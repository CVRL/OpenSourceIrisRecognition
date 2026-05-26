import numpy as np
import cv2

# ======================================================================
# Label rescaling functions (image resize)
# ======================================================================

def rescale_homography(labels, sx, sy):
    h11, h12, h13, h21, h22, h23, h31, h32 = labels
    return np.array([
        h11, (sx / sy) * h12, sx * h13,
        (sy / sx) * h21, h22, sy * h23,
        h31 / sx, h32 / sy,
    ], dtype=np.float32)

def rescale_corners(labels, sx, sy):
    x1, y1, x2, y2 = labels
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)

def rescale_circles(labels, sx, sy):
    px, py, pr, ix, iy, ir = labels
    return np.array([px * sx, py * sy, pr * sx, ix * sx, iy * sy, ir * sx], dtype=np.float32)

def rescale_eyelid_poly(labels, sx, sy):
    return np.array(labels, dtype=np.float32) * sy


# ======================================================================
# Augmentation helpers
# ======================================================================

def translate_eyelid_parabola(labels, dx, dy, target_w=640.0):
    s = dx / target_w
    ua, ub, uc, la, lb, lc = labels
    return np.array([
        ua,
        ub - 2 * ua * s,
        ua * s**2 - ub * s + uc + dy,
        la,
        lb - 2 * la * s,
        la * s**2 - lb * s + lc + dy,
    ], dtype=np.float32)

def translate_eyelid_cubic(labels, dx, dy, target_w=640.0):
    s = dx / target_w

    def _shift(a, b, c, d):
        return (
            a,
            b - 3 * a * s,
            c + 3 * a * s**2 - 2 * b * s,
            d - a * s**3 + b * s**2 - c * s + dy,
        )

    ua, ub, uc, ud, la, lb, lc, ld = labels
    return np.array([*_shift(ua, ub, uc, ud), *_shift(la, lb, lc, ld)], dtype=np.float32)

def translate_circles(labels, dx, dy, target_w=None):
    px, py, pr, ix, iy, ir = labels
    return np.array([px + dx, py + dy, pr, ix + dx, iy + dy, ir], dtype=np.float32)


# ======================================================================
# Visualization functions
# ======================================================================

def viz_homography(img, params):
    h = params
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=np.float32)
    h_img, w_img = img.shape[:2]
    return cv2.warpPerspective(img, H, (w_img, h_img), borderMode=cv2.BORDER_CONSTANT, borderValue=(192, 192, 192))

def viz_corners(img, params):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    x1, y1, x2, y2 = [int(v) for v in params]
    cv2.circle(img, (x1, y1), 6, ((0, 255, 0)), -1)
    cv2.circle(img, (x2, y2), 6, (0, 0, 255), -1)
    cv2.putText(img, "L", (x1 - 15, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0, 255, 0)), 2)
    cv2.putText(img, "R", (x2 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

def align_horizontal(img, params):
    x1, y1, x2, y2 = params[:4]
    angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(192, 192, 192))

def viz_eyelid(img, params, degree):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    h_img, w_img = img.shape[:2]
    n_coeffs = degree + 1
    t = np.linspace(0, 1, 300)
    for coeffs, color in [(params[:n_coeffs], (255, 255, 0)),
                          (params[n_coeffs:], (0, 255, 255))]:
        y = np.polyval(coeffs, t)
        x = t * w_img
        mask = (x >= 0) & (x < w_img) & (y >= 0) & (y < h_img)
        pts = np.stack([x[mask], y[mask]], axis=1).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 2)
    return img

def viz_circles(img, params):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    px, py, pr, ix, iy, ir = [int(v) for v in params]
    cv2.circle(img, (ix, iy), ir, (0, 255, 255), 2)
    cv2.circle(img, (px, py), pr, (255, 255, 0), 2)
    cv2.circle(img, (px, py), 3, (255, 255, 0), -1)
    cv2.circle(img, (ix, iy), 3, (0, 255, 255), -1)
    return img


# ======================================================================
# Task registry
# ======================================================================

TASKS = {
    "h8net": {
        "label_columns": ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"],
        "num_outputs": 8,
        "normalization": "zscore",
        "norm_axes": None,
        "has_subject_id": True,
        "subject_id_col": "subject_id",
        "filename_col": "filename",
        "use_sigmoid": False,
        "val_filter_col": None,
        "rescale_fn": rescale_homography,
        "viz_fn": viz_homography,
        "viz_kwargs": {},
    },
    "cornernet": {
        "label_columns": ["x1", "y1", "x2", "y2"],
        "num_outputs": 4,
        "normalization": "wh",
        "norm_axes": ["x", "y", "x", "y"],
        "has_subject_id": True,
        "subject_id_col": "subject_id",
        "filename_col": "filename",
        "use_sigmoid": False,
        "val_filter_col": None,
        "rescale_fn": rescale_corners,
        "viz_fn": viz_corners,
        "viz_kwargs": {},
    },
    "eyelid_parabola": {
        "label_columns": ["upper_a", "upper_b", "upper_c",
                          "lower_a", "lower_b", "lower_c"],
        "num_outputs": 6,
        "normalization": "zscore",
        "norm_axes": None,
        "has_subject_id": True,
        "subject_id_col": "subject_id",
        "filename_col": "filename",
        "use_sigmoid": False,
        "val_filter_col": None,
        "rescale_fn": rescale_eyelid_poly,
        "viz_fn": lambda img, p: viz_eyelid(img, p, degree=2),
        "viz_kwargs": {},
        "translate_fn": translate_eyelid_parabola,
        "aug_color_jitter": True,
    },
    "eyelid_cubic": {
        "label_columns": ["upper_a", "upper_b", "upper_c", "upper_d",
                          "lower_a", "lower_b", "lower_c", "lower_d"],
        "num_outputs": 8,
        "normalization": "zscore",
        "norm_axes": None,
        "has_subject_id": True,
        "subject_id_col": "subject_id",
        "filename_col": "filename",
        "use_sigmoid": False,
        "val_filter_col": None,
        "rescale_fn": rescale_eyelid_poly,
        "viz_fn": lambda img, p: viz_eyelid(img, p, degree=3),
        "viz_kwargs": {},
        "translate_fn": translate_eyelid_cubic,
        "aug_color_jitter": True,
    },
    "circlenet": {
        "label_columns": ["pupil_x", "pupil_y", "pupil_r",
                          "iris_x", "iris_y", "iris_r"],
        "num_outputs": 6,
        "normalization": "wh",
        "norm_axes": ["x", "y", "r", "x", "y", "r"],
        "has_subject_id": False,
        "subject_id_col": None,
        "filename_col": "filepath",
        "use_sigmoid": False,
        "val_filter_col": None,
        "rescale_fn": rescale_circles,
        "viz_fn": viz_circles,
        "viz_kwargs": {},
        "translate_fn": translate_circles,
        "loss_weights": [4.0, 4.0, 4.0, 1.0, 1.0, 1.0],
    },
}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Choose from: {list(TASKS.keys())}")
    return TASKS[name]
