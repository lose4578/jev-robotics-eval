"""Optional projected navigation labels; no simulator state is modified."""

import numpy as np
from PIL import ImageDraw


def project_point(env, camera_name, point, size):
    width, height = size
    camera = env.model.camera(camera_name).id
    rotation = env.data.cam_xmat[camera].reshape(3, 3)
    q = rotation.T @ (np.asarray(point) - env.data.cam_xpos[camera])
    if q[2] >= -1e-6:
        return None
    focal = height / (2 * np.tan(np.deg2rad(env.model.cam_fovy[camera]) / 2))
    x = width / 2 + focal * q[0] / -q[2]
    y = height / 2 - focal * q[1] / -q[2]
    if camera_name == "corner2":
        x, y = width - 1 - x, height - 1 - y
    return (float(x), float(y)) if 0 <= x < width and 0 <= y < height else None


def annotate_frame(frame, env, camera_name, state):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    markers = [("TCP", state["control_xyz"], "#00e5ff", 3)]
    waypoint = state.get("active_waypoint")
    if waypoint:
        markers.append(("WAYPOINT", waypoint["target_xyz"], "#ffad33", 6))
        draw.rectangle((0, 0, frame.width, 17), fill="#172032")
        draw.text((4, 2), f"Phase: {waypoint['phase']}", fill="white")
    for label, xyz, color, radius in markers:
        pixel = project_point(env, camera_name, xyz, frame.size)
        if pixel is None:
            continue
        x, y = pixel
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        draw.text((max(0, min(x+8, frame.width-65)), max(18, min(y-12, frame.height-12))), label, fill=color)
    return frame
