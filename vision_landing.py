# vision_landing.py - outline only (must be integrated to mission_control precision callback)
# Requirements: opencv-python, apriltag (pip packages), numpy

import cv2
import numpy as np
import apriltag
import math
# Camera intrinsics (replace with real calibration)
fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0
K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]])

detector = apriltag.Detector()

def detect_apriltag(frame):
    """
    Input: BGR frame (numpy)
    Output: (found:bool, tx, ty, tz) where t* are translation in meters in camera frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    if len(detections) == 0:
        return (False, None, None, None)
    # choose first detection
    d = detections[0]
    # tag size in meters (set to real size)
    tag_size = 0.3
    # corners -> object points
    object_points = np.array([[-tag_size/2, tag_size/2, 0],
                              [ tag_size/2, tag_size/2, 0],
                              [ tag_size/2,-tag_size/2, 0],
                              [-tag_size/2,-tag_size/2, 0]], dtype=np.float32)
    image_points = np.array(d.corners, dtype=np.float32)
    # solvePnP
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)
    if not ok:
        return (False, None, None, None)
    # tvec: translation vector from camera to tag (meters)
    tx, ty, tz = tvec.flatten()
    return (True, tx, ty, tz)
