import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def get_keypoints_rectangle(keypoints, threshold):
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img, keypoints, pairs, colors, thicknessCircleRatio, thicknessLineRatioWRTCircle, poseScales, threshold=0.1, alpha=1.0):
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    numberScales = len(poseScales)
    thresholdRectangle = 0.1
    numberKeypoints = keypoints.shape[0]

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thicknessCircleRatio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thicknessLineRatioWRTCircle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * poseScales[0]))
                colorIndex = index2
                # color = colors[colorIndex % numberColors]
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int)
                keypoint2 = keypoints[index2, :-1].astype(np.int)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * poseScales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * poseScales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    img = alpha * img + (1-alpha) * img_orig
    return img

def render_body_keypoints(img, body_keypoints, threshold, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        thicknessCircleRatio = 1./75 * map_fn(body_keypoints[:, -1])
    else:
        thicknessCircleRatio = 1./75. * np.ones(body_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = []
    pairs = [1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21,11,22,22,23,11,24]
    pairs = np.array(pairs).reshape(-1,2)
    colors = [255.,     0.,     85.,
              255.,     0.,     0.,
              255.,    85.,     0.,
              255.,   170.,     0.,
              255.,   255.,     0.,
              170.,   255.,     0.,
               85.,   255.,     0.,
                0.,   255.,     0.,
              255.,     0.,     0.,
                0.,   255.,    85.,
                0.,   255.,   170.,
                0.,   255.,   255.,
                0.,   170.,   255.,
                0.,    85.,   255.,
                0.,     0.,   255.,
              255.,     0.,   170.,
              170.,     0.,   255.,
              255.,     0.,   255.,
               85.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.]
    colors = np.array(colors).reshape(-1,3)
    poseScales = [1]
    return render_keypoints(img, body_keypoints, pairs, colors, thicknessCircleRatio, thicknessLineRatioWRTCircle, poseScales, 0.1, alpha=alpha)

def render_openpose(img, body_keypoints, threshold=0.01, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0, plot_body=True, plot_hands=True, plot_face=True):
    img = render_body_keypoints(img, body_keypoints, threshold=threshold, use_confidence=use_confidence, map_fn=map_fn, alpha=alpha)
    return img
