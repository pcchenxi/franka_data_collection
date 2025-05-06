import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys
import pyrealsense2 as rs

def get_RGBframe(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color: 
        return None
    else:
        color_np = np.asanyarray(color.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

        return color

def pose_esitmation(frame, aruco_dict_type, aruco_side_length, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)
    frame = aruco_display(corners, ids, rejected_img_points, frame)

    # # If markers are detected
    # if len(corners) > 0:
    #     for i in range(0, len(ids)):
    #         # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
    #         rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_side_length, matrix_coefficients, distortion_coefficients)
    #         # Draw a square around the markers
    #         cv2.aruco.drawDetectedMarkers(frame, corners) 
    #         # Draw Axis
    #         cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
    #         print('id', ids[i])
    #         print('tvec', tvec)
    #         # print('rvec', rvec)
    return frame

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
ap.add_argument("-l", "--length", type=float, default=0.0375, help="Type of ArUCo tag to detect")

args = vars(ap.parse_args())

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

selected_serial = "109622072337" # "315122271073" "109622072337" "419122270338"
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(selected_serial)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intri = color_stream.as_video_stream_profile().get_intrinsics()

k = np.array([[intri.fx, 0, intri.ppx],[0, intri.fy, intri.ppy], [0, 0, 1]])
d = np.array([intri.coeffs])

print(intri)
print(k)
print(d)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

while True:
    frame = get_RGBframe(pipeline)
    if frame is None:
        continue
	
    # h, w, _ = frame.shape
    # output = pose_esitmation(frame, ARUCO_DICT[args["type"]], 0.0375, k, d)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    output = aruco_display(corners, ids, rejected_img_points, frame)

    print(corners)

    cv2.imshow("Image", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
pipeline.stop()

