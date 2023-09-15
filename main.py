import math
import threading
import time
import os
import cv2
import numpy as np
import yaml

# Get if production from env
print("Starting server...")
production = os.getenv("DAPRILTAG_PRODUCTION") == "1"

if production:
    print("Production mode enabled")
else:
    print("Production mode disabled")
    cv2.destroyAllWindows()

use_threads = False

tvecs = []
objps = []


def get_objp(id):
    if id in objps:
        objp = objps[id]
        return np.array([[objp["pos"]["topLeft"], objp["pos"]["topRight"], objp["pos"]["bottomLeft"],
                          objp["pos"]["bottomRight"]]]).astype(
            'float32')
    else:
        return None


def solvePnP(objp, corners, mtx, dist):
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
    return ret, rvecs, tvecs


def perform_localisation(img, mtx, dist):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(img)

    if len(corners) == 0:
        return
    for corner, id in zip(corners, ids):
        if get_objp(id[0]) is None:
            print("Unknown ID: " + str(id))
            continue
        else:
            if not production:
                print("Found " + str(id) + "; Location: " + str(get_objp(id[0])))
            ret, rvecs, tvec = solvePnP(get_objp(id[0]), np.array(corner).astype('float32'), mtx, dist)

            tvecs.append(tvec)

        img = cv2.drawFrameAxes(img, mtx, dist, rvecs, tvec, 10)


cameras = []
camera_current_frames = {}

camera_matricies = {}
camera_dists = {}


def get_camera_matrix(camera):
    if camera in camera_matricies:
        return camera_matricies[camera]

    with open("conf/camerainfos.yml", "r") as stream:
        try:
            cameras = yaml.safe_load(stream)
            ret = np.array(cameras[camera]["mtx"])
            camera_matricies[camera] = ret
            return ret
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load camera info")


def get_camera_distortion(camera):
    if camera in camera_dists:
        return camera_dists[camera]

    with open("conf/camerainfos.yml", "r") as stream:
        try:
            cameras = yaml.safe_load(stream)
            ret = np.array(cameras[camera]["dist"])
            camera_dists[camera] = ret
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load camera info")


def run_on_camera(camera, cam):
    print("Video capture created for camera " + str(camera))
    print(cam.isOpened())
    print(cam)

    ret, frame = cam.read()

    print(ret)

    if ret:
        perform_localisation(frame, get_camera_matrix(camera), get_camera_distortion(camera))
        camera_current_frames[camera] = frame
    else:
        raise Exception("Couldn't grab camera")


def per_camera_thread(camera):
    print("Starting camera " + str(camera))

    cam = cv2.VideoCapture(camera)

    while True:
        ret, frame = cam.read()

        if ret:
            perform_localisation(frame)
            camera_current_frames[camera] = frame
        else:
            raise Exception("Couldn't grab camera")


def main():
    with open("conf/apriltaglocs.yml", "r") as stream:
        try:
            global objps
            objps = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load apriltag info")

    with open("conf/installable/connectedcameras.yml", "r") as stream:
        try:
            global cameras
            cameras = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load camera info")

    captures = []
    camera_threads = []
    if not use_threads:
        for camera in cameras:
            captures.append(cv2.VideoCapture(camera))
    else:
        for camera in cameras:
            thread = threading.Thread(target=per_camera_thread, args=(camera,), daemon=True).start()
            camera_threads.append(thread)

    while True:
        if not use_threads:
            for camera, capture in zip(cameras, captures):
                run_on_camera(camera, capture)

        global tvecs
        print(tvecs)
        print("")

        if not production:
            for i in camera_current_frames:
                cv2.namedWindow("test" + str(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow("test" + str(i), camera_current_frames[i])

        if len(tvecs) > 0:
            tvec = np.mean(tvecs, axis=0)
            if not production:
                cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
                graph = np.zeros((512, 512, 3), np.uint8)
                graph = cv2.line(graph, (256, 0), (256, 512), (255, 255, 255), 1)
                graph = cv2.line(graph, (0, 256), (512, 256), (255, 255, 255), 1)
                graph = cv2.circle(graph, (int((tvec[0][0] * 0.1) + 256), int((tvec[1][0] * 0.1) + 256)), 5,
                                   (255, 255, 255), -1)
                graph = cv2.circle(graph,
                                   (int((get_objp(1)[0][0][0] * 0.1) + 256), int((get_objp(1)[0][0][1] * 0.1) + 256)), 5,
                                   (255, 255, 0), -1)
                cv2.imshow("test", graph)

        if not production:
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        tvecs = []

    if use_threads:
        for thread in camera_threads:
            thread.join()
    else:
        for capture in captures:
            capture.release()


main()
