import threading
import os
import cv2
import numpy as np

from helpers.get_conf import get_attached_cameras, get_tag_positions, get_camera_matrices, get_camera_distortions, \
    get_redis_info
from helpers.io import bcolors, prettify_bool
from helpers.redis_mgr import RedisHandler

print(bcolors.HEADER + "AprilTag Localisation Server\n" + bcolors.ENDC + """
                    ##        .            
              ## ## ##       ==            
           ## ## ## ##      ===            
       /\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\\___/ ===        
  ~~~ {~~ ~~~~ ~~~ ~~~~ ~~ ~ /  ===- ~~~   
       \\______ o          __/            
         \\    \\        __/             
          \\____\\______/                
""" + bcolors.HEADER + "\nAprilTag Localisation Server\n" + bcolors.ENDC)

# Get if production from env
production = os.getenv("DAPRILTAG_PRODUCTION") == "1"
host_id = os.getenv("DAPRILTAG_HID")
use_threads = os.getenv("DAPRILTAG_USE_THREADS") == "1"
running_in_docker = os.getenv("DAPRILTAG_RUNNING_IN_DOCKER") == "1"
attached_cameras = get_attached_cameras(host_id)
tag_positions = get_tag_positions()
camera_matrices = get_camera_matrices(attached_cameras)
camera_distortion_coefficients = get_camera_distortions(attached_cameras)
redis_info = get_redis_info()

if production:
    print("Production mode enabled")
else:
    print(bcolors.OKBLUE + "Production mode: " + prettify_bool(production))
    cv2.destroyAllWindows()
print(bcolors.OKBLUE + "Host ID: " + bcolors.ENDC + str(host_id))
print(bcolors.OKBLUE + "Using threads: " + prettify_bool(use_threads, False))
print(bcolors.OKBLUE + "Running in docker: " + prettify_bool(running_in_docker))
print(bcolors.OKBLUE + "Redis enabled: " + prettify_bool(redis_info["enabled"]))
print(bcolors.OKBLUE + "Redis host: " + bcolors.ENDC + str(redis_info["host"]))
print(bcolors.OKBLUE + "Redis port: " + bcolors.ENDC + str(redis_info["port"]))
print(bcolors.OKBLUE + "Number tags: " + bcolors.ENDC + str(len(tag_positions)))
print(bcolors.OKBLUE + "Attached cameras: " + bcolors.ENDC + str(attached_cameras))
print(bcolors.OKBLUE + "Number attached cameras: " + bcolors.ENDC + str(len(attached_cameras)))
print(bcolors.OKBLUE + "Number camera matrices: " + bcolors.ENDC + str(len(camera_matrices)))
print(bcolors.OKBLUE + "Number distortion coefficients: " + bcolors.ENDC + str(len(camera_distortion_coefficients)))
assert len(attached_cameras) == len(camera_matrices) == len(camera_distortion_coefficients), ("Every camera must have "
                                                                                              "a matrix and "
                                                                                              "distortion coefficient "
                                                                                              "defined in "
                                                                                              "camera-info.yml")

if redis_info["enabled"]:
    redis_mgr = RedisHandler(redis_info["host"], redis_info["port"])
translation_vectors = []
camera_current_frames = {}


def get_objp(id):
    if id in tag_positions:
        objp = tag_positions[id]
        return np.array([[objp["pos"]["topLeft"], objp["pos"]["topRight"], objp["pos"]["bottomLeft"],
                          objp["pos"]["bottomRight"]]]).astype(
            'float32')
    else:
        return None


def solvePnP(objp, corners, mtx, dist):
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
    return ret, rvecs, tvecs


def perform_localisation(img, mtx, dist, camera="unknown"):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(img)

    # No markers found, return
    if len(corners) == 0:
        return

    image_points = []

    for corner in corners:
        image_points += [
            [corner[0][0][0], corner[0][0][1]],
            [corner[0][1][0], corner[0][1][1]],
            [corner[0][2][0], corner[0][2][1]],
            [corner[0][3][0], corner[0][3][1]]
        ]

    # One marker found, use that
    if len(corners) == 1:
        fid_width = tag_positions[ids[0][0]]["size"]["width"]
        fid_height = tag_positions[ids[0][0]]["size"]["height"]
        object_points = np.array([[-fid_width / 2.0, fid_height / 2.0, 0.0],
                                  [fid_width / 2.0, fid_height / 2.0, 0.0],
                                  [fid_width / 2.0, -fid_height / 2.0, 0.0],
                                  [-fid_width / 2.0, -fid_height / 2.0, 0.0]])

        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(object_points, np.array(image_points),
                                                          mtx,
                                                          dist,
                                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)

            # https://github.com/Mechanical-Advantage/AdvantageKit/blob/ns-dev/akit/py/northstar/pipeline/CameraPoseEstimator.py#L98
            # field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            # field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
            # camera_to_tag_0 = np.dot(tvecs[0], cv2.Rodrigues(rvecs[0])[0])
            # camera_to_tag_1 = np.dot(rvecs[1], cv2.Rodrigues(rvecs[1])[0])
            # field_to_camera_translation_0 = np.dot(get_objp(ids[0][0]), np.linalg.inv(camera_to_tag_0))
            # field_to_camera_rotation_0 = np.dot(get_objp(ids[0][0]), np.linalg.inv(rvecs[0]))
            # field_to_camera_translation_1 = np.dot(get_objp(ids[0][0]), np.linalg.inv(camera_to_tag_1))
            # field_to_camera_rotation_1 = np.dot(get_objp(ids[0][0]), np.linalg.inv(rvecs[1]))

            translation_vectors.append(tvecs[0])
            translation_vectors.append(tvecs[1])

        except Exception as e:
            # TODO: Handle this better
            print("Error in solvingPnP (one tag): " + str(e))
            return
    else:
        # for corner, id in zip(corners, ids):
        #     if get_objp(id[0]) is None:
        #         print("Unknown ID: " + str(id))
        #         continue
        #     else:
        #         if not production:
        #             print("Found " + str(id[0]) + " in camera " + camera + "; Location: " + str(get_objp(id[0])))
        #         ret, rvecs, tvec = solvePnP(get_objp(id[0]), np.array(corner).astype('float32'), mtx, dist)
        #
        #         translation_vectors.append(tvec)
        object_points = []
        fake_object_points = []
        for tag_id in ids:
            fid_width = tag_positions[tag_id[0]]["size"]["width"]
            fid_height = tag_positions[tag_id[0]]["size"]["height"]
            base_obj_points = np.array([[-fid_width / 2.0, fid_height / 2.0, 0.0],
                                        [fid_width / 2.0, fid_height / 2.0, 0.0],
                                        [fid_width / 2.0, -fid_height / 2.0, 0.0],
                                        [-fid_width / 2.0, -fid_height / 2.0, 0.0]])
            fake_object_points += [[-fid_width / 2.0, fid_height / 2.0, 0.0],
                                   [fid_width / 2.0, fid_height / 2.0, 0.0],
                                   [fid_width / 2.0, -fid_height / 2.0, 0.0],
                                   [-fid_width / 2.0, -fid_height / 2.0, 0.0]]
            object_points += (get_objp(tag_id[0])[0]).tolist()
        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(np.array(object_points).astype("float32"),
                                                          np.array(image_points).astype("float32"),
                                                          mtx,
                                                          dist,
                                                          flags=cv2.SOLVEPNP_SQPNP)
            # https://github.com/Mechanical-Advantage/AdvantageKit/blob/ns-dev/akit/py/northstar/pipeline/CameraPoseEstimator.py#L117
            # camera_to_field_pose = openCvPoseToWpilib(tvecs[0], rvecs[0])
            # camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            # field_to_camera = camera_to_field.inverse()
            # field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())
            # field_to_camera_translation = np.linalg.inv(tvecs[0])

            translation_vectors.append(tvecs[0])
        except Exception as e:
            # TODO: Handle this better
            print("Error in solvingPnP (multitag): " + str(e))
            return

    # img = cv2.drawFrameAxes(img, mtx, dist, rvecs[0], tvecs[0], 10)


def run_on_camera(camera, cam):
    ret, frame = cam.read()

    if ret:
        perform_localisation(frame, camera_matrices[camera],
                             camera_distortion_coefficients[camera],
                             camera)
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
    captures = []
    camera_threads = []
    if not use_threads:
        for camera in attached_cameras:
            captures.append(cv2.VideoCapture(camera["vid"]))
            print("Starting camera " + str(camera["vid"]) + "; is open: " + str(captures[-1].isOpened()))
    else:
        for camera in attached_cameras:
            thread = threading.Thread(target=per_camera_thread, args=(camera["vid"],), daemon=True).start()
            camera_threads.append(thread)

    while True:
        if not use_threads:
            for camera, capture in zip(attached_cameras, captures):
                run_on_camera(camera["id"], capture)

        global translation_vectors

        if not production:
            for i in camera_current_frames:
                cv2.namedWindow("camera-" + str(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow("camera-" + str(i), camera_current_frames[i])
                # redis_mgr.send_image(camera_current_frames[i], i)

        if len(translation_vectors) > 0:
            tvec = np.mean(translation_vectors, axis=0)
            if not production:
                cv2.namedWindow("overview", cv2.WINDOW_AUTOSIZE)
                graph = np.zeros((512, 512, 3), np.uint8)
                graph = cv2.line(graph, (256, 0), (256, 512), (255, 255, 255), 1)
                graph = cv2.line(graph, (0, 256), (512, 256), (255, 255, 255), 1)
                graph = cv2.circle(graph, (int((tvec[0][0] * 0.1) + 256), int((tvec[1][0] * 0.1) + 256)), 5,
                                   (255, 255, 255), -1)
                graph = cv2.circle(graph,
                                   (int((get_objp(1)[0][0][0] * 0.1) + 256), int((get_objp(1)[0][0][1] * 0.1) + 256)),
                                   5,
                                   (255, 255, 0), -1)
                cv2.imshow("overview", graph)

        if not production:
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break

        if redis_info["enabled"]:
            redis_mgr.send_tvecs(translation_vectors)
        translation_vectors = []

    if use_threads:
        for thread in camera_threads:
            thread.join()
    else:
        for capture in captures:
            capture.release()


main()
