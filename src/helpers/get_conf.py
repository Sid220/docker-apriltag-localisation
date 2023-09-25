import numpy as np
import yaml


def get_attached_cameras(host_id):
    with open("conf/connected-cameras.yml", "r") as stream:
        try:
            cameras = yaml.safe_load(stream)
            cameras = [camera for camera in cameras if camera["attached_to"] == int(host_id)]
            return cameras
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load camera info")


def get_camera_vid(camera, attached_cameras):
    for cam in attached_cameras:
        if cam["id"] == camera:
            return cam["vid"]


def get_camera_info(attached_cameras, info):
    with open("conf/camera-info.yml", "r") as stream:
        try:
            cameras = yaml.safe_load(stream)
            ret = {}
            for camera in attached_cameras:
                for cam in cameras:
                    if cam["id"] == camera["id"]:
                        ret[camera["id"]] = np.array(cam[info])
            return ret
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load camera info")


def get_camera_matrices(attached_cameras):
    return get_camera_info(attached_cameras, "mtx")


def get_camera_distortions(attached_cameras):
    return get_camera_info(attached_cameras, "dist")


def get_tag_positions():
    with open("conf/apriltag-info.yml", "r") as stream:
        try:
            objps = yaml.safe_load(stream)
            return objps
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load apriltag info")


def get_redis_info():
    with open("conf/redis-info.yml", "r") as stream:
        try:
            redis_info = yaml.safe_load(stream)
            return redis_info
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Couldn't load redis info")
