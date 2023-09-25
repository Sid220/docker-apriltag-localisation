import struct
from time import time

import numpy as np
import redis


class RedisHandler:
    def __init__(self, host, port):
        self.redis = redis.Redis(host=host, port=port)
        if not self.redis.ping():
            raise Exception("Couldn't connect to Redis")

    class RedisStream:
        def __init__(self, name):
            self.name = name
            self.stream_key = f"{name}_stream"

    tvecs_stream = RedisStream("tvecs")
    camera_stream = RedisStream("camera")

    @staticmethod
    def arr_to_redis(a: np.ndarray) -> bytes:
        """Store given Numpy array 'a' in Redis under key 'n'"""
        h, w, d = a.shape
        shape = struct.pack('>III', h, w, d)
        encoded = shape + a.tobytes()

        return encoded

    @staticmethod
    def arr_from_redis(encoded: bytes) -> np.ndarray:
        """Retrieve Numpy array from Redis key 'n'"""
        h, w, d = struct.unpack('>III', encoded[:12])
        # Add slicing here, or else the array would differ from the original
        a = np.frombuffer(encoded[12:]).reshape(h, w, d)
        return a

    def send_tvecs(self, tvecs):
        if len(tvecs) == 0:
            return
        tvecs = np.array(tvecs)
        for i in range(1):
            self.redis.xadd(self.tvecs_stream.stream_key, {'ts': time(), 'tvecs': self.arr_to_redis(tvecs)})

    def send_image(self, image, cam):
        self.redis.xadd(self.camera_stream.stream_key, {'ts': time(), 'cam': cam, 'image': self.arr_to_redis(image)})
