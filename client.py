import redis
import numpy as np
import cv2

from src.helpers.circular_buffer import CircularBuffer
from src.helpers.get_conf import get_redis_info
from src.helpers.redis_mgr import RedisHandler

redis_info = get_redis_info(False)
redis = redis.Redis(host="localhost", port=redis_info["port"])
assert redis.ping()

buffer = CircularBuffer(10)

l = redis.xread({"tvecs_stream": "$"}, block=0)
while True:
    print("New Loop")
    print(l)

    last_id_returned = l[0][1][-1][0]
    l = redis.xread({"tvecs_stream": last_id_returned}, count=10, block=0)
    print(l)

    translation_vectors = RedisHandler.arr_from_redis(l[0][1][0][1][b'tvecs'])
    tvec = np.mean(translation_vectors, axis=0)

    cv2.namedWindow("overview2", cv2.WINDOW_AUTOSIZE)
    graph = np.zeros((512, 512, 3), np.uint8)
    graph = cv2.line(graph, (256, 0), (256, 512), (255, 255, 255), 1)
    graph = cv2.line(graph, (0, 256), (512, 256), (255, 255, 255), 1)
    graph = cv2.circle(graph, (int((tvec[0][0] * 0.1) + 256), int((tvec[1][0] * 0.1) + 256)), 5,
                       (255, 255, 255), -1)
    # graph = cv2.circle(graph,
    #                    (int((get_objp(1)[0][0][0] * 0.1) + 256), int((get_objp(1)[0][0][1] * 0.1) + 256)),
    #                    5,
    #                    (255, 255, 0), -1)
    cv2.imshow("overview2", graph)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break
    # buffer.append(l[0][1])
    # print("New Loop")
    # print(l)
    # for id, value in l[0][1]:
    #     print(f"id: {id} value: {value[b'ts']}")

