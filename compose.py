from jinja_compose_wrapper.libjinja_compose import JinjaComposeInject
import socket


class HostMethods(JinjaComposeInject):
    @staticmethod
    def is_leader():
        return HostMethods.get_host_id() == "0"

    @staticmethod
    def get_host_id():
        try:
            return socket.gethostname().split("dapriltag-")[1]
        except IndexError:
            return "0"
