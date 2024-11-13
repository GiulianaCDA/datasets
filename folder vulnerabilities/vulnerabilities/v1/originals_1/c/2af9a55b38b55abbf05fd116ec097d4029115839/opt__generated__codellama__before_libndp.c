[PYTHON]
import logging
from ctypes import *
from typing import List

class Ndp:
    def __init__(self, sock: int):
        self.sock = sock
        self.msgrcv_handler_list: List[int] = []
        self.log_priority: int = logging.ERROR
    
    def set_log_priority(self, priority: int) -> None:
        self.log_priority = priority
    
    def add_msgrcv_handler(self, handler: int) -> None:
        self.msgrcv_handler_list.append(handler)
    
    def call_eventfd_handler(self) -> int:
        # TODO: Implement this method
        return 0
    
    def callall_eventfd_handler(self) -> int:
        # TODO: Implement this method
        return 0

class MsgrcvHandler:
    def __init__(self, handler: int):
        self.handler = handler

def ndp_open() -> Ndp:
    # TODO: Implement this function
    sock = 0
    ndp = Ndp(sock)
    return ndp

def ndp_close(ndp: Ndp) -> None:
    # TODO: Implement this function
    pass

def ndp_get_eventfd(ndp: Ndp) -> int:
    # TODO: Implement this function
    return 0

def ndp_call_eventfd_handler(ndp: Ndp) -> int:
    # TODO: Implement this function
    return 0

def ndp_callall_eventfd_handler(ndp: Ndp) -> int:
    # TODO: Implement this function
    return 0

def main() -> None:
    # TODO: Implement this function
    ndp = ndp_open()
    ndp.set_log_priority(logging.DEBUG)
    ndp.add_msgrcv_handler(1234)
    print("callall_eventfd_handler:", ndp_callall_eventfd_handler(ndp))
    print("call_eventfd_handler:", ndp_call_eventfd_handler(ndp))
    ndp_close(ndp)

if __name__ == "__main__":
    main()
[/]
