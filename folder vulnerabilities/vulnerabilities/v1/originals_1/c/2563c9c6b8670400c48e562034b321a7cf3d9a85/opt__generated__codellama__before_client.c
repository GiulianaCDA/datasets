[PYTHON]
import ctypes as ct
from datetime import datetime

LIB = None
TIME_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

def init(library_path):
    global LIB
    LIB = ct.CDLL(library_path)

def datetime_to_timespec(dt):
    ts = ct.c_longlong()
    ts.value = (dt - datetime(1970, 1, 1)).total_seconds()
    return ts

def timespec_to_datetime(ts):
    return datetime(1970, 1, 1) + timedelta(seconds=ts.value)

def nbd_init(fd, ioc, flags, size):
    return LIB.nbd_init(fd, ct.c_void_p(ioc), ct.c_uint16(flags), ct.c_int64(size))

def nbd_client(fd):
    return LIB.nbd_client(fd)

def nbd_disconnect(fd):
    return LIB.nbd_disconnect(fd)

def nbd_send_request(ioc, request):
    buf = (ct.c_uint8 * NBD_REQUEST_SIZE)()
    buf[0] = NBD_REQUEST_MAGIC
    buf[1] = ct.c_uint16(request.flags)
    buf[2] = ct.c_uint16(request.type)
    buf[3] = ct.c_uint64(request.handle)
    buf[4] = ct.c_uint64(request.from_)
    buf[5] = ct.c_uint32(request.len)
    return LIB.nbd_send_request(ct.c_void_p(ioc), buf)

def nbd_receive_reply(ioc, reply):
    buf = (ct.c_uint8 * NBD_REPLY_SIZE)()
    ret = LIB.nbd_receive_reply(ct.c_void_p(ioc), buf)
    if ret < 0:
        return ret
    reply.error = ct.c_uint32(buf[4]).value
    reply.handle = ct.c_uint64(buf[8]).value
    return ret
[/PYTHON]
