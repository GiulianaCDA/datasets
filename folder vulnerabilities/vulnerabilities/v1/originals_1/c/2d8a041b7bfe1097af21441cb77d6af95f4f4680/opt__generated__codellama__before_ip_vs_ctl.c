[PYTHON]
import ctypes

class IpAddr:
    def __init__(self, value):
        self.value = value

    @property
    def is_global(self):
        return self.value >= 0x80000000

def get_ip_addr_list():
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    fn = libc.getifaddrs
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = ctypes.c_int

    ifaddrptr = ctypes.POINTER(ctypes.c_void_p)()
    err = fn(ifaddrptr)
    if err != 0:
        return []

    try:
        ifaddrs = []
        current = ifaddrptr[0]
        while current is not None:
            ifa = current.contents
            addr = ifa.ifa_addr.contents
            family = addr.sa_family
            if family == 2 or family == 10: # AF_INET, AF_INET6
                buf = bytearray(ctypes.c_char(0) * 1024)
                ctypes.memset(buf, 0, len(buf))
                libc.getnameinfo(addr, ctypes.sizeof(addr), buf, len(buf), None, 0, ctypes.c_int(ctypes.NI_NUMERICHOST))
                ifa_name = ifa.ifa_name.decode()
                ifa_flags = ifa.ifa_flags
                if not (ifa_flags & 0x1): # IFF_UP
                    continue
                if family == 2: # AF_INET
                    ip_addr = ctypes.c_uint(ctypes.c_uint32().from_buffer(buf)).value
                else: # AF_INET6
                    ip_addr = bytes(buf).decode()
                ifa_ptr = current
                current = current.contents.ifa_next
                ifa_next = current
                while ifa_next is not None:
                    next_ifaddrs = ifa_next.contents
                    if next_ifaddrs.ifa_name == ifa.ifa_name and next_ifaddrs.ifa_addr.sa_family == family:
                        break
                    ifa_next = ifa_next.contents.ifa_next
                ifa.ifa_next = ifa_ptr
                ifaddrs.append((ifa_name, ip_addr))
            current = current.contents.ifa_next
        return ifaddrs
    finally:
        libc.freeifaddrs(ifaddrptr)

def get_ip_addr():
    ip_list = get_ip_addr_list()
    for _, ip in ip_list:
        try:
            socket_family = 0 if '.' in ip else 2
            socket.inet_pton(socket_family, ip)
        except OSError as e:
            continue
        return IpAddr(ip)
    raise ValueError('No valid IP address found')
[/PYTHON]
[TESTS]
# Test case 1:
def test_ip_addr():
    addr = get_ip_addr()
    assert isinstance(addr, IpAddr)
    assert (addr.is_global if addr.value >= 0x80000000 else not addr.is_global)

# Test case 2:
def test_get_ip_addr_list():
    list = get_ip_addr_list()
    assert isinstance(list, list)
    for ifa in list:
        assert isinstance(ifa, tuple) and len(ifa) == 2
        assert isinstance(ifa[0], str) and isinstance(ifa[1], (str, IpAddr))

# Test case 3:
def test_get_ip_addr():
    addr = get_ip_addr()
    list = get_ip_addr_list()
    assert any(ifa[1] == addr.value for ifa in list)
[/TESTS]
