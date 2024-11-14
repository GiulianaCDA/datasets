  This code defines the main module entry points for the AX.25 link layer protocol implementation in Linux. The following functions are exported:
    1. ax25_init() - Called by the kernel on initialization, this function registers the AX.25 protocol with the socket layer and creates the /proc/net/ax25 directory for dumping routing information. It also sets up a packet type handler to process incoming AX.25 packets and a notifier to watch for device events.
    2. ax25_exit() - Called by the kernel on unloading, this function removes the /proc/net/ax25 directory and unregisters the AX.25 protocol from the socket layer. It also frees up resources used by the routing table, user ID table, and device list.
  The ax25_proto structure defines the properties of the AX.25 protocol for use with sockets:
    1. name - "AX.25"
    2. family - PF_AX25 (the AX.25 protocol family)
    3. create - creates a new socket
    4. owner - THIS_MODULE (this module)
  The ax25_family_ops structure defines the operations to be performed on a AX.25 protocol socket:
    1. family - PF_AX25 (the AX.25 protocol family)
    2. create - creates a new socket
    3. owner - THIS_MODULE (this module)
  The ax25_proto_ops structure defines the operations to be performed on a AX.25 protocol socket:
    1. family - PF_AX25 (the AX.25 protocol family)
    2. owner - THIS_MODULE (this module)
    3. release - releases any resources associated with the socket
    4. bind - binds a socket to a local address
    5. connect - establishes a connection to a remote address
    6. socketpair - creates a pair of connected sockets
    7. accept - accepts an incoming connection on a listening socket
    8. getname - retrieves the local or remote address associated with a socket
    9. poll - polls for events on the socket
    10. ioctl - performs operations on the socket
    11. gettstamp - retrieves the timestamp of the last packet received on the socket
    12. listen - places a socket in listening state
    13. shutdown - disables further sends and receives on a socket
    14. setsockopt - sets options on a socket
    15. getsockopt - retrieves options from a socket
    16. sendmsg - sends data on a socket
    17. recvmsg - receives data on a socket
    18. mmap - maps the file associated with the socket into memory
    19. sendpage - used for optimized kernel-based sending
  The ax25_packet_type structure is used to register a packet type handler for incoming AX.25 packets:
    1. type - ETH_P_AX25 (the AX.25 protocol)
    2. func - the function that handles incoming AX.25 packets (ax25_kiss_rcv())
  The ax25_dev_notifier structure is used to register a notifier for device events:
    1. notifier_call - the function that handles device events (ax25_device_event())