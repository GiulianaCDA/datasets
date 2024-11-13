[PYTHON]
def rtnetlink_init():
    global rta_max
    rta_max = [0] * 32
    for i in range(len(rta_max)):
        if rta_max[i] > rtattr_max:
            rtattr_max = rta_max[i]
    global rta_buf
    rta_buf = [None] * rtattr_max
    for i in range(len(rta_buf)):
        if rta_buf[i] == None:
            rta_buf[i] = rtnetlink.rtattr()

def register_netdevice():
    global netdevices
    netdevices = {}
    for k, v in network.netdevices.items():
        if not v.name.startswith("lo"):
            netdevices[k] = v

def rtnetlink_net_init(net):
    global sock
    sk = netlink_kernel_create(net, NETLINK_ROUTE, {
        'groups': RTNLGRP_MAX,
        'input': rtnetlink_rcv,
        'cb_mutex': &rtnl_mutex,
        'flags': NL_CFG_F_NONROOT_RECV,
    })
    if not sk:
        panic("rtnetlink_init: cannot allocate rta_buf\n")
    net.rtnl = sk

def rtnetlink_net_exit(net):
    netlink_kernel_release(net.rtnl)
    net.rtnl = None

def __net_init():
    register_pernet_subsys(&rtnetlink_net_ops)

def __net_exit():
    unregister_pernet_subsys(&rtnetlink_net_ops)

def rtnetlink_init():
    global rta_max
    for i in range(len(rta_buf)):
        if rta_buf[i] == None:
            rta_buf[i] = rtnetlink.rtattr()
    register_pernet_subsys(&rtnetlink_net_ops)
    register_netdevice()

if __name__ == "__main__":
    rtnetlink_init()

 [/PYTHON]
