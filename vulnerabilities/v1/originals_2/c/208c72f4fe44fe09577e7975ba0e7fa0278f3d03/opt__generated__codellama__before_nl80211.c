  This is a portion of the nl80211.c file from the Linux kernel source code. It contains various functions related to creating and managing netlink sockets for communication between the kernel and userspace applications, as well as handling messages sent over those sockets. The specific function you are interested in is "nl80211_mlme_notify()", which sends a notification message to all registered listeners on a given netlink socket about a change in the wireless network state (e.g., a new beacon frame received from an access point).

The function takes several arguments:

* nb: A pointer to a structure that represents the notifier block being used to send the message, which is defined in "include/linux/netdevice.h" and contains the callback functions for handling the notification (in this case, it's "nl80211_notify()")
* state: An unsigned long value indicating what type of event has occurred (e.g., a change in the wireless network state)
* _notify: A pointer to a structure containing information about the specific event that has occurred

The function first retrieves the wireless device and registered socket from the notification data using rcu_read_lock() and list_for_each_entry_rcu(). Then it creates a new netlink message using genlmsg_new() and sets its headers using genlmsg_put() with a length of 0, an operation ID (nl80211_cmd), a flags value of 0, and the family ID (the 802.11 subsystem). Next, it uses nla_put() to add the relevant attributes (e.g., NL80211_ATTR_IFINDEX, NL80211_ATTR_STA_ADDR, etc.) to the message, followed by an nla_nest_start() call to start a new attribute for the per-device information.

It then uses nla_put() again to add the "NL80211_ATTR_CQM_PKT_LOSS_EVENT" attribute (a 32-bit value containing the number of packets that have been lost), followed by an nla_nest_end() call to end the per-device information. Finally, it uses genlmsg_end() and genlmsg_multicast_netns() to finish off the message and send it over the netlink socket for the relevant wireless device using a multicast group ID (nl80211_mlme_mcgrp).

I hope that helps! Let me know if you have any further questions.