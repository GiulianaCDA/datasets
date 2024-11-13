 This is a Linux kernel module for the USB CDC NCM (Network Control Model) host driver. The purpose of this code is to allow communication between a USB device that uses CDC-NCM and a computer running a Linux kernel.

   The module defines a list of supported devices in `cdc_devs`, which includes various Mobile Broadband Modems (MBMs), such as the DW5812 LTE Verizon Mobile Broadband Card, Dell branded MBM devices, Toshiba branded MBM devices, Huawei devices, and Infineon(now Intel) HSPA Modem platform.

   The module also includes a Generic CDC-NCM section for USB devices that use the CDC-NCM protocol but are not specifically listed in `cdc_devs`. Each entry in `cdc_devs` includes information about the device, such as its vendor and product ID, interface class, subclass, and protocol.

   The module uses the `usbnet_probe`, `usbnet_disconnect`, `usbnet_suspend`, and `usbnet_resume` functions to handle events such as connecting or disconnecting a USB device, suspending or resuming the computer, and resetting the USB bus.

   Finally, the module is registered with the Linux kernel using `module_usb_driver(cdc_ncm_driver)`. The author of this code is Hans Petter Selasky, and the license is Dual BSD/GPL.