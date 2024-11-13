**Module Description:**

This module is a USB CDC NCM (Communications Device Class - Non-Controlled Markup) host driver. It supports various USB devices, including:

- Dell MBM devices
- Toshiba MBM devices
- Huawei devices
- Infineon HSPA modems
- Generic CDC-NCM devices

**Functionality:**

- Detects and connects to CDC-NCM devices based on the specified device IDs and interface information.
- Uses the `usbnet` driver to handle USB data transfer and device operations.
- Provides a `cdc_ncm_driver` structure to represent the driver.
- Uses the `cdc_devs` device table to define the supported devices.

**Supported Devices:**

- Dell branded MBM devices (with or without `FLAG_NOARP`)
- Toshiba branded MBM devices (with or without `FLAG_NOARP`)
- Huawei devices
- Infineon HSPA modems
- Generic CDC-NCM devices

**Configuration:**

- The `cdc_devs` device table defines the supported devices.
- The `cdc_ncm_driver` structure specifies the driver name, probe and disconnect functions, and other driver properties.

**Usage:**

This module is automatically loaded when the kernel boots. It detects and connects to supported devices and provides a CDC-NCM interface for communication.

**Additional Notes:**

- The `FLAG_NOARP` flag is used for certain devices that require additional configuration or handling.
- The driver supports autosuspend and hub-initiated LPM.

**Module Information:**

- Author: Hans Petter Selasky
- License: Dual BSD/GPL
- Description: USB CDC NCM host driver