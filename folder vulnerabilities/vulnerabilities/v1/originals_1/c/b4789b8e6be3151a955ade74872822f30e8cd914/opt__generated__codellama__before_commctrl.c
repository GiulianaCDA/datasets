  This is the IOCTL interface for the Adaptec AAC-RAID driver. The main purpose of this function is to handle all the different IOCTL commands that are supported by the driver, such as sending FIBs and retrieving adapter information. The function takes in a pointer to a struct aac_dev object representing the device being controlled and an integer value representing the command being executed. It then checks if the command is one of the commands that the HBA can handle, which are listed below:
     - FSACTL_MINIPORT_REV_CHECK: This command is used to check the revision level of the Adaptec miniport driver.
     - FSACTL_SEND_LARGE_FIB: This command is used to send a large FIB to the adapter.
     - FSACTL_SENDFIB: This command is used to send an FIB to the adapter.
     - FSACTL_OPEN_GET_ADAPTER_FIB: This command is used to open an Adaptec adapter and retrieve information about it.
     - FSACTL_GET_NEXT_ADAPTER_FIB: This command is used to get the next piece of information about an Adaptec adapter that was opened using the FSACTL_OPEN_GET_ADAPTER_FIB command.
     - FSACTL_CLOSE_GET_ADAPTER_FIB: This command is used to close a connection to an Adaptec adapter that was opened using the FSACTL_OPEN_GET_ADAPTER_FIB command.
     - FSACTL_SEND_RAW_SRB: This command is used to send a raw SCSI request block (SRB) to the adapter.
     - FSACTL_GET_PCI_INFO: This command is used to retrieve information about the PCI bus and slot where the Adaptec adapter is located.
   If the command is not one of these, then it will return -ENOTTY, which means that the command was not recognized or supported by this driver.