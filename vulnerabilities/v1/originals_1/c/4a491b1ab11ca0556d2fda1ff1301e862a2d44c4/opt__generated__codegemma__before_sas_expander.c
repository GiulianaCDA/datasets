## SAS Discovery Process

This code describes the SAS discovery process, which involves two main functions:

**1. sas_revalidate_domain:**

- Finds all SAS devices connected to a specific port.
- For each device, it iterates through its physical ports.
- For each port, it checks for presence of other devices connected through wide ports.
- If no wide ports are found, it discovers new devices on the port.

**2. sas_rediscover:**

- Determines if a specific SAS device originated a BROADCAST(CHANGE) message.
- If so, it checks if any other physical port in the expander is connected to the same device.
- If so, it revalidates the entire domain.
- If not, it discovers new devices on the specified port.

**Additional Notes:**

- The discovery process quits and returns if any connection errors are encountered.
- The smp_handler function handles SMP requests, which are used to communicate with SAS devices.
- The code is written in C and uses the SAS Device Management (SDM) API.

## Summary

The SAS discovery process is responsible for finding all SAS devices connected to a network and ensuring that they are properly connected to each other. It works by iterating through physical ports, checking for wide ports, and discovering new devices as needed.