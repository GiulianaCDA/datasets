**Code Overview:**

The provided code snippet defines functions for running various cryptographic operations on the Crypto Engine (CCP) in a secure manner. It handles commands for AES, XTS-AES, DES3, SHA, RSA, and ECC operations.

**Key Functions:**

* **ccp_run_ecc_cmd:** This function handles ECC (Elliptic Curve Cryptography) commands. It performs multiplication, addition, inversion, doubling, and padding operations based on the specified function.
* **ccp_run_cmd:** The main function that handles all CCP commands. It receives a command and determines which engine (AES, XTS-AES, etc.) to use based on the command type.
* **ccp_run_aes_cmd:** Function for running AES operations (encryption, decryption, CMAC, GCM).
* **ccp_run_xts_aes_cmd:** Function for running XTS-AES operations.
* **ccp_run_des3_cmd:** Function for running DES3 operations.
* **ccp_run_sha_cmd:** Function for running SHA operations (hashing).
* **ccp_run_rsa_cmd:** Function for running RSA operations (encryption, decryption).
* **ccp_run_passthru_cmd:** Function for running passthrough operations (copying data between memory).

**Key Points:**

* The code uses data structures called `ccp_cmd_queue` and `ccp_cmd` to represent command queues and individual commands.
* It uses the `ccp_dm_free()` function to release memory used by data structures.
* The code handles errors and returns appropriate error codes.
* It uses inline assembly code for efficient operations.

**Usage:**

To use these functions, you would need to:

1. Create a `ccp_cmd_queue` object.
2. Create a `ccp_cmd` object with the desired parameters.
3. Call the `ccp_run_cmd()` function with the command queue and command object.

**Conclusion:**

The provided code provides a comprehensive set of functions for running various cryptographic operations on the CCP, enabling secure data encryption, decryption, hashing, and other cryptographic tasks.