**Purpose:**

The code snippet defines a Linux kernel module for the Counter with CBC MAC (CCM) cryptographic algorithm. It provides support for the CCM, CCM-base, and RFC4309 cryptographic modes.

**Key Features:**

- Implements the CCM, CCM-base, and RFC4309 cryptographic modes.
- Uses the CBCMAC cipher as the underlying block cipher.
- Provides functions for initialization, key setting, digest calculation, and finalization.
- Supports various cryptographic attributes, such as algorithm type, block size, digest size, context size, and priority.

**Implementation:**

- The module uses the `crypto` framework to implement the CCM algorithms.
- It allocates and initializes `shash_instance` objects for each mode.
- The `crypto_cbcmac_tmpl` template defines the common attributes and operations for all CCM modes.
- The `crypto_ccm_base_tmpl`, `crypto_ccm_tmpl`, and `crypto_rfc4309_tmpl` templates specify the specific configuration for each mode.
- The `cbcmac_create()` function initializes the `shash_instance` objects for each mode.
- The `crypto_cbcmac_digest_*` functions handle the digest calculations for each mode.

**Module Initialization and Exit:**

- The `crypto_ccm_module_init()` function registers the CCM templates and initializes the module.
- The `crypto_ccm_module_exit()` function unregisters the templates and performs cleanup.

**Module Attributes:**

- `name`: "cbcmac", "ccm_base", "rfc4309", and "ccm"
- `create`: Function to initialize the corresponding shash_instance objects.
- `free`: Function to release the shash_instance objects.
- `module`: This module.

**Module License and Aliases:**

- GPL license
- Aliases for the CCM-base, RFC4309, and CCM modes.

**Usage:**

To use the CCM module, you can load the module into the kernel and use the corresponding cryptographic functions.

**Additional Notes:**

- The code assumes that the CBCMAC cipher is available.
- The digest size and block size values are based on the underlying CBCMAC cipher.
- The `crypto_check_attr_type()` function is used to validate the cryptographic attributes.