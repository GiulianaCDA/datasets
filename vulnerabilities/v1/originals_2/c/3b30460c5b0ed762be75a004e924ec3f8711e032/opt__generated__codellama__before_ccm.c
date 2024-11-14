  This is a kernel module for Counter with CBC MAC (CCM) encryption. It uses the `crypto_register_template` function to register the CCM template, which allows userspace applications to use the CCM algorithm through the `cryptodev` device.

The module also includes a `crypto_cbcmac` template for CBC-MAC (Cipher Block Chaining Message Authentication Code) encryption, which is used as a building block for the CCM algorithm.

The module exports several functions:

* `crypto_ccm_module_init`: Initializes the CCM and CBC-MAC templates and registers them with the kernel.
* `crypto_ccm_module_exit`: Unregisters the CCM and CBC-MAC templates from the kernel.

The module also includes a `cbcmac_create` function that creates an instance of the CBC-MAC algorithm, which is then registered with the kernel.

This module uses the `crypto_mod_put` function to decrement the reference count of the algorithm when it is unloaded from the kernel. The `shash_free_instance` function is used to free the instance of the CBC-MAC algorithm.

The module also includes a `crypto_rfc4309_tmpl` template that registers the RFC 4309 algorithm, which is used for IPsec ESP encryption.