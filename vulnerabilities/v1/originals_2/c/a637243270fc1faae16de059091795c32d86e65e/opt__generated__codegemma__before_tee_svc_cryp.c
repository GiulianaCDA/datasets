**Purpose:**

The provided code snippet defines a function called `tee_svc_cryp_verify` within a TEE (Trusted Execution Environment) Trusted Application. It handles signature verification operations using different cryptographic algorithms.

**Function Description:**

- **Input:**
    - `state`: A pointer to a `tee_cryp_state` structure containing the cryptographic state.
    - `data`: A pointer to the data to be verified.
    - `data_len`: The length of the data.
    - `sig`: A pointer to the signature.
    - `sig_len`: The length of the signature.
    - `usr_params`: An array of user parameters.
    - `num_params`: The number of user parameters.

- **Output:**
    - A TEE status code indicating success or an error.

**Algorithm Verification:**

The function performs signature verification using the following steps:

1. **Get the cryptographic state:** It retrieves the `tee_cryp_state` structure from the session.
2. **Validate mode:** It checks if the mode is set to `TEE_MODE_VERIFY`.
3. **Check access rights:** It verifies the user's access rights to the data, signature, and key.
4. **Parse user parameters:** It converts the user parameters into a `TEE_Attribute` array.
5. **Retrieve the key:** It obtains the key associated with the cryptographic state.
6. **Determine the hash algorithm:** It extracts the hash algorithm from the cryptographic state.
7. **Calculate hash size:** It calculates the size of the hash output based on the hash algorithm.
8. **Validate data length:** It checks if the data length matches the hash size for RSA signature verification.
9. **Perform signature verification:** It uses the appropriate cryptographic algorithm (RSA, DSA, ECDSA) to verify the signature against the data and key.

**Error Handling:**

- The function returns an appropriate TEE error code if any validation fails or an unsupported algorithm is encountered.
- It frees the allocated memory and returns the error code.

**Additional Notes:**

- The `tee_mmu_check_access_rights()` function is used to enforce memory access control.
- The `crypto_*` functions handle the cryptographic operations.
- The `pkcs1_get_salt_len()` function calculates the salt length for RSA signature verification.