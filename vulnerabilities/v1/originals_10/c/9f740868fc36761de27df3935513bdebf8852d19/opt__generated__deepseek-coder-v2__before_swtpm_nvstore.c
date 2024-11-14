 The provided code snippets are from a software TPM emulator (swtpm) that handles NVRAM operations and prints JSON output. Let's break down the functionality of each snippet:

### Function: `SWTPM_NVRAM_MigrateBlob`
This function is responsible for decrypting a blob using either a migration key or a state encryption key, depending on whether it is encrypted with a 256-bit key or not. Here's the breakdown of its steps:
1. **Check if Migration Key is Available**: If the blob is encrypted with a 256-bit key (BLOB_FLAG_MIGRATION_256BIT_KEY), it tries to decrypt using this key.
2. **Decrypt Using Migration Key**: If available, it uses `SWTPM_NVRAM_DecryptData` with the migration key.
3. **Direct Decryption if Not Encrypted**: If not encrypted or only encrypted with a 256-bit key, it directly decrypts using `SWTPM_NVRAM_GetPlainData`.
4. **Further Decryption Using State Key**: If the decrypted data is still encrypted (indicated by `is_encrypted` or `BLOB_FLAG_ENCRYPTED`), it attempts to decrypt again using the state encryption key.
5. **SetState**: Finally, if decryption is successful, it copies the decrypted data into a TPM state (`st`) and frees allocated memory.

### Function: `SWTPM_NVRAM_PrintJson`
This function generates a JSON representation of the NVRAM states managed by swtpm. It does so by querying each predefined state (permanent all, volatile, save) to get their sizes and then formatting this information into a JSON array under the "states" key. The function:
1. **Initializes NVRAM**: Calls `SWTPM_NVRAM_Init` to ensure the backend is initialized.
2. **Iterates Over States**: Checks each predefined state (defined in an array) for existence using `g_nvram_backend_ops->check_state`.
3. **Formats JSON**: If a state exists, it appends its details (name and size) to a string buffer, which is then formatted into the final JSON structure along with other metadata.
4. **Handles Errors**: Stops processing on encountering errors or if a state does not exist.
5. **Prints JSON**: Outputs the constructed JSON object when all checks are done successfully.

### Summary
Both functions interact with NVRAM operations and handle different aspects of data decryption and retrieval, providing crucial functionality for TPM emulation in software environments. They also format output as JSON, which is useful for reporting or integration with other systems requiring a structured representation of the state information.