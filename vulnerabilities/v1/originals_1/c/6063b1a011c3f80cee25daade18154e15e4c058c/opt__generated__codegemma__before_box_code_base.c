**Purpose:**

The code snippet you provided defines a set of functions for parsing and writing Dolby Vision (DV) configuration data in an ISO file.

**Key Features:**

* **`dvcC_New()`:** Creates a new `GF_DOVIConfigurationBox` instance.
* **`dvcC_del()`:** Frees the memory allocated for a `GF_DOVIConfigurationBox` instance.
* **`dvcC_Read()`:** Reads Dolby Vision configuration data from a bitstream.
* **`dvcC_Write()`:** Writes Dolby Vision configuration data to a bitstream.
* **`dvcC_Size()`:** Calculates the size of the Dolby Vision configuration box.

**Data Structure:**

The `GF_DOVIConfigurationBox` structure contains the following fields:

* `type`: Box type identifier.
* `size`: Box size.
* `DOVIConfig`: Dolby Vision configuration parameters.

**Usage:**

To use the functions, follow these steps:

1. Create a new `GF_Box` instance of type `GF_ISOM_BOX_TYPE_DVCC`.
2. Call `dvcC_Read()` to parse Dolby Vision configuration data from a bitstream.
3. Call `dvcC_Write()` to write Dolby Vision configuration data to a bitstream.
4. Call `dvcC_Size()` to calculate the size of the Dolby Vision configuration box.

**Assumptions:**

* The code assumes that the ISO file format is being used.
* The Dolby Vision configuration data is in bitstream format.

**Additional Notes:**

* The code includes error handling for invalid data.
* The `GPAC_DISABLE_ISOM_WRITE` macro controls whether the write functions are included.