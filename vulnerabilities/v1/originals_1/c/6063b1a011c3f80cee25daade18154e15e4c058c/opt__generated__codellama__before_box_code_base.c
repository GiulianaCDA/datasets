 This is the implementation of a DVC C (DOVI configuration) box for the GPAC multimedia container library.

The `GF_DOVIDecoderConfigurationRecord` structure represents the contents of the box, and it has the following fields:

* `dv_version_major`: The major version number of the DOVI standard that this decoder implements.
* `dv_version_minor`: The minor version number of the DOVI standard that this decoder implements.
* `dv_profile`: The profile of the DOVI standard that this decoder implements.
* `dv_level`: The level of the DOVI standard that this decoder implements.
* `rpu_present_flag`: A flag indicating whether RPU (Rapid Picture Update) is supported by this decoder.
* `el_present_flag`: A flag indicating whether EL (Enhanced Layer) is supported by this decoder.
* `bl_present_flag`: A flag indicating whether BL (Base Layer) is supported by this decoder.

The `dvcC_new()` function creates a new instance of the box and returns a pointer to it. It takes no arguments.

The `dvcC_del()` function frees the memory used by the box. It takes a single argument, which is the box pointer itself.

The `dvcC_Read()` function reads the contents of the box from the bitstream and stores them in the `GF_DOVIDecoderConfigurationRecord` structure. It takes two arguments:

* `s`: The box pointer.
* `bs`: A bitstream containing the data to read from.

The `dvcC_Write()` function writes the contents of the box to the bitstream. It takes three arguments:

* `s`: The box pointer.
* `bs`: The bitstream to write to.
* `size`: The maximum size of the box in bytes. This is used for padding purposes.

The `dvcC_Size()` function computes the size of the box in bytes and stores it in the `size` field. It takes a single argument:

* `s`: The box pointer.