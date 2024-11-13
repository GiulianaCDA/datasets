 This code defines a set of functions for handling Dolby Vision (`dvcC`) configuration boxes within an ISO media file. The `dvcC` box is used to store the Dolby Vision decoder configuration, which includes version information, profile, level, and flags indicating the presence of various elements like mastering display color volume (MDCV) and EL metadata.

Here's a breakdown of the functions defined in this code:

1. **dvcC_New()**: This function allocates memory for a new `GF_DOVIConfigurationBox` and initializes it with default values, setting its type to `GF_ISOM_BOX_TYPE_DVCC`. It returns a pointer to the newly allocated box or NULL if allocation fails.

2. **dvcC_del()**: This function releases the memory associated with a given `GF_Box` pointer by casting it to a `GF_DOVIConfigurationBox*`, then freeing that memory using `gf_free`.

3. **dvcC_Read()**: Reads the Dolby Vision configuration from a bitstream (`GF_BitStream`), extracting version major, minor, profile, level, and flags indicating the presence of RPU (Reference Picture Usage), EL (Electrical Levels), and BL (Brightness Level) metadata. It checks that reserved bytes are zero but does not return an error if they are not, merely logging a warning instead.

4. **dvcC_Write()**: Writes the Dolby Vision configuration to a bitstream, encoding version major, minor, profile, level, and flags into the appropriate fields of the box. It also writes five reserved bytes which are set to zero in this implementation. This function is only compiled if `GPAC_DISABLE_ISOM_WRITE` is not defined, indicating it should be used for writing Dolby Vision configuration boxes during ISO media file creation.

5. **dvcC_Size()**: Estimates the size of the box by adding 24 to its current size estimate, which includes space for the version major, minor, profile, level, and flags fields plus five reserved bytes set to zero. This function is only compiled if `GPAC_DISABLE_ISOM_WRITE` is not defined.

These functions are part of a larger library that likely handles ISO base media file format (MP4) containers, including support for various codecs like Dolby Vision. The presence or absence of write-related functions (`dvcC_Write`, `dvcC_Size`) can be controlled via build configuration flags to optimize the library size and performance without writing capabilities in certain contexts, such as when dealing with read-only media streams where no write operations are necessary.