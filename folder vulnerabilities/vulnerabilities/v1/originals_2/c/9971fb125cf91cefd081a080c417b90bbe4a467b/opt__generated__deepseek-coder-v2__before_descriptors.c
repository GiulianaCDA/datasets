 These functions are part of a library that handles parsing and writing configuration for audio formats such as AC-3 (used in the first example) and Opus (used in the second example). Let's break down each function to understand their purpose and functionality.

### Function 1: `gf_odf_ac3_config_parse`
This function parses an AC-3 configuration from a bitstream (`dsi`) of length `dsi_len`. The function uses the helper function `gf_odf_ac3_config_parse_bs` to do the actual parsing. If the parsed configuration includes an EC3 extension (determined by `is_ec3`), it reads additional data from the bitstream:
- Reads 7 bits, which is likely unused but required in AC-3 or EC3 formats.
- Reads a single bit indicating the complexity index type of the EC3 extension.
- Reads an 8-bit value for the complexity index type.

### Function 2: `gf_odf_opus_cfg_parse`
This function parses an Opus configuration from a bitstream (`dsi`) of length `dsi_len`. The function uses the helper function `gf_odf_opus_cfg_parse_bs` to do the actual parsing. It initializes a new `GF_OpusConfig` structure and fills it with data read from the bitstream.

### Function 3: `gf_odf_opus_cfg_write_bs`
This function writes an Opus configuration into a bitstream (`bs`). The Opus configuration includes various parameters such as version, output channel count, pre-skip, input sample rate, and more, which are written in sequence to the bitstream.

### Function 4: `gf_odf_opus_cfg_write`
This function writes an Opus configuration from a given `GF_OpusConfig` structure into a dynamically allocated byte array (`data`) and updates the size of this array (`size`). It uses a temporary bitstream to write the data, then retrieves the content from the bitstream.

### Error Handling
All functions check if the input parameters (like `cfg` or `dsi`) are valid pointers before proceeding with their operations. If any parameter is null, they return `GF_BAD_PARAM`, indicating an invalid argument error.