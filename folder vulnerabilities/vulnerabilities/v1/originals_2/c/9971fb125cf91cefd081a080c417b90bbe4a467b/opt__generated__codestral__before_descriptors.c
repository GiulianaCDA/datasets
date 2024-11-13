 This code is part of the GPAC open-source library. It provides functions for parsing and writing Opus and AC3/E-AC3 configuration data from a bitstream.

Here's a breakdown of each function:

1. `gf_odf_ac3_config_parse_bs` - This function parses the AC3 or E-AC3 configuration from a given bitstream and stores the result in a `GF_For_AC3Config` structure. It handles the standard fields, as well as any optional EC3 extension data that may be present.
2. `gf_odf_ac3_config_parse` - This function creates a new bitstream from the given DSI (decoder specific info) and calls `gf_odf_ac3_config_parse_bs` to parse the configuration.
3. `gf_odf_opus_cfg_parse_bs` - This function parses an Opus configuration from a given bitstream and stores the result in a `GF_OpusConfig` structure. It handles all fields defined by the Opus specification, including any optional channel mapping data that may be present.
4. `gf_odf_opus_cfg_parse` - This function creates a new bitstream from the given DSI and calls `gf_odf_opus_cfg_parse_bs` to parse the configuration.
5. `gf_odf_opus_cfg_write_bs` - This function writes an Opus configuration to a given bitstream from a `GF_OpusConfig` structure. It handles all fields defined by the Opus specification, including any optional channel mapping data that may be present.
6. `gf_odf_opus_cfg_write` - This function creates a new bitstream and calls `gf_odf_opus_cfg_write_bs` to write the configuration. It then returns the resulting DSI data and size in the output parameters.