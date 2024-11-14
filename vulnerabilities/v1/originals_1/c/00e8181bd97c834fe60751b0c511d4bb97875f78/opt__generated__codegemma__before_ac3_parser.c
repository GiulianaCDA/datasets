// Initialize the parser context
AVCodecParserContext *parser = avcodec_parser_init();

// Parse the AC3 header from a byte buffer
uint8_t header[AC3_HEADER_SIZE];
int size = sizeof(header);
int err = ff_ac3_parse(&parser, header, size);

// Check for errors
if (err < 0) {
    // Handle error
}

// Extract header information
AC3HeaderInfo hdr;
av_ac3_parse_header(header, size, &hdr.bitstream_id, &hdr.frame_size);

// Use header information for further processing
