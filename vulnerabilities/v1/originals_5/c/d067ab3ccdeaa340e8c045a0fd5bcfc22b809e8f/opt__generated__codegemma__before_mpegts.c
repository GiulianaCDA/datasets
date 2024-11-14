// Assuming buffer contains an MPEG2TS stream
int result = resample_mpeg2ts(buffer, buffer_size);

if (result == GF_OK) {
    // Resampling successful
} else {
    // Error during resampling
}
