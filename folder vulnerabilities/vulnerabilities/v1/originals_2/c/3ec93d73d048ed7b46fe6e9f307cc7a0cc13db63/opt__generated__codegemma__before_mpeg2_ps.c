// Create an instance of the MPEG2PS stream
mpeg2ps_t *ps = mpeg2ps_create();

// Load the MPEG2PS stream from a file
mpeg2ps_open(ps, "input.mp2");

// Seek to a video frame with a timestamp of 10 seconds
mpeg2ps_seek_video_frame(ps, 0, 10000000);
