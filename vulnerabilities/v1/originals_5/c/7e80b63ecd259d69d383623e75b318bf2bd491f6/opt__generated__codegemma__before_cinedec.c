int main() {
    AVFormatContext *avctx;
    AVPacket pkt;

    // Open the Cine file
    avctx = avformat_alloc_context3();
    avformat_open_input(&avctx, "filename.cine", NULL, NULL);

    // Read the header
    avformat_read_header(avctx, NULL);

    // Extract video frames
    while (av_read_frame(avctx, &pkt) >= 0) {
        // Process video frame here
        av_packet_unref(&pkt);
    }

    // Close the Cine file
    avformat_close_input(&avctx);

    return 0;
}
