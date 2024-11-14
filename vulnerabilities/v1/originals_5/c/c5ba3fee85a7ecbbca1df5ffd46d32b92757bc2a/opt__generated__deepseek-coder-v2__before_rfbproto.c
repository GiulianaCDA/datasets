#include "your_header_file_with_definitions.h"  // Include all necessary definitions and macros

int main() {
    rfbPixelFormat format;
    uint8_t data[4] = {0x12, 0x34, 0x56, 0x78};
    uint8_t pixelData[4];
    
    // Example usage of GET_PIXEL32 macro
    GET_PIXEL32(format, data);
    
    // Print information about the pixel format
    PrintPixelFormat(&format);
    
    return 0;
}
