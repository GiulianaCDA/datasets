
This implementation of the `WriteYUVImage` function is similar to the previous one, but it uses a different approach to write the image data. Instead of using a separate `chroma_image`, this version creates an YUV-encoded image directly from the input image and then writes its pixel values to the output file in three planes (Y, U, and V).

The main difference between this implementation and the previous one is that it uses the `WriteBlob()` function to write the pixel values of the Y, U, and V planes directly to the output file. This approach allows the image data to be written in a more efficient way, as it avoids the need for an intermediate buffer and the associated memory allocation and deallocation.

The implementation also includes some additional error checking and progress reporting features, such as checking whether the output file was opened successfully, checking whether the input image is valid, and updating the progress meter while writing the image data to disk.