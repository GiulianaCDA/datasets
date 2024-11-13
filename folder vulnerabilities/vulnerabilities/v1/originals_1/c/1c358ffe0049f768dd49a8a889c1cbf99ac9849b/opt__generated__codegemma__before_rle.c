// Register the RLE image format
size_t result = RegisterRLEImage();

// Check if an image is in RLE format
if (IsRLE(image)) {
  // Read the RLE image
  Image *rleImage = ReadRLEImage(image);
  // Use the rleImage for further processing
}

// Unregister the RLE image format
UnregisterRLEImage();
