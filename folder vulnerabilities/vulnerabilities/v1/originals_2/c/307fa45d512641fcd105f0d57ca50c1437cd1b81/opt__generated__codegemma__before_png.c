Image image;
ImageInfo image_info;

// Initialize image and image information
// ...

// Write the image in MNG format
MagickBooleanType status = WriteMNGImage(&image_info, &image);

// Check for errors
if (status == MagickFalse) {
  // Handle error
}
