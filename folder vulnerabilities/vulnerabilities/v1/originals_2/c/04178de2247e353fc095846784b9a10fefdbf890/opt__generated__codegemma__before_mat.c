// Load an image
MagickImage image;
// ...

// Write the image as a MATLAB file
status = WriteMATLABImage(&image, "matlab_image.mat");

// Check for errors
if (status == MagickFalse) {
  // Handle error
}
