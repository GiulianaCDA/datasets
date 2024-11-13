Image image;

// Load an image with 8BIM and EXIF profiles

MagickBooleanType status = SyncImageProfiles(&image);

if (status == MagickTrue) {
  // Profiles are synchronized successfully
} else {
  // Error occurred during profile synchronization
}
