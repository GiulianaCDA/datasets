Image image;
// Load image...

// Sync image profiles
MagickBooleanType status = SyncImageProfiles(&image);

if (status == MagickTrue) {
  // Profiles synchronized successfully
} else {
  // Error syncing profiles
}
