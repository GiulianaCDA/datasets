// Allocate an ffs device
struct ffs_data *ffs_dev = start_dev("device_name");

// Check if the device is ready
if (ffs_ready(ffs_dev)) {
    // Device is ready for use
}

// Release the ffs device
ffs_release_dev(ffs_dev);
