 /* This is the code for loading a MTM (MultiTracker Module) file in the OpenMPT library. Here's a brief explanation of what the code does:

1. The function `mtm_load` takes a `file_handle *fp`, a pointer to a `csf_t *song`, and an integer `lflags`. It initializes some variables for tracking progress, current sample rate, and the number of samples in the file.
2. The function reads the header data from the file and checks if it's a valid MTM file by comparing the magic number with the expected value.
3. It then reads the song data from the file, including information about the song title, message, number of channels, patterns, samples, and instrument offsets.
4. The function creates an array to store track data for each channel in the module, allocates memory for it, and unpacks the track data from the file into this array.
5. It then reads the pattern data from the file and stores it in the `csf_t` structure. If there are too many patterns, a warning is logged.
6. The function also reads the sample data from the file and stores it in the `csf_t` structure if the `LOAD_NOSAMPLES` flag is not set.
7. Finally, the function sets some additional flags for the song and checks if there are any TODO items related to the file format that need attention. It then returns `LOAD_SUCCESS` if the loading process was successful. */