struct magic_set ms = { ... }; // Initialize magic set

// Perform magic command
int result = magic(&ms, "myfile.txt");

if (result == -1) {
    // Error handling
} else {
    // Print magic result
    printf("%s\n", ms.buffer);
}
