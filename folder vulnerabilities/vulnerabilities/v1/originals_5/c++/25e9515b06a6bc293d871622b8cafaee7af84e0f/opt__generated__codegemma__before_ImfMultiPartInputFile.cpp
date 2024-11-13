// Create an instance of MultiPartInputFile
MultiPartInputFile inputFile("input.exr");

// Get the number of parts
int numParts = inputFile.parts();

// Check if part 0 is complete
bool part0Complete = inputFile.partComplete(0);

// Access part 0
InputPartData* part0 = inputFile.getPart(0);
