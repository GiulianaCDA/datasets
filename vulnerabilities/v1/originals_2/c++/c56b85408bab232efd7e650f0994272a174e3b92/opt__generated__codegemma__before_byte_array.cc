// Create a byte array with a capacity of 1024 bytes
ByteArray array(0, 1024);

// Copy the entire byte array to an output stream
array.CopyTo(outputStream);

// Copy a specific portion of the byte array to an output stream
array.CopyTo(outputStream, offset, length);

// Read data from an input stream and copy it into the byte array
array.CopyFrom(inputStream, length);

// Read data from an input stream until the end of the stream and copy it into the byte array
array.CopyFrom(inputStream);
