// Load a Java bytecode file
RBinJavaObj *bin = r_bin_java_load_bin(filename, file_size);

// Get the class name of the first class in the archive
const char *class_name = r_bin_java_get_class_name(bin);

// Extract the bytecode of the "main" method
ut8 *bytecode = r_bin_java_get_idx_bytes(bin, U(r_bin_java_get_cp_idx_with_name)(bin, "main", 4), NULL);
