 This code is a part of a larger program that reads and writes DWG (Drawing Interchange File) files, which are used by AutoCAD to store vector graphics data. The function `dwg_write_object` is responsible for writing an object's data to a DWG file.

   The function takes several arguments: a pointer to the DWG structure `dwg`, a pointer to the object `_obj`, and a boolean value `recursive` that indicates whether to write nested objects recursively.

   The function first checks if the object is valid and not already written to the file. It then calculates the size of the object's data and allocates memory for it. If necessary, it writes the object's control data (such as its type and handle) to the file.

   The function then enters a loop that reads each piece of extended data (xdata) associated with the object and writes it to the file in binary format using bitwise operations. The loop iterates over all xdata elements, handling different types of data such as integers, floating-point numbers, strings, and handles.

   If there is an error during the writing process, the function sets an appropriate error code and returns it. Otherwise, it returns 0 to indicate success.