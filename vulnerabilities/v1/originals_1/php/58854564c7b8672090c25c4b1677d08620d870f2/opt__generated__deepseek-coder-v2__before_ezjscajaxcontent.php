 The provided PHP code snippet is a part of a larger class, likely belonging to an eZ Publish-like CMS system. This snippet contains several methods designed for encoding data into XML format and handling various attributes related to images and other content types. Here's a breakdown of each method and its functionality:

1. **extractAttributesFromContentObject**:
   - Extracts attributes from a content object, including their type, identifier, and content. It also handles image attributes separately, checking if they need preprocessing based on predefined sizes.
   - **Parameters**: 
     - `$contentObject`: The content object from which to extract attributes.
     - `$params`: An array containing parameters like `imageDataTypes`, `imageSizes`, and `imagePreGenerateSizes`.
   - **Returns**: An associative array with extracted data, including image attributes if applicable.

2. **xmlEncode**:
   - Converts a multi-level array or hash into a valid XML string.
   - **Parameters**: 
     - `$hash`: The array or hash to be encoded.
     - `$childName`: The name of the child element (default is 'child').
   - **Returns**: A string representing the XML data.

3. **xmlWrite**:
   - A recursive function used by `xmlEncode` to build the XML structure. It handles both elements and nested arrays.
   - **Parameters**: 
     - `$xml`: An instance of `XMLWriter`.
     - `$hash`: The array or hash to be encoded.
     - `$childName`: The name of the child element (default is 'child').

4. **formatLocaleDate**:
   - Formats a given timestamp according to the current locale settings, supporting various date-time formats.
   - **Parameters**: 
     - `$timestamp`: A Unix timestamp.
     - `$type`: The format type as described in the method's documentation.
   - **Returns**: A formatted date string based on the specified format.

Each method uses PHP's `XMLWriter` class for generating XML and handles nested arrays and specific data types (like images) gracefully. This snippet is particularly useful for systems where content attributes need to be serialized in a way that can be easily parsed by other applications or stored efficiently, such as eZ Publish, which this code appears to be adapted from.

This code snippet does not include error handling for file operations or missing data types, and it assumes certain configurations (like the presence of specific configuration settings in `$params`) are present at runtime. However, it provides a robust framework for dealing with content objects and their attributes in a CMS environment where such functionality is crucial for API outputs, backups, or general interoperability between different components.