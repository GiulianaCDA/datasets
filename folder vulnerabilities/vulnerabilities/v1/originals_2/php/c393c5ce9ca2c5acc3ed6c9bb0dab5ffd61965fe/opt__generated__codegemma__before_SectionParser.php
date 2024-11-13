// Get parsed information from file content
$parsed = CmsParser::parse($fileContent);

// Get line numbers where each section starts
$offsets = CmsParser::parseOffset($fileContent);
