  $parsedContent = CodeParser::parse($content);
  echo $parsedContent['settings']; // Outputs settings section if present
  echo $parsedContent['code'];     // Outputs PHP code section if present
  echo $parsedContent['markup'];   // Outputs markup section if present
  