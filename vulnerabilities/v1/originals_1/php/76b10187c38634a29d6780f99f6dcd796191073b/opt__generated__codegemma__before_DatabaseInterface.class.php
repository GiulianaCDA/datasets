// Create a database link object
$db = new PMALink();

// Execute a query
$result = $db->query("SELECT * FROM users");

// Get the field name of the first field
$fieldName = $db->fieldName($result, 0);

// Print the field name
echo $fieldName;
