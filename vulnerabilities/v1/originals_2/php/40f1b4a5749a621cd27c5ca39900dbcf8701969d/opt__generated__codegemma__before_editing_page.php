// Include the gp_uploaded class
includeFile('admin/admin_uploaded.php');

// Create an instance of the class
$uploaded = new gp_uploaded();

// Display the form for creating a new directory
$uploaded->NewDirForm();
