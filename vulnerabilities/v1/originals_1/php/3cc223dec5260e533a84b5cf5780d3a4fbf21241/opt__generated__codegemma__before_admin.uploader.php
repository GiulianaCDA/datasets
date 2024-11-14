/**
 * uploader.php
 *
 * Uploads files to the server.
 *
 * @author your_name
 * @version 1.0
 */

<?php

// Include necessary files
require_once('functions.php');

// Create a new instance of the uploader class
$uploader = new Uploader();

// Handle the upload request
$success = $uploader->upload();

// Assign the success variable to the template
$smarty->assign('success', $success);

// Display the template
$smarty->display('uploader.tpl');

?>
