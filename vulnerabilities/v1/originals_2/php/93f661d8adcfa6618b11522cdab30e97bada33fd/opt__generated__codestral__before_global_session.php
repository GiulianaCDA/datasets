<?php
global $config, $refresh;

$allowed_scripts = ['graph_view.php', 'graph.php']; // Add allowed script names here

$script = basename($_SERVER['SCRIPT_NAME']);
if (in_array($script, $allowed_scripts)) {
    if (isset($_SESSION['custom']) && $_SESSION['custom'] === true) {
        $refreshIsLogout = 'true';
    } elseif (isset($_GET['action']) && $_GET['action'] == 'zoom') { // Add additional validation checks as needed
        $refreshIsLogout = 'true';
    } else {
        $refreshIsLogout = 'false';
    }
} elseif (strpos($script, 'plugins.php') !== false) { // Add more conditions based on your application's requirements
    $refreshIsLogout = 'false';
} else {
    die('Invalid script name'); // Or handle it appropriately based on your application's security policy
}

// Continue the rest of the code with proper input sanitization and output encoding/escaping.
?>
<script type='text/javascript'>
    var cactiVersion = <?php echo json_encode($config['cacti_version']); ?>;
    // ... continue the rest of the JavaScript code with proper security measures.
</script>
