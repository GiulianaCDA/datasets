<?php
// +-----------------------------------------------------------------------+
// | Piwigo - a PHP based photo gallery                                    |
// +-----------------------------------------------------------------------+
// | Copyright(C) 2008-2016 Piwigo Team                  http://piwigo.org |
// +-----------------------------------------------------------------------+

if (!defined("PHPWG_ROOT_PATH")) {
    die ("Hacking attempt!");
}

include_once(PHPWG_ROOT_PATH . 'admin/include/functions.php');
check_status(ACCESS_ADMINISTRATOR);

$sections = array_filter(explode('/', $_GET['section']), function($value) {
    return !empty($value) && $value !== '..';
});

if (count($sections) < 2) {
    die('Invalid plugin URL');
}

$plugin_id = $sections[0];

// Added strict character validation for plugin identifier
if (!preg_match('/^[\w-]+$/', $plugin_id)) {
    die('Invalid plugin identifier');
}

// Check if the plugin is active to prevent directory traversal and accessing non-active plugins.
if (!isset($pwg_loaded_plugins[$plugin_id])) {
    die('Invalid URL - plugin ' . $plugin_id . ' not active');
}

$filename = PHPWG_PLUGINS_PATH . implode('/', $sections);
// Check if the file exists and is a regular file to prevent directory traversal and accessing non-file resources.
if (is_file($filename)) {
    include_once($filename);
} else {
    die('Missing file ' . $filename);
}
?>
