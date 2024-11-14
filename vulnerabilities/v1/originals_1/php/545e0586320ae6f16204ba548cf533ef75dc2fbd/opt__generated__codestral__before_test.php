<?php
ini_set('max_execution_time', 300);

if (isset($_POST['url']) && filter_var($_POST['url'], FILTER_VALIDATE_URL) !== false) {
    $url = filter_var($_POST['url'], FILTER_SANITIZE_URL);
} else {
    $url = '';
}

if ($url) {
    // rest of the code...
}
?>
