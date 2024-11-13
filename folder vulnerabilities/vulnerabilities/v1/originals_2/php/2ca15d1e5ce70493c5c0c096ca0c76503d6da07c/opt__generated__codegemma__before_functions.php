$directory = '/path/to/directory';
$exclude = array('/exclude/directory1', '/exclude/directory2');
$max_execution_time = 30; // Maximum execution time in seconds

$total_size = recurse_dirsize($directory, $exclude, $max_execution_time);

if ($total_size === null) {
    // Time limit exceeded
} else {
    // Directory size in bytes
    echo 'Total size: ' . $total_size;
}
