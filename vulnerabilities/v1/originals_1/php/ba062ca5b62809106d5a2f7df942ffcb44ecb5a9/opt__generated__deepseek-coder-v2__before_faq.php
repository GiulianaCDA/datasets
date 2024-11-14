<?php
include_once('./_common.php');

// Ensure the necessary configuration variables are set
if (!isset($g5['faq_table']) || !isset($g5['faq_master_table'])) {
    die('<meta charset="utf-8">관리자 모드에서 게시판관리->FAQ관리를 먼저 확인해 주세요.');
}

// Initialize variables and sanitize inputs
$faq_master_list = array();
$fm_id = isset($_GET['fm_id']) ? intval($_GET['fm_id']) : null; // Ensure $fm_id is an integer if provided via GET
$stx = isset($_GET['stx']) ? htmlspecialchars(trim($_GET['stx']), ENT_QUOTES) : ''; // Sanitize and trim search query
$page = isset($_GET['page']) ? max(1, intval($_GET['page'])) : 1; // Ensure page is a valid integer greater than or equal to 1

// Fetch FAQ master list from the database
$sql = "SELECT * FROM {$g5['faq_master_table']} ORDER BY fm_order, fm_id";
$result = sql_query($sql);
while ($row = sql_fetch_array($result)) {
    $key = $row['fm_id'];
    $faq_master_list[$key] = $row;
}

// Prepare query string for URL
$qstr = '';
if (isset($fm_id) && $fm_id) {
    $qstr .= '&amp;fm_id=' . urlencode($fm_id); // Use urlencode to ensure the ID is safe in URLs
}

// Fetch the selected FAQ master record
$fm = isset($faq_master_list[$fm_id]) ? $faq_master_list[$fm_id] : null;
if (!$fm || !$fm['fm_id']) {
    alert('등록된 내용이 없습니다.');
}

// Set the title and other necessary variables
$g5['title'] = $fm['fm_subject'];

// Include necessary files for the page
include_once('./_head.php');

// Check if the skin file exists and include it if true
$skin_file = $faq_skin_path . '/list.skin.php';
if (is_file($skin_file)) {
    // Additional logic for admin links, image sources, etc.
    include_once($skin_file);
} else {
    echo '<p>' . str_replace(G5_PATH . '/', '', $skin_file) . '이 존재하지 않습니다.</p>';
}

include_once('./_tail.php');
?>
