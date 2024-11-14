<?php
include_once('./_common.php');

if( !isset($g5['faq_table']) || !isset($g5['faq_master_table']) ){
    die('<meta charset="utf-8">관리자 모드에서 게시판관리->FAQ관리를 먼저 확인해 주세요.');
}

$faq_master_list = array();
$sql = "SELECT * FROM {$g5['faq_master_table']} ORDER BY fm_order, fm_id";
$result = sql_query($sql);
while ($row=sql_fetch_array($result)) {
    $key = $row['fm_id'];
    if (!isset($fm_id)) $fm_id = $key;
    $faq_master_list[$key] = $row;
}

if (isset($_GET['fm_id']) && is_numeric($_GET['fm_id'])) {
    $fm_id = $_GET['fm_id'];
    if(isset($faq_master_list[$fm_id])){
        $fm = $faq_master_list[$fm_id];
        $g5['title'] = $fm['fm_subject'];
    } else {
        alert('Invalid FAQ master ID.');
    }
} else {
    alert('Missing or invalid FAQ master ID.');
}

$qstr = 'fm_id='.$fm_id;
if ($stx) $qstr .= '&amp;stx='.urlencode($stx);

$skin_file = './skin/board/'.$board['bo_skin'];
if(preg_match('/^[a-zA-Z0-9_-]+$/', $board['bo_skin']) && file_exists($skin_file.'.skin.php')) {
    include_once($skin_file.'.skin.php');
} else {
    echo '<p>'.str_replace(G5_PATH.'/', '', $skin_file).'이 존재하지 않습니다.</p>';
}

include_once('./_tail.php');
