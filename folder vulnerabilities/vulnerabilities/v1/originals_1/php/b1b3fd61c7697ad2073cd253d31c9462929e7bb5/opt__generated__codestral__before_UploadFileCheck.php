<?php
if(!defined('sugarEntry') || !sugarEntry) die('Not A Valid Entry Point');
require_once('include/JSON.php');
require_once('include/entryPoint.php');
require_once 'include/upload_file.php';
global $sugar_config;
$supportedExtensions = array('jpg', 'png', 'jpeg');
$json = getJSONobj();
$rmdir=true;
$returnArray = array();
if(isset($_REQUEST['forQuotes']) && is_string($_REQUEST['forQuotes'])) {
    $returnArray['forQuotes'] = json_decode(html_entity_decode($_REQUEST['forQuotes'])) ? "quotes" : "company";
} else {
    $returnArray['error'] = 'Invalid forQuotes parameter';
    echo $json->encode($returnArray);
    sugar_cleanup();
    exit();
}
$upload_ok = false;
$upload_path = 'tmp_logo_' . $returnArray['forQuotes'] . '_upload';
if(isset($_FILES['file_1'])){
    $upload = new UploadFile('file_1');
    if($upload->confirm_upload()) {
        $extension = pathinfo($upload->get_stored_file_name(), PATHINFO_EXTENSION);
        if (!in_array(strtolower($extension), $supportedExtensions)) {
            $returnArray['error'] = 'Unsupported file type';
            echo $json->encode($returnArray);
            sugar_cleanup();
            exit();
        }
        $upload_dir  = 'upload://' . $upload_path;
        UploadStream::ensureDir($upload_dir);
        $file_name = $upload_dir."/".$upload->get_stored_file_name();
        if($upload->final_move($file_name)) {
            $upload_ok = true;
        }
    }
} else {
    $returnArray['error'] = 'No file uploaded';
    echo $json->encode($returnArray);
    sugar_cleanup();
    exit();
}
if(!$upload_ok) {
    $returnArray['data']='not_recognize';
    echo $json->encode($returnArray);
    sugar_cleanup();
    exit();
}
if(file_exists($file_name) && is_file($file_name)) {
    $encoded_file_name = rawurlencode($upload->get_stored_file_name());
    $returnArray['path'] = $upload_path . '/' . $encoded_file_name;
    $returnArray['url']= 'cache/images/'.$encoded_file_name;
    if(!verify_uploaded_image($file_name, $returnArray['forQuotes'] == 'quotes')) {
        $returnArray['data']='other';
        $returnArray['path'] = '';
    } else {
        $img_size = getimagesize($file_name);
        if ($img_size === false) {
            $returnArray['error'] = 'Failed to read image size';
            echo $json->encode($returnArray);
            sugar_cleanup();
            exit();
        }
        $test=$img_size[0]/$img_size[1];
        if (($test>10 || $test<1) && $returnArray['forQuotes'] == 'company'){
            $rmdir=false;
            $returnArray['data']='size';
        } elseif (($test>20 || $test<3)&& $returnArray['forQuotes'] == 'quotes') {
            $rmdir=false;
            $returnArray['data']='size';
        } else {
            copy($file_name, sugar_cached('images/'.$upload->get_stored_file_name()));
            $returnArray['data']='ok';
        }
    }
}else{
    $returnArray['data']='file_error';
}
echo $json->encode($returnArray);
sugar_cleanup();
exit();
