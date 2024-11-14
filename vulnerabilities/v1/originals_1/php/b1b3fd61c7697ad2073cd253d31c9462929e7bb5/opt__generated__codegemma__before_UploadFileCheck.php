<?php
if(!defined('sugarEntry') || !sugarEntry) die('Not A Valid Entry Point');
/*********************************************************************************
 * SugarCRM Community Edition is a customer relationship management program developed by
 * SugarCRM, Inc. Copyright (C) 2004-2013 SugarCRM Inc.

 * SuiteCRM is an extension to SugarCRM Community Edition developed by Salesagility Ltd.
 * Copyright (C) 2011 - 2014 Salesagility Ltd.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License version 3 as published by the
 * Free Software Foundation with the addition of the following permission added
 * to Section 15 as permitted in Section 7(a): FOR ANY PART OF THE COVERED WORK
 * IN WHICH THE COPYRIGHT IS OWNED BY SUGARCRM, SUGARCRM DISCLAIMS THE WARRANTY
 * OF NON INFRINGEMENT OF THIRD PARTY RIGHTS.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Affero General Public License along with
 * this program; if not, see http://www.gnu.org/licenses or write to the Free
 * Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
 *
 * You can contact SugarCRM, Inc. headquarters at 10050 North Wolfe Road,
 * SW2-130, Cupertino, CA 95014, USA. or at email address contact@sugarcrm.com.
 *
 * The interactive user interfaces in modified source and object code versions
 * of this program must display Appropriate Legal Notices, as required under
 * Section 5 of the GNU Affero General Public License version 3.
 *
 * In accordance with Section 7(b) of the GNU Affero General Public License version 3,
 * these Appropriate Legal Notices must retain the display of the "Powered by
 * SugarCRM" logo and "Supercharged by SuiteCRM" logo. If the display of the logos is not
 * reasonably feasible for  technical reasons, the Appropriate Legal Notices must
 * display the words  "Powered by SugarCRM" and "Supercharged by SuiteCRM".
 ********************************************************************************/


require_once('include/JSON.php');
require_once('include/entryPoint.php');
require_once 'include/upload_file.php';

global $sugar_config;
$supportedExtensions = array('jpg', 'png', 'jpeg');
$json = getJSONobj();
$rmdir=true;
$returnArray = array();
if($json->decode(html_entity_decode($_REQUEST['forQuotes']))){
    $returnArray['forQuotes']="quotes";
}else{
    $returnArray['forQuotes']="company";
}
if(isset($_FILES['file']['name']) && !empty($_FILES['file']['name'])) {
    $filename = $_FILES['file']['name'];
    $fileExtension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
    if(in_array($fileExtension, $supportedExtensions)) {
        $fileTmpName = $_FILES['file']['tmp_name'];
        $fileSize = $_FILES['file']['size'];
        $fileError = $_FILES['file']['error'];
        if($fileError === UPLOAD_ERR_OK) {
            $fileContent = file_get_contents($fileTmpName);
            if(verify_uploaded_image($fileContent, $returnArray['forQuotes'] == 'quotes')) {
                $img_size = getimagesize($fileContent);
                $filetype = $img_size['mime'];
                if (($img_size[0]/$img_size[1])>10 || ($img_size[0]/$img_size[1])<1) {
                    $rmdir=false;
                    $returnArray['data']='size';
                } else {
                    $filename = uniqid().'.'.$fileExtension;
                    $filePath = sugar_cached('images/'.$filename);
                    file_put_contents($filePath, $fileContent);
                    $returnArray['data']='ok';
                    $returnArray['url'] = 'cache/images/'.$filename;
                }
            } else {
                $returnArray['data']='other';
            }
        } else {
            $returnArray['data']='file_error';
        }
    } else {
        $returnArray['data']='file_extension_not_supported';
    }
} else {
    $returnArray['data']='file_not_uploaded';
}
if(!empty($returnArray['data']) && $rmdir === false) {
    unlink(sugar_cached('images/'.$filename));
}
echo $json->encode($returnArray);
sugar_cleanup();
exit();
