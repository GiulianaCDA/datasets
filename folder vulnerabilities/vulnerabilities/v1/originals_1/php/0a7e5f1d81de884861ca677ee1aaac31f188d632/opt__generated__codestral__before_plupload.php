 // Your existing code...

if (isset($_FILES['upload'])) {
    $file = $_FILES['upload'];

    if ($file['error'] !== UPLOAD_ERR_OK) {
        throw new UploadException($file['error']);
    }

    $fileName = $file['name'];
    $filePath = mw()->user_manager->path() . DS . 'uploads' . DS . $fileName;

    if (!move_uploaded_file($file['tmp_name'], $filePath)) {
        throw new UploadException(UPLOAD_ERR_CANT_WRITE);
    }

    // Image resizing logic... (you might want to move this to a separate function)
    list($width, $height, $type, $attr) = getimagesize($filePath);

    if ($width > 0 && $height > 0) {
        // Your existing image resizing code...
    } else {
        @unlink($filePath);
        die('{"jsonrpc" : "2.0", "error" : {"code": 107, "message": "File is not an image"}, "id" : "id"}');
    }
}

// The rest of your code...