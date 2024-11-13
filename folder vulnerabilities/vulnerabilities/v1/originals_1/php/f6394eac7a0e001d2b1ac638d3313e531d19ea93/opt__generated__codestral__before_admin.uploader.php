 The code you've provided is a function that handles file uploads in PHP. It performs several checks to ensure the security and validity of the uploaded files, and then moves them to their appropriate directory. Here's a breakdown of what it does:

1. It first retrieves the uploaded files from the $_FILES array.
2. For each file, it extracts its name, temporary location, size, and error status.
3. It checks if there was an error during the upload. If so, it assigns -1 to the success variable.
4. The function then checks the file extension against a blacklist of forbidden extensions. This is done in two steps: first, it checks the last extension (e.g., "jpg" in "file.php.jpg"), and if that's not forbidden, it checks the second-to-last extension. If either check fails, it assigns -1 to the success variable.
5. If the file hasn't been marked as forbidden yet, the function uses the finfo_open function (available since PHP 5.3) to determine the MIME type of the file. If this is "text/x-php" or "text/html", it assigns -1 to the success variable.
6. The function then checks if the file has an image extension ("jpg", "jpeg", "gif", "png"). If so, it sets the destination directory to IMAGES_DIR. Otherwise, it doesn't change the directory.
7. It sanitizes the filename by removing any invalid characters and appending the original extension.
8. The function moves the uploaded file from its temporary location to the target directory, and sets the appropriate permissions.
9. If the move was successful, it adds the filename to the $uploaded_files array. Otherwise, it assigns -1 to the success variable.
10. Finally, it assigns the success variable to either 1 (if all files were uploaded successfully) or -1 (if there was an error with any file), and stores the $uploaded_files array in a session variable named 'admin_uploader_files'. It then returns 1, indicating that the function has finished executing.