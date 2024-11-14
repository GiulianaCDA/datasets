[PYTHON]
import os
from cgi import FieldStorage
from smarty import Smarty

class AdminUploader:
    def __init__(self, smarty):
        self.smarty = smarty
        
    def upload(self):
        uploaded_files = []
        max_file_size = 10 * 1024 * 1024 # 10 MB
        blacklist_extensions = ['.php', '.html', '.htm']
        
        form = FieldStorage()
        files = form.getlist('files[]')
        
        for file in files:
            name = file.filename
            content = file.file.read()
            
            # Check if the uploaded file is valid
            if not name or not content or len(content) > max_file_size:
                continue
            
            # Get the extension of the uploaded file
            ext = os.path.splitext(name)[1]
            
            # Check if the uploaded file is in the blacklist
            if ext.lower() in blacklist_extensions:
                continue
            
            # Save the uploaded file to the server
            with open('uploads/' + name, 'wb') as f:
                f.write(content)
            
            uploaded_files.append(name)
        
        if uploaded_files:
            self.smarty.assign('success', 1)
            self.smarty.assign('uploaded_files', uploaded_files)
    
    def download(self):
        form = FieldStorage()
        filename = form.getvalue('filename')
        
        if not filename:
            return -1
        
        # Check if the file exists and is safe to open
        filepath = 'uploads/' + filename
        if not os.path.isfile(filepath) or os.access(filepath, os.R_OK):
            return -1
        
        self.smarty.assign('filename', filename)
        
        return 1
[/PYTHON]
