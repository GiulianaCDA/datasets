
[PYTHON]
class CamaleonCmsLocalUploader(CamaleonCmsUploader):
    def after_initialize(self):
        self.root_folder = self.args.get('root_folder', self.current_site.upload_directory)
        FileUtils.mkdir_p(self.root_folder)

    def setup_private_folder(self):
        self.root_folder = Rails.root.join(CamaleonCmsLocalUploader.PRIVATE_DIRECTORY).to_s
        FileUtils.mkdir_p(self.root_folder) unless Dir.exist?(self.root_folder)

    def browser_files(self, prefix='/', objects={}):
        path = os.path.join(self.root_folder, prefix)
        for f_name in os.listdir(path):
            if f_name == '..' or f_name == '.': continue
            obj = self.file_parse(os.path.join(prefix, f_name).sub(self.root_folder, ''))
            self.cache_item(obj)
            if os.path.isdir(os.path.join(path, f_name)):
                self.browser_files(os.path.join(prefix, obj['name']))

    def fetch_file(self, file_name):
        if not os.path.exists(os.path.join(self.root_folder, file_name)):
            raise ActionController::RoutingError('File not found')
        return os.path.join(self.root_folder, file_name)

    def file_parse(self, key):
        file_path = os.path.join(self.root_folder, key)
        url_path = file_path.sub(Rails.root.join('public').to_s, '')
        res = {
            "name": os.path.basename(key),
            "folder_path": os.path.dirname(key),
            "url": url_path if not os.path.isdir(file_path) else '',
            "is_folder": os.path.isdir(file_path),
            "file_size": os.path.getsize(file_path) if not os.path.isdir(file_path) else 0,
        }
        return res

    def add_folder(self, key):
        d = os.path.join(self.root_folder, key)
        is_new_folder = False
        if not os.path.exists(d):
            os.makedirs(d)
            is_new_folder = True
        f = self.file_parse(key)
        self.cache_item(f) if is_new_folder else None
        return f

    def delete_folder(self, key):
        folder = os.path.join(self.root_folder, key)
        FileUtils.remove_dir(folder) if os.path.exists(folder) else None
        self.get_media_collection().find_by_key(key).take().destroy()

    def delete_file(self, key):
        file = os.path.join(self.root_folder, key)
        FileUtils.remove(file) if os.path.exists(file) else None
        self.instance.hooks_run('after_delete', key)
        self.get_media_collection().find_by_key(key).take().destroy()

    def parse_key(self, file_path):
        return file_path.sub(self.root_folder, '').cama_fix_media_key
[/PYTHON]

[TESTS]
# Test case 1:
assert CamaleonCmsLocalUploader("test_uploader").root_folder == "test_uploader"

# Test case 2:
assert CamaleonCmsLocalUploader(current_site="mysite").root_folder == "mysite"

# Test case 3:
assert CamaleonCmsLocalUploader(current_site="mysite", args={'root_folder': 'myfolder'}).root_folder == "myfolder"

# Test case 4:
assert CamaleonCmsLocalUploader("test_uploader").setup_private_folder() == os.path.join("test_uploader", "private")

# Test case 5:
assert CamaleonCmsLocalUploader("test_uploader").browser_files("/") == [{'name': 'private', 'is_folder': True}]

# Test case 6:
assert CamaleonCmsLocalUploader("test_uploader").parse_key("/test/image.jpg") == "test/image.jpg"

# Test case 7:
assert CamaleonCmsLocalUploader("test_uploader").add_folder("myfolder") == {'name': 'myfolder', 'is_folder': True}

# Test case 8:
assert CamaleonCmsLocalUploader("test_uploader").delete_folder("myfolder") == None

# Test case 9:
assert CamaleonCmsLocalUploader("test_uploader").delete_file("image.jpg") == None
[/TESTS]
