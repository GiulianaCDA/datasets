class CamaleonCmsLocalUploader < CamaleonCmsUploader
  def after_initialize
    @root_folder = @args[:root_folder] || ensure_secure_directory(@current_site.upload_directory)
  end

  def setup_private_folder
    @root_folder = Rails.root.join(self.class::PRIVATE_DIRECTORY).to_s
    ensure_secure_directory(@root_folder) unless Dir.exist?(@root_folder)
  end

  private

  def ensure_secure_directory(path)
    # Ensure the directory is secure and does not allow traversal beyond it
    raise ArgumentError, "Invalid path" unless Pathname.new(path).relative?
    path
  end
end
