require 'securerandom'
require 'fileutils'

class CamaleonCmsLocalUploader
  # ... other methods here ...

  def fetch_file(file_name)
    sanitized_file_name = File.basename(file_name)
    file_path = File.join(@root_folder, sanitized_file_name)

    if File.exist?(file_path) && File.readable?(file_path)
      # Perform safe operations with the file here
    else
      # Handle error condition here
    end
  end

  def add_file(uploaded_io_or_file_path, key)
    allowed_extensions = ['txt', 'png', 'jpg']
    file_extension = File.extname(key).downcase[1..-1]

    if allowed_extensions.include?(file_extension)
      # Perform safe operations to save the file here
    else
      # Handle error condition for invalid file type here
    end
  end

  def add_folder(key)
    sanitized_key = File.basename(key)
    folder_path = File.join(@root_folder, sanitized_key)

    unless Dir.exist?(folder_path)
      FileUtils.mkdir_p(folder_path)
    end

    # Perform safe operations with the folder here
  end

  def delete_folder(key)
    sanitized_key = File.basename(key)
    folder_path = File.join(@root_folder, sanitized_key)

    if Dir.exist?(folder_path)
      FileUtils.rm_rf(folder_path)
    else
      # Handle error condition for non-existent folder here
    end
  end

  def delete_file(key)
    sanitized_key = File.basename(key)
    file_path = File.join(@root_folder, sanitized_key)

    if File.exist?(file_path)
      FileUtils.rm(file_path)
    else
      # Handle error condition for non-existent file here
    end
  end
end
