 
[PYTHON]
import os

def get_directories(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def get_files(directory):
    return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]

def search(query, directory):
    directories = get_directories(directory)
    files = get_files(directory)
    results = []
    for filename in files:
        if query in filename:
            results.append((directory, filename))
    for subdirectory in directories:
        results += search(query, os.path.join(directory, subdirectory))
    return results

if __name__ == "__main__":
    root_dir = input("Enter the root directory you would like to start from: ")
    query = input("Enter your query (regex can be used): ")
    results = search(query, root_dir)
    print("Results:")
    for directory, filename in results:
        print(f"{directory}/{filename}")
[/PYTHON]
 
[RUBY]
def get_directories(directory)
  directories = []
  Dir.glob(File.join(directory, "*")) do |path|
    if File.directory?(path)
      directories << path
    end
  end
  return directories
end

def get_files(directory)
  files = []
  Dir.glob(File.join(directory, "**", "*.*")) do |path|
    if File.file?(path)
      files << path
    end
  end
  return files
end

def search(query, directory)
  results = []
  directories = get_directories(directory)
  files = get_files(directory)
  for filename in files do
    if filename.match(query)
      results << File.join(directory, filename)
    end
  end
  directories.each do |subdirectory|
    results += search(query, subdirectory)
  end
  return results
end

if __FILE__ == $0
  root_dir = gets.chomp
  query = gets.chomp
  puts "Results:"
  search(query, root_dir).each do |path|
    puts path
  end
end
[/RUBY]
