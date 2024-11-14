#include <iostream>
#include "your_bencode_parsing_library.h" // Assume this is where your library resides

int main() {
    std::string bencodedData = "d4:spami5:helloe"; // Example of a simple bencoded string and dictionary
    lazy_entry e;
    if (parse_bencode(bencodedData, e)) {
        std::cout << print_entry(e) << std::endl;
    } else {
        boost::system::error_code ec = make_error_code(bdecode_errors::unexpected_end_of_file);
        std::cerr << "Error: " << ec.message() << std::endl;
    }
    return 0;
}
