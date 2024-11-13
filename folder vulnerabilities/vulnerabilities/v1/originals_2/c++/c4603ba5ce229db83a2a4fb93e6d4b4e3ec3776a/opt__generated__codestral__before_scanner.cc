#include <stdio.h>
#include <algorithm>
#include "src/util/c99_stdint.h"
#include <limits>
#include <string.h>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace re2c {
    // ... rest of the code remains unchanged ...

    bool Scanner::read(size_t need) {
        if (eof || need == 0) return false;

        size_t bytesRead = ::read(files.back()->fd, lim, need);
        if (bytesRead <= 0) {
            eof = lim;
            memset(lim, 0, YYMAXFILL);
            lim += YYMAXFILL;
            return false;
        }

        lim += bytesRead;
        return true;
    }

    void Scanner::pop_finished_files() {
        while (files.size() > 1 && files.back()->eof) {
            close(files.back()->fd);
            delete files.back();
            files.pop_back();
        }
    }

    void Scanner::open(const std::string& filename) {
#ifdef _WIN32
        int fd = ::open(filename.c_str(), O_RDONLY);
#else
        int fd = ::open(filename.c_str(), O_RDONLY | O_NOFOLLOW);
#endif
        if (fd < 0) throw std::runtime_error("Failed to open file");
        files.emplace_back(new InputFile{fd, ENDPOS, ENDPOS});
    }
} // namespace re2c
