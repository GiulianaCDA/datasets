```c++
#include <stdio.h>
#include <algorithm>
#include "src/util/c99_stdint.h"
#include <limits>
#include <string.h>

#include "src/msg/msg.h"
#include "src/parse/scanner.h"
#include "src/debug/debug.h"


namespace re2c {

const char *const Scanner::ENDPOS = (const char*) std::numeric_limits<uint64_t>::max();

Scanner::~Scanner()
{
    for (size_t i = files.size(); i --> 0; ) {
        delete files[i];
    }
}

size_t Scanner::get_input_index() const
{
    // Find index of the current input file: the one corresponding to
    // buffer fragment that contains cursor.
    size_t i = files.size();
    DASSERT(i > 0);
    for (;;) {
        --i;
        Input *in = files[i];
        if (i == 0 || (cur >= in->so && cur <= in->eo)) break;
    }
    return i;
}

bool Scanner::open(const std::string &filename, const std::string *parent)
{
    Input *in = new Input(msg.filenames.size());
    files.push_back(in);
    if (!in->open(filename, parent, globopts->incpaths)) {
        return false;
    }
    msg.filenames.push_back(in->escaped_name);
    return true;
}

bool Scanner::include(const std::string &filename)
{
    // get name of the current file (before unreading)
    DASSERT(!files.empty());
    const std::string &parent = files.back()->escaped_name;

    // unread buffer tail: we'll return to it later
    // In the buffer nested files go before outer files. In the file stack,
    // however, outer files go before nested files (nested are at the top).
    // We want to break from the unreading cycle early, therefore we go in
    // reverse order of file offsets in buffer and break as soon as we find
    // a file that hasn't been read yet.
    size_t i = files.size();
    DASSERT(i > 0);
    for (;;) {
        --i;
        Input *in = files[i];
        if (i == 0 || in->so != ENDPOS) break;
    }

    files.insert(files.begin() + i + 1, new Input(msg.filenames.size()));
    Input *new_file = files[i + 1];

    if (!new_file->open(filename, &parent, globopts->incpaths)) {
        return false;
    }

    return true;
}

void Scanner::shift_ptrs_and_fpos(ptrdiff_t offs)
{
    DASSERT(bot <= tok && tok <= lim);
    tok += offs;
    cur += offs;
    lim += offs;

    for (size_t i = 0; i < files.size(); ++i) {
        Input *in = files[i];
        in->so += offs;
        in->eo += offs;
    }
}

void Scanner::pop_finished_files()
{
    // Pop all files that have been fully processed (file upper bound
    // in buffer points before the first character of current lexeme),
    // except for the first (main) file which must always remain at the
    // bottom of the stack.
    size_t i = files.size();
    DASSERT(i > 0);
    for (;;) {
        --i;
        Input *in = files[i];
        if (i == 0 || in->eo >= tok) break;
        files.pop_back();
        delete in;
    }
}

bool Scanner::fill(size_t need)
{
    if (eof) return false;

    pop_finished_files();

    DASSERT(bot <= tok && tok <= lim);
    size_t free = static_cast<size_t>(tok - bot);
    size_t copy = static_cast<size_t>(lim - tok);

    if (free >= need) {
        memmove(bot, tok, copy);
        shift_ptrs_and_fpos(-static_cast<ptrdiff_t>(free));
    }
    else {
        BSIZE += std::max(BSIZE, need);
        char * buf = new char[BSIZE + YYMAXFILL];
        if (!buf) fatal("out of memory");

        memmove(buf, tok, copy);
        shift_ptrs_and_fpos(buf - bot);
        delete [] bot;
        bot = buf;

        free = BSIZE - copy;
    }

    if (!read(free)) {
        eof = lim;
        memset(lim, 0, YYMAXFILL);
        lim += YYMAXFILL;
    }

    return true;
}

} // namespace re2c
