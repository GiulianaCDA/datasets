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

Scanner::~Scanner() {
    for (size_t i = files.size(); i --> 0; ) {
        delete files[i];
    }
}

size_t Scanner::get_input_index() const {
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

bool Scanner::open(const std::string &filename, const std::string *parent) {
    Input *in = new Input(msg.filenames.size());
    files.push_back(in);
    if (!in->open(filename, parent, globopts->incpaths)) {
        return false;
    }
    msg.filenames.push_back(in->escaped_name);
    return true;
}

bool Scanner::include(const std::string &filename) {
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

    // Open new file and push it on the stack
    return open(filename);
}

bool Scanner::read(size_t need) {
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

void Scanner::shift_ptrs_and_fpos(ptrdiff_t offs) {
    // shift buffer pointers
    shift_ptrs(offs);

    // shift file pointers
    for (size_t i = files.size(); i --> 0; ) {
        Input *in = files[i];
        if (in->so == ENDPOS && in->eo == ENDPOS) break;
        DASSERT(in->so != ENDPOS && in->eo != ENDPOS);
        in->so += offs;
        in->eo += offs;
    }
}

void Scanner::pop_finished_files() {
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

bool Scanner::fill(size_t need) {
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

void Scanner::shift_ptrs(ptrdiff_t offs) {
    tok += offs;
    yytext += offs;
    yyleng -= offs;
    for (size_t i = 0; i < files.size(); ++i) {
        Input *in = files[i];
        if (in->so > in->eo) {
            // This file has been pushed onto the stack, but it hasn't been filled yet.
            // So we need to update its pointers accordingly.
            in->so += offs;
            in->eo += offs;
        }
    }
}

void Scanner::fatal(const char *msg) {
    std::cerr << "Scanner fatal error: " << msg << "\n";
    exit(1);
}
