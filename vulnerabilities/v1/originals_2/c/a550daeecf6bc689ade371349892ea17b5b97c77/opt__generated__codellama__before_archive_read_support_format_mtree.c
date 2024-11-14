 
[PYTHON]
def readline(a, mtree, start, limit):
    total_size = 0
    find_off = 0
    t = __archive_read_ahead(a, 1, &bytes_read)
    if (t == NULL):
        return (0)
    elif (bytes_read < 0):
        return (ARCHIVE_FATAL)
    nl = memchr(t, '\n', bytes_read)
    # If we found '\n', trim the read to end exactly there.
    if (nl != NULL):
        bytes_read = ((const char *)nl) - ((const char *)t) + 1
    if (total_size + bytes_read + 1 > limit):
        archive_set_error(&a.archive, ARCHIVE_ERRNO_FILE_FORMAT, "Line too long")
        return (ARCHIVE_FATAL)
    if (archive_string_ensure(&mtree.line, total_size + bytes_read + 1) == NULL):
        archive_set_error(&a.archive, ENOMEM, "Can't allocate working buffer")
        return (ARCHIVE_FATAL)
    # Append new bytes to string.
    memcpy(mtree.line.s + total_size, t, bytes_read)
    __archive_read_consume(a, bytes_read)
    total_size += bytes_read
    mtree.line.s[total_size] = '\0'

    for u in mtree.line.s + find_off:
        if (u == '\n'):
            # Ends with unescaped newline.
            start = mtree.line.s
            return total_size
        elif (u == '#'):
            # Ends with comment sequence #...\n
            if (nl == NULL):
                # But we've not found the \n yet
                break
        elif (u == '\\'):
            if (u[1] == '\n'):
                # Trim escaped newline.
                total_size -= 2
                mtree.line.s[total_size] = '\0'
                break
            elif (u[1] != '\0'):
                # Skip the two-char escape sequence
                u += 1
        find_off = u - mtree.line.s
    return (ARCHIVE_FATAL)
[/PYTHON]
