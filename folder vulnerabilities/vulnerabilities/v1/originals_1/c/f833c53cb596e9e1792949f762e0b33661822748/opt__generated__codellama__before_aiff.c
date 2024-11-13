  Note that the `aiff_next_chunk_iterator` function is not used by the SF library. Instead, the `sf_read_chunk_iterator`, `sf_next_chunk_iterator`, and `sf_free_chunk_iterator` functions are used to iterate over the chunks in an AIFF file.

The `aiff_get_chunk_size` function returns the size of a chunk specified by an iterator, while the `aiff_get_chunk_data` function reads the data contained within a chunk into a buffer. The `aiff_set_chunk` function is not used by the SF library.

Note that the `aiff_read_chanmap` function is not necessary for the SF library to read AIFF files. Instead, the `sf_command` function can be used to extract channel map information from an AIFF file.