  This is the implementation of a symlink using a pagecache. The `page_symlink` function creates a new symlink by writing the link contents to the page cache. The `page_readlink` function reads the symlink contents from the page cache and copies them into the user buffer. The `page_follow_link_light` function returns the symlink contents as a string.

The `page_getlink` function is used by both `page_readlink` and `page_follow_link_light` to retrieve the symlink contents from the page cache. It allocates a new page, reads the link contents into it, and then returns a pointer to the kmap of the allocated page.

The `pagecache_write_begin` function is used by both `page_symlink` and `page_getlink` to write the symlink contents to the page cache. It takes care of mapping the page into kernel space, so that it can be written directly from user space. The `pagecache_write_end` function is used to finish writing the link contents to the page cache.

The `page_put_link` function is used by both `generic_readlink` and `page_follow_link_light` to release the allocated page that was used to store the symlink contents in the page cache.