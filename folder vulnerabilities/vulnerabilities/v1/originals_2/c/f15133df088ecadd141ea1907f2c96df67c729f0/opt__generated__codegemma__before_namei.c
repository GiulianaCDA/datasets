// Read the link contents of a file
int res = generic_readlink(dentry, buffer, buflen);

// Create a symlink to another file
int res = page_symlink(inode, "target_file", strlen("target_file"));
