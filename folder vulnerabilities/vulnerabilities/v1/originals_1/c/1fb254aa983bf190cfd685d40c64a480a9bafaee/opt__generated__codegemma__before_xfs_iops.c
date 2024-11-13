**Function start_inode()**

- Takes a pointer to an xfs_inode struct as input.
- Creates a new inode from the xfs_inode struct.
- Initializes the inode with the following fields:
    - Inode number
    - State set to I_NEW
    - Added to the inode_sb_list
    - Hashed for writeback code
    - User and group IDs
    - File size
    - File attributes
    - Lock class set for directory or non-directory inode
    - GFP mask for page cache allocations set to exclude GFP_FS
    - ACL and xattr status based on the presence of an attribute fork

**Function xfs_setup_iops()**

- Takes a pointer to an xfs_inode struct as input.
- Sets the inode operations and file operations based on the inode type:
    - Regular file: xfs_inode_operations, xfs_file_operations, address space operations
    - Directory: xfs_dir_inode_operations, xfs_dir_file_operations
    - Symbolic link: xfs_symlink_inode_operations
    - Other special file: xfs_inode_operations, init_special_inode

**Purpose**

The `start_inode()` function initializes a new inode from an xfs_inode struct, while the `xfs_setup_iops()` function sets the inode operations and file operations based on the inode type. These functions are used in the inode creation process for the XFS filesystem.