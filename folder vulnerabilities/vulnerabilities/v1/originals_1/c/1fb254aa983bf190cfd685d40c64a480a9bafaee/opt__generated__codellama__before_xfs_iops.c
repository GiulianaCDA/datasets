
[PYTHON]
class Inode:
    def __init__(self, mode):
        self.mode = mode

    def set_ops(self):
        if self.mode == S_IFREG:
            self.op = xfs_inode_operations
            self.fop = xfs_file_operations
            self.mapping.a_ops = xfs_address_space_operations
        elif self.mode == S_IFDIR:
            if xfs_sb_version_hasasciici(&XFS_M(inode->i_sb)->m_sb):
                self.op = xfs_dir_ci_inode_operations
            else:
                self.op = xfs_dir_inode_operations
            self.fop = xfs_dir_file_operations
        elif self.mode == S_IFLNK:
            if self.i_df.if_flags & XFS_IFINLINE:
                self.op = xfs_inline_symlink_inode_operations
            else:
                self.op = xfs_symlink_inode_operations
        else:
            self.op = xfs_inode_operations
            init_special_inode(self, self.mode, self.i_rdev)

[/PYTHON]
[TESTS]
# Test case 1:
assert Inode(S_IFREG).set_ops().op == xfs_inode_operations
# Test case 2:
assert Inode(S_IFDIR).set_ops().op == xfs_dir_inode_operations
# Test case 3:
assert Inode(S_IFLNK).set_ops().op == xfs_symlink_inode_operations
[/TESTS]
