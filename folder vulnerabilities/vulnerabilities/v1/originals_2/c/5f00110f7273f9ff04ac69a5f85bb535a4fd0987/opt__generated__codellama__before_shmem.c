[PYTHON]
def shmem_file_setup(name, size, flags):
    dentry = d_alloc(root, &this)
    if (!path.dentry)
        goto put_memory;
    path.mnt = mntget(shm_mnt);

    error = -ENOSPC;
    inode = shmem_get_inode(root->d_sb, NULL, S_IFREG | S_IRWXUGO, 0, flags);
    if (!inode)
        goto put_dentry;

    d_instantiate(path.dentry, inode);
    inode->i_size = size;
    clear_nlink(inode); # It is unlinked
#ifndef CONFIG_MMU
    error = ramfs_nommu_expand_for_mapping(inode, size);
    if (error)
        goto put_dentry;
#endif

    error = -ENFILE;
    file = alloc_file(&path, FMODE_WRITE | FMODE_READ, &shmem_file_operations);
    if (!file)
        goto put_dentry;

    return file;

put_dentry:
    path_put(&path);
put_memory:
    shmem_unacct_size(flags, size);
    return ERR_PTR(error);
}

def shmem_zero_setup(vma):
    file = shmem_file_setup("dev/zero", vma.vm_end - vma.vm_start, vma.vm_flags)
    if (IS_ERR(file)):
        return PTR_ERR(file);

    if (vma.vm_file):
        fput(vma.vm_file);
    vma.vm_file = file;
    vma.vm_ops = &shmem_vm_ops;
    return 0;
[/PYTHON]
