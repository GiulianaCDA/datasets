static struct task_struct *ecryptfs_kthread;

/**
 * ecryptfs_threadfn
 * @ignored: ignored
 *
 * The eCryptfs kernel thread that has the responsibility of getting
 * the lower file with RW permissions.
 *
 * Returns zero on success; non-zero otherwise
 */
static int ecryptfs_threadfn(void *ignored)
{
	set_freezable();
	while (1)  {
		struct ecryptfs_open_req *req;

		wait_event_freezable(
			ecryptfs_kthread_ctl.wait,
			!list_empty(&ecryptfs_kthread_ctl.req_list) ||
			ecryptfs_kthread_ctl.flags & ECRYPTFS_KTHREAD_ZOMBIE);

		if (ecryptfs_kthread_ctl.flags & ECRYPTFS_KTHREAD_ZOMBIE) {
			break;
		}

		req = list_entry(ecryptfs_kthread_ctl.req_list.next, struct ecryptfs_open_req, kthread_ctl_list);
		list_del(&req->kthread_ctl_list);

		req->lower_file = dentry_open(&req->path, req->path.mnt, req->path.dentry, req->path.flags, req->path.cred);

		if (!IS_ERR(req->lower_file)) {
			complete(&req->done);
		} else {
			printk(KERN_ERR "%s: Failed to open lower file; rc = [%ld]"
				   "\n", __func__, PTR_ERR(req->lower_file));
			complete(&req->done);
		}
	}

	return 0;
}

/**
 * ecryptfs_privileged_open
 * @lower_file: Result of dentry_open by root on lower dentry
 * @lower_dentry: Lower dentry for file to open
 * @lower_mnt: Lower vfsmount for file to open
 *
 * This function gets a r/w file opened againt the lower dentry.
 *
 * Returns zero on success; non-zero otherwise
 */
int ecryptfs_privileged_open(struct file **lower_file,
			     struct dentry *lower_dentry,
			     struct vfsmount *lower_mnt,
			     const struct cred *cred)
{
	struct ecryptfs_open_req req;
	int flags = O_LARGEFILE;
	int rc = 0;

	init_completion(&req.done);
	req.lower_file = lower_file;
	req.path.dentry = lower_dentry;
	req.path.mnt = lower_mnt;

	/* Corresponding dput() and mntput() are done when the
	 * lower file is fput() when all eCryptfs files for the inode are
	 * released. */
	flags |= IS_RDONLY(d_inode(lower_dentry)) ? O_RDONLY : O_RDWR;
	(*lower_file) = dentry_open(&req.path, flags, cred);
	if (!IS_ERR(*lower_file))
		goto out;
	if ((flags & O_ACCMODE) == O_RDONLY) {
		rc = PTR_ERR((*lower_file));
		goto out;
	}
	list_add_tail(&req.kthread_ctl_list, &ecryptfs_kthread_ctl.req_list);
	wake_up(&ecryptfs_kthread_ctl.wait);
	wait_for_completion(&req.done);
	if (IS_ERR(*lower_file))
		rc = PTR_ERR(*lower_file);
out:
	return rc;
}
