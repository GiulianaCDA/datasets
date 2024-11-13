[PYTHON]
import logging
import os
import re
from xml.etree import ElementTree
from xml.etree.ElementTree import ParseError

from svn import core, repos, fs, ra, diff, swap


class SVNRepository(object):
    """
    Provides an abstraction over the svn C library and its Python bindings.
    """

    def __init__(self, url, username=None, password=None):
        self.url = url
        self.username = username
        self.password = password

        # Initialize the core module with our desired error handlers
        core.svn_error_clear()
        core.svn_error_set_handlers(core._handle_error, core._report_warning)

    def __del__(self):
        # Clean up any open resources on deletion
        core.svn_pool_destroy(self.pool)

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        if not isinstance(value, str):
            raise ValueError("Invalid URL type: %s" % type(value))
        self._url = value

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        if not isinstance(value, str):
            raise ValueError("Invalid username type: %s" % type(value))
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        if not isinstance(value, str):
            raise ValueError("Invalid password type: %s" % type(value))
        self._password = value

    def _get_username_and_password(self):
        return (self.username, self.password)

    def checkout(self, path=None, revision=None):
        """
        Checkout the repository to the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Checkout the repository at the given revision or HEAD if no revision is specified
            repos.svn_checkout2(ctx, path, self._get_username_and_password(),
                               revision, 0, False, True, None)
        except Exception as e:
            logging.error("Error checking out repository: %s" % e)
            raise

    def update(self, path=None, revision=None):
        """
        Update the specified path or cwd if no path is specified to a specific revision.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Update the working copy at the specified path or cwd to a specific revision
            repos.svn_update2(path, revision, 0, False, True, None)
        except Exception as e:
            logging.error("Error updating repository: %s" % e)
            raise

    def get_url(self, path=None):
        """
        Returns the URL of the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            return fs.svn_fs_get_file_url(path, self.pool)
        except Exception as e:
            logging.error("Error retrieving URL from path: %s" % e)
            raise

    def get_revision(self, path=None):
        """
        Returns the revision of the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            return fs.svn_fs_get_revision_number(path, self.pool)
        except Exception as e:
            logging.error("Error retrieving revision from path: %s" % e)
            raise

    def get_logs(self, path=None, start_revision=0, end_revision=None):
        """
        Returns a list of logs for the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Get the logs for the specified path or cwd from the given start revision to HEAD
            return ra._get_logs(ctx, path, start_revision, end_revision, self.pool)
        except Exception as e:
            logging.error("Error retrieving logs from repository: %s" % e)
            raise

    def get_file_status(self, path=None):
        """
        Returns the status of a file at the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Get the status of the file at the specified path or cwd from HEAD
            return fs.svn_fs_check_path(ctx, path, self.pool)
        except Exception as e:
            logging.error("Error retrieving status from path: %s" % e)
            raise

    def diff(self, old_path=None, new_path=None):
        """
        Returns the difference between two files at the specified paths or cwd if no paths are specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not old_path:
            old_path = os.getcwd()

        if not new_path:
            new_path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Get the diff between the two files at the specified paths or cwd if no paths are specified
            return ra._get_diff(ctx, old_path, new_path, self.pool)
        except Exception as e:
            logging.error("Error retrieving diff from repository: %s" % e)
            raise

    def diff_tree(self, old_path=None, new_path=None):
        """
        Returns the difference between two trees at the specified paths or cwd if no paths are specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not old_path:
            old_path = os.getcwd()

        if not new_path:
            new_path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Get the diff between the two trees at the specified paths or cwd if no paths are specified
            return ra._get_diff(ctx, old_path, new_path, self.pool)
        except Exception as e:
            logging.error("Error retrieving diff from repository: %s" % e)
            raise

    def get_file_contents(self, path=None):
        """
        Returns the contents of a file at the specified path or cwd if no path is specified.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Get the contents of the file at the specified path or cwd if no path is specified
            return ra._get_file_contents(ctx, path, self.pool)
        except Exception as e:
            logging.error("Error retrieving contents from path: %s" % e)
            raise

    def add(self, path=None):
        """
        Adds a file or directory at the specified path to the repository.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not path:
            path = os.getcwd()

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Add the file or directory at the specified path or cwd if no path is specified
            return ra._add(ctx, path, self.pool)
        except Exception as e:
            logging.error("Error adding item to repository: %s" % e)
            raise

    def commit(self, message=None):
        """
        Commits all changes made in the working copy to the repository.
        """
        # Initialize a new pool that will be used for all Subversion operations
        self.pool = core.svn_pool_create(None)

        # Create a credential object from our username and password
        creds = repos.svn_auth_open([])
        core.svn_auth_set_credentials(creds, self.url, self.username, self.password, None)

        if not message:
            raise Exception("A commit message is required.")

        # Create a new repository context for the given URL and username
        ctx = repos.svn_repos_open3(self.url, creds, None, self.pool)
        if not ctx:
            raise Exception("Unable to open repository")

        try:
            # Commit all changes made in the working copy to the repository with the specified commit message
            return ra._commit(ctx, message, self.pool)
        except Exception as e:
            logging.error("Error committing changes to repository: %s" % e)
            raise
