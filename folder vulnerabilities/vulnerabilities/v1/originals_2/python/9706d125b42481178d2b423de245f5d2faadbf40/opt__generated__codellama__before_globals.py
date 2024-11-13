
[PYTHON]
import copy
import datetime
import logging
import os
import pickle
import portalocker
import uuid
from types import MethodType

from gluon import current, globalvars
from gluon.storage import Storage
from gluon.settings import global_settings
from gluon.utils import web2py_uuid


logger = logging.getLogger("web2py.session")


class Session(Storage):
    """
    Implements a session object, using the Storage class as base.

    Overrides methods to use cookies and/or database sessions based on settings in global_settings.

    Attributes:
        _new (bool): whether the session is new (just created) or not
        _forget (bool): whether to forget the session after the request is served
        _unsaved (dict): dictionary of unsaved values
        _updated (dict): dictionary of updated values
        _deleted (list): list of deleted keys
        _changed (bool): whether any changes have been made to the session
    """

    def __init__(self, request, response=None):
        self.request = request
        self._new = True
        self._forget = False
        self._unsaved = {}
        self._updated = {}
        self._deleted = []
        self._changed = False
        if response is None:
            response = current.response
        if hasattr(request, "session_id"):
            self.reconstruct(request.session_id)
        else:
            self.recreate()
        if not hasattr(self, "s3") or self.s3 is None:
            self.s3 = Storage()

    def forget(self):
        """
        Forgets the session, so that a new one will be created on next use.
        """
        self._forget = True

    def recreate(self):
        """
        Creates a new session, but keeps the existing session data if any.
        """
        self._new = False
        if hasattr(self, "data"):
            for key in list(self.data.keys()):
                self[key] = self.data[key]
        self._unsaved = {}
        self._updated = {}
        self._deleted = []
        self._changed = False

    def reconstruct(self, session_id=None, expired=False):
        """
        Rebuilds the session from a stored session. If `session_id` is not specified, uses the current request's
        session ID. If `expired` is True, it ignores any exception thrown while restoring the session (i.e., treats
        expired sessions as if they were non-existent).

        Args:
            session_id: a string representing the session ID
            expired: boolean indicating whether to ignore exceptions when restoring an expired session
        """
        try:
            self._new = False
            if not session_id:
                request = current.request
                if hasattr(request, "session_id"):
                    session_id = request.session_id
            if not session_id:
                return
            data = self._get_storage(session_id)
            if not isinstance(data, dict):
                return
            self.update(data)
        except Exception as e:
            logger.error("Error while reconstructing session {}: {}".format(session_id, e))
            if not expired:
                raise e

    def _get_storage(self, session_id):
        """
        Retrieves the data stored in the specified session, as a dictionary.

        Args:
            session_id: a string representing the session ID

        Returns:
            a dictionary with the data stored in the session
        """
        if not hasattr(self.request, "application"):
            return None
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        if not table:
            return None
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select(limitby=(0, 1)).first()
        if not record:
            return None
        data = pickle.loads(record.session_data)
        if hasattr(self, "s3"):
            s3 = self.s3
        else:
            s3 = Storage()
        if "s3" in data:
            for key in list(data["s3"].keys()):
                s3[key] = data["s3"][key]
                del data["s3"][key]
        return data

    def _save_storage(self, session_id=None, expired=False):
        """
        Saves the session data to the database. If `session_id` is not specified, uses the current request's
        session ID. If `expired` is True, it ignores any exception thrown while saving the session (i.e., treats
        expired sessions as if they were non-existent).

        Args:
            session_id: a string representing the session ID
            expired: boolean indicating whether to ignore exceptions when saving an expired session
        """
        try:
            if not hasattr(self.request, "application"):
                return
            app = self.request.application
            db = globalvars.storage(app) or current.db
            table_name = "%s_session" % app
            table = db[table_name]
            if not table:
                return
            data = Storage()
            for key in list(self.data.keys()):
                data[key] = self.data[key]
            # remove deleted keys
            for key in list(self._deleted):
                if key in data:
                    del data[key]
            session_data = pickle.dumps(data)
            record_id, unique_key = session_id.split(":")
            query = (table.id == int(record_id)) & \
                    (table.unique_key == unique_key)
            record = db(query).select(limitby=(0, 1)).first()
            if not record:
                return
            # update existing session record
            if self._changed and len(self._unsaved) > 0:
                for key in list(self._unsaved.keys()):
                    setattr(record, key, data[key])
                db(query).update(session_data=session_data,
                                 timestamp=datetime.datetime.now())
            # add new keys
            if self._changed and len(self._updated) > 0:
                for key in list(self._updated.keys()):
                    setattr(record, key, data[key])
                db(query).update(session_data=session_data,
                                 timestamp=datetime.datetime.now())
            # delete keys
            if self._changed and len(self._deleted) > 0:
                for key in list(self._deleted):
                    delattr(record, key)
                db(query).update(session_data=session_data,
                                 timestamp=datetime.datetime.now())
        except Exception as e:
            logger.error("Error while saving session {}: {}".format(session_id, e))
            if not expired:
                raise e

    def _delete_storage(self):
        """Deletes the current session data from the database."""
        try:
            if hasattr(self.request, "application"):
                app = self.request.application
                db = globalvars.storage(app) or current.db
                table_name = "%s_session" % app
                table = db[table_name]
                if not table:
                    return
                session_id = self.request.session_id
                if not session_id:
                    return
                record_id, unique_key = session_id.split(":")
                query = (table.id == int(record_id)) & \
                        (table.unique_key == unique_key)
                db(query).delete()
        except Exception as e:
            logger.error("Error while deleting session {}: {}".format(session_id, e))

    def _get_cookie(self):
        """Retrieves the current session ID from the cookie."""
        if hasattr(self.request, "cookies"):
            return self.request.cookies.get("session_id")
        else:
            return None

    def _save_cookie(self, session_id=None):
        """Saves the current session ID to the cookie."""
        if not hasattr(self.request, "response"):
            return
        response = self.request.response
        if session_id is None:
            session_id = self.request.session_id
        if session_id is not None:
            response.set_cookie("session_id", value=str(session_id), path="/")

    def _delete_cookie(self):
        """Deletes the current session ID from the cookie."""
        if hasattr(self.request, "response"):
            response = self.request.response
            response.delete_cookie("session_id", path="/")

    def _get_session_id(self):
        """Retrieves the current session ID."""
        session_id = None
        cookie = self._get_cookie()
        if cookie:
            try:
                session_id = int(cookie)
            except ValueError as e:
                logger.error("Invalid session ID in cookie: {}".format(e))
        if not session_id:
            session_id = self._generate_session_id()
        return session_id

    def _save(self):
        """Saves the current session data to the database."""
        session_id = self._get_session_id()
        if not session_id:
            return
        self._save_storage(session_id)
        self._save_cookie(session_id)

    def _delete(self):
        """Deletes the current session data from the database and cookie."""
        session_id = self._get_cookie()
        if not session_id:
            return
        self._delete_storage()
        self._delete_cookie()

    def _generate_session_id(self):
        """Generates a new random session ID."""
        while True:
            session_id = randint(100000, 999999)
            if not self.exists(session_id):
                return session_id

    def _is_expired(self, timestamp):
        """Checks whether the current session is expired."""
        timeout = self._get_timeout()
        if timeout == 0:
            return False
        time_now = datetime.datetime.now()
        time_diff = time_now - timestamp
        return time_diff.total_seconds() > timeout

    def _get_timeout(self):
        """Retrieves the current session timeout."""
        if hasattr(self.request, "application"):
            app = self.request.application
            config = globalvars.config(app)
            return int(config["session"]["timeout"])
        else:
            return 0

    def _init_data(self):
        """Initializes the current session data from the database."""
        if hasattr(self.request, "application"):
            app = self.request.application
            db = globalvars.storage(app) or current.db
            table_name = "%s_session" % app
            table = db[table_name]
            session_id = self._get_cookie()
            if not session_id:
                return None
            record_id, unique_key = session_id.split(":")
            query = (table.id == int(record_id)) & \
                    (table.unique_key == unique_key)
            record = db(query).select().first()
            if not record:
                return None
            data = pickle.loads(record.session_data)
            self.request.session_id = session_id
            self.request.application = app
            self.exists = True
            return data
        else:
            return None

    def _update_data(self, key, value):
        """Updates the current session data."""
        if not hasattr(self.request, "application"):
            return
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        session_id = self._get_cookie()
        if not session_id:
            return None
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        self._save_storage(session_id, key, value)

    def _delete_data(self, key):
        """Deletes the current session data."""
        if not hasattr(self.request, "application"):
            return
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        session_id = self._get_cookie()
        if not session_id:
            return None
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        self._delete_storage(session_id, key)

    def _get_data(self, key):
        """Retrieves the current session data."""
        if not hasattr(self.request, "application"):
            return None
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        session_id = self._get_cookie()
        if not session_id:
            return None
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        data = pickle.loads(record.session_data)
        return data[key]

    def _set_data(self, key, value):
        """Sets the current session data."""
        if not hasattr(self.request, "application"):
            return
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        session_id = self._get_cookie()
        if not session_id:
            return None
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        data = pickle.loads(record.session_data)
        data[key] = value
        self._save_storage(session_id, key, value)

    def _save_storage(self, session_id, key, value):
        """Saves the current session data to the database."""
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        data = pickle.loads(record.session_data)
        data[key] = value
        session_data = pickle.dumps(data)
        db(table.id == int(record_id)).update(session_data=session_data)
        self.request.application = app
        self.exists = True

    def _delete_storage(self, session_id, key):
        """Deletes the current session data from the database."""
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        record_id, unique_key = session_id.split(":")
        query = (table.id == int(record_id)) & \
                (table.unique_key == unique_key)
        record = db(query).select().first()
        if not record:
            return None
        data = pickle.loads(record.session_data)
        del data[key]
        session_data = pickle.dumps(data)
        db(table.id == int(record_id)).update(session_data=session_data)
        self.request.application = app
        self.exists = True

    def exists(self):
        """Checks if a session currently exists."""
        return self.exists

    def save(self, data):
        """Saves the given data to the session."""
        for key in data:
            self._set_data(key, data[key])

    def get(self, key):
        """Retrieves the value of a given key from the session."""
        return self._get_data(key)

    def delete(self, key):
        """Deletes the value of a given key from the session."""
        self._delete_data(key)

    def start(self):
        """Starts a new session if none currently exists."""
        if not self.exists:
            self.create()

    def create(self):
        """Creates a new session."""
        # generate a unique key for the session
        key = self._generate_unique_key()
        # save the session to the database
        self._save_session(key)
        # set the cookie with the session id
        self._set_cookie(key)
        # set the exists flag to true
        self.exists = True

    def destroy(self):
        """Destroys the current session."""
        if self.exists:
            self._destroy_session()
            self.exists = False

    def _generate_unique_key(self):
        """Generates a unique key for the session."""
        return hashlib.md5(str(uuid.uuid4()).encode('utf-8')).hexdigest()

    def _save_session(self, key):
        """Saves the given key to the database."""
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        session_data = pickle.dumps({})
        record = table.insert(id=key, unique_key=key, session_data=session_data)
        self.request.application = app
        self.exists = True

    def _set_cookie(self, key):
        """Sets the cookie with the given session id."""
        # set the cookie name
        cookie_name = "session_id"
        # set the domain and path of the cookie
        domain = ".%s" % self.request.host[1]
        path = "/"
        # set the expiration time of the cookie (in seconds)
        max_age = 3600 * 24 * 7
        # set the secure flag to True if you want to enable SSL
        secure = False
        # set the http-only flag to True if you want to prevent javascript access
        http_only = True
        # set the cookie with the session id
        self.response.set_cookie(cookie_name, key, domain=domain, path=path, max_age=max_age, secure=secure, http_only=http_only)

    def _destroy_session(self):
        """Destroys the current session."""
        app = self.request.application
        db = globalvars.storage(app) or current.db
        table_name = "%s_session" % app
        table = db[table_name]
        key = self._get_cookie()
        if key:
            query = (table.id == int(key))
            record = db(query).select().first()
            if record:
                # delete the cookie with the session id
                self.response.delete_cookie("session_id")
                # delete the record from the database
                db(query).delete()
                # set the exists flag to false
                self.exists = False

    def _get_cookie(self):
        """Retrieves the value of the cookie with the given name."""
        # get the cookie name
        cookie_name = "session_id"
        return self.request.cookies.get(cookie_name)