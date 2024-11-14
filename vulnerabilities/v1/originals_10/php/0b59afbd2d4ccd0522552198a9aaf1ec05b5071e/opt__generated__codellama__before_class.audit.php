
 

 

[PYTHON]
class AuditLog:
    def __init__(self, timestamp=None):
        self.timestamp = timestamp or datetime.datetime.now()

    @staticmethod
    def log(staff_id, user_id, event_id, object_type, object_id, data, ip=None):
        audit_log = AuditLog()
        audit_log.staff_id = staff_id
        audit_log.user_id = user_id
        audit_log.event_id = event_id
        audit_log.object_type = object_type
        audit_log.object_id = object_id
        audit_log.data = data
        if ip:
            audit_log.ip = ip
        db.session.add(audit_log)
        db.session.commit()

    @staticmethod
    def log_page_hit(user_id, page):
        AuditLog.log(None, user_id, 1, 'page', page, None)

    @staticmethod
    def log_login(user_id, ip=None):
        AuditLog.log(None, user_id, 2, 'user', user_id, None, ip)

    @staticmethod
    def log_logout(user_id, ip=None):
        AuditLog.log(None, user_id, 3, 'user', user_id, None, ip)

    @staticmethod
    def log_event(staff_id, event_id, object_type, object_id, data, ip=None):
        AuditLog.log(staff_id, None, event_id, object_type, object_id, data, ip)

    @staticmethod
    def log_message(staff_id, message):
        AuditLog.log(staff_id, None, 4, 'message', staff_id, message)

    @staticmethod
    def log_note(staff_id, note):
        AuditLog.log(staff_id, None, 5, 'note', staff_id, note)

class User:
    id = 1234
    name = "John Doe"

class Staff:
    id = 5678
    name = "Jane Smith"

# Example usage:
AuditLog.log_event(Staff.id, 1, 'user', User.id, {'action': 'login'})

# Output:
# INSERT INTO audit (staff_id, event_id, object_type, object_id, data) VALUES (5678, 1, 'user', 1234, '{"action": "login"}')
[/PYTHON]
