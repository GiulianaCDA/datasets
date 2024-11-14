// Log a ticket creation event
$ticket = new Ticket();
$auditEvent = new AuditEvent();
$auditEvent->auditEvent(EVENT_TICKET_CREATE, $ticket, array('abbrev' => 'Ticket'));

// Log a staff login event
$staff = new Staff();
$auditEvent->auditEvent(EVENT_STAFF_LOGIN, $staff, array('abbrev' => 'Staff'));
