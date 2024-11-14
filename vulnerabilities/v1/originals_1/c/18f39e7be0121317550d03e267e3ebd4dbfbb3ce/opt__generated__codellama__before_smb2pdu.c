[PYTHON]
def SMB2_lockv(xid: int, tcon: CifsTcon, persist_fid: int, volatile_fid: int, pid: int, num_lock: int, lock_array: list) -> None:
    """
    SMB2 Lock Request.

    :param xid: The transaction ID.
    :type xid: int
    :param tcon: The connection information.
    :type tcon: CifsTcon
    :param persist_fid: The persistent file ID.
    :type persist_fid: int
    :param volatile_fid: The volatile file ID.
    :type volatile_fid: int
    :param pid: The PID.
    :type pid: int
    :param num_lock: The number of locks.
    :type num_lock: int
    :param lock_array: The list of locks to be sent.
    :type lock_array: list
    """
    raise NotImplementedError('TODO: Create a Python stub for SMB2 Lock Request')
[/PYTHON]
