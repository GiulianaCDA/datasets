 
[PYTHON]
def mci_soap_fault_login_failed():
    return soap_fault('Client', '', 'Access denied.')

def mci_soap_fault_access_denied(p_user_id, p_detail=''):
    t_user_name = user_get_name(p_user_id)
    t_reason = f'Access denied for user {t_user_name}.'
    if not is_blank(p_detail):
        t_reason += f' Reason: {p_detail}.'
    return soap_fault('Client', '', t_reason)
[/PYTHON]
