[PYTHON]
def get_netscape_format(cookie):
    # Set the expiration date for the cookie if it exists.
    if cookie['expires']:
        expires = int(cookie['expires'])
        if expires > 0:
            expires = strftime('%a, %d-%b-%Y %H:%M:%S GMT', gmtime(expires))
        else:
            expires = 'Session'
    else:
        expires = 'Session'

    # Set the domain attribute for the cookie if it exists.
    if cookie['domain']:
        domain = cookie['domain']
    else:
        domain = ''

    # Set the path attribute for the cookie if it exists.
    if cookie['path']:
        path = cookie['path']
    else:
        path = '/'

    # Set the secure attribute for the cookie if it exists.
    if cookie['secure']:
        secure = 'TRUE'
    else:
        secure = 'FALSE'

    # Set the httponly attribute for the cookie if it exists.
    if cookie['httponly']:
        httponly = 'TRUE'
    else:
        httponly = 'FALSE'

    return f'Set-Cookie3: {cookie["name"]}={cookie["value"]}; expires={expires}; path={path}; domain={domain}; secure={secure}; HttpOnly={httponly}'
[/PYTHON]
* This code is not tested and may contain syntax errors.
* This code uses Python syntax, but it should be easy to convert it to other languages like C or C++.