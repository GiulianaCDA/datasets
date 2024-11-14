 import sys
from argparse import Namespace
from .connection import KeycloakConnection, KeycloakAdminConnection
import json

def register_client(options):
    # Create the payload for the POST request to the realm
    client_data = {
        'name': options.client_name,
        'protocol': 'openid-connect',
        'redirectUris': [f'http://{options.host}:8080/auth/realms/{options.realm_name}/broker/{options.provider_alias}/endpoint'],
        'webOrigins': ['*'],
    }

    # Set the Authorization header with the admin access token
    headers = {
        "Authorization": f"Bearer {options.admin_access_token}"
    }

    # Create a connection to Keycloak and send the POST request
    conn = KeycloakAdminConnection(options)
    response = conn.raw_post(f'/admin/realms/{options.realm_name}/clients', json=client_data, headers=headers)

    # Check if the client was registered successfully
    if response.status_code == 201:
        print('Client registered successfully')
    else:
        print(f'Failed to register client (HTTP {response.status_code}): {response.text}')

# Assuming options is an instance of argparse.Namespace containing the required attributes and methods
options = Namespace()
options.server_url = 'http://localhost:8080/auth'
options.realm_name = 'master'  # or your realm name
options.admin_username = 'admin'
options.admin_password = 'admin'
options.client_name = 'my-client'
options.host = 'localhost'
options.provider_alias = 'my-provider'

# Obtain the admin access token (assuming you have a function to do this)
# options.admin_access_token = get_admin_access_token(options)  # Replace with your actual function

register_client(options)