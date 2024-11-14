  [PYTHON]
class SP:
    def __init__(self, config):
        self.config = config

    def login(self, request):
        # get the necessary information from the request
        sp_entity_id = request.get('sp_entity_id')
        idp_entity_id = request.get('idp_entity_id')
        relay_state = request.get('relay_state')

        # find the SP and IDP metadata
        sp_metadata = self.config.get_sp_metadata(sp_entity_id)
        idp_metadata = self.config.get_idp_metadata(idp_entity_id)

        # build the authentication request
        ar = sso_lib.build_auth_request(sp_metadata, idp_metadata)

        # sign the authentication request
        sign_alg = sp_metadata.get('signature_algorithm', 'http://www.w3.org/2001/04/xmldsig-more#rsa-sha256')
        digest_alg = sp_metadata.get('digest_algorithm', 'http://www.w3.org/2001/04/xmlenc#sha256')
        sso_lib.add_sign(sp_metadata, idp_metadata, ar, sign_alg, digest_alg)

        # set the destination URL for the authentication request
        consumer_url = idp_metadata.get('single_sign_on_service')['url']

        # send the authentication request to the IDP
        response = requests.post(consumer_url, data=ar.to_xml())

        # process the IDP's response
        if response.status_code == 200:
            idp_response = sso_lib.process_idp_response(sp_metadata, idp_metadata, response.content)

            # extract the necessary information from the IDP's response
            session_index = idp_response.get('session_index')
            name_id = idp_response.get('name_id')
            attributes = idp_response.get('attributes', {})

            # create a user object with the necessary information
            user = User(name_id, attributes)

            # return the user object to the SP
            return user
        else:
            raise Exception('IDP responded with an error')

    def logout(self, request):
        # get the necessary information from the request
        sp_entity_id = request.get('sp_entity_id')
        idp_entity_id = request.get('idp_entity_id')
        relay_state = request.get('relay_state')
        name_id = request.get('name_id')
        session_index = request.get('session_index')

        # find the SP and IDP metadata
        sp_metadata = self.config.get_sp_metadata(sp_entity_id)
        idp_metadata = self.config.get_idp_metadata(idp_entity_id)

        # build the logout request
        lr = sso_lib.build_logout_request(sp_metadata, idp_metadata, name_id, session_index, relay_state)

        # sign the logout request
        sign_alg = sp_metadata.get('signature_algorithm', 'http://www.w3.org/2001/04/xmldsig-more#rsa-sha256')
        digest_alg = sp_metadata.get('digest_algorithm', 'http://www.w3.org/2001/04/xmlenc#sha256')
        sso_lib.add_sign(sp_metadata, idp_metadata, lr, sign_alg, digest_alg)

        # set the destination URL for the logout request
        consumer_url = idp_metadata.get('single_logout_service')['url']

        # send the logout request to the IDP
        response = requests.post(consumer_url, data=lr.to_xml())

        # process the IDP's response
        if response.status_code == 200:
            idp_response = sso_lib.process_idp_response(sp_metadata, idp_metadata, response.content)

            # extract the necessary information from the IDP's response
            in_response_to = idp_response.get('in_response_to')
            status_code = idp_response.get('status_code')
            status_message = idp_response.get('status_message')

            # return the IDP's response to the SP
            return {'in_response_to': in_response_to, 'status_code': status_code, 'status_message': status_message}
        else:
            raise Exception('IDP responded with an error')
[/PYTHON]
[TESTS]
# Test case 1:
def test_build_auth_request():
    # setup
    sp_entity_id = 'https://sp.example.com'
    idp_entity_id = 'https://idp.example.com'
    relay_state = 'https://sp.example.com/relay-state'

    # expected result
    expected_destination = 'https://idp.example.com/sso/login'
    expected_issuer = 'https://sp.example.com'
    expected_service_provider = sp_entity_id
    expected_assertion_consumer_service_url = 'https://sp.example.com/acs'

    # test
    ar = sso_lib.build_auth_request(sp_entity_id, idp_entity_id)

    # verify
    assert ar.destination == expected_destination
    assert ar.issuer == expected_issuer
    assert ar.service_provider == expected_service_provider
    assert ar.assertion_consumer_service_url == expected_assertion_consumer_service_url

# Test case 2:
def test_build_logout_request():
    # setup
    sp_entity_id = 'https://sp.example.com'
    idp_entity_id = 'https://idp.example.com'
    relay_state = 'https://sp.example.com/relay-state'
    name_id = 'name-id'
    session_index = 'session-index'

    # expected result
    expected_destination = 'https://idp.example.com/sso/logout'
    expected_issuer = 'https://sp.example.com'
    expected_service_provider = sp_entity_id
    expected_name_id = name_id
    expected_session_index = session_index

    # test
    lr = sso_lib.build_logout_request(sp_entity_id, idp_entity_id, name_id, session_index)

    # verify
    assert lr.destination == expected_destination
    assert lr.issuer == expected_issuer
    assert lr.service_provider == expected_service_provider
    assert lr.name_id == expected_name_id
    assert lr.session_index == expected_session_index

# Test case 3:
def test_process_idp_response():
    # setup
    sp_entity_id = 'https://sp.example.com'
    idp_entity_id = 'https://idp.example.com'
    relay_state = 'https://sp.example.com/relay-state'

    ar = sso_lib.build_auth_request(sp_entity_id, idp_entity_id)
    response = '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"><saml:Issuer>https://idp.example.com</saml:Issuer><samlp:Status><samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success" /></samlp:Status></samlp:Response>'

    # expected result
    expected_in_response_to = ar.id
    expected_issuer = 'https://idp.example.com'
    expected_status_code = 'urn:oasis:names:tc:SAML:2.0:status:Success'

    # test
    ir = sso_lib.process_idp_response(sp_entity_id, idp_entity_id, response, relay_state)

    # verify
    assert ir.in_response_to == expected_in_response_to
    assert ir.issuer == expected_issuer
    assert ir.status_code == expected_status_code
[/TESTS]
