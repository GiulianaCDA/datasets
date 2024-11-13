<?php
// ... (other code)

api_expose_admin('users/register_email_send_test', function () {
    // Add authorization checks to ensure only authenticated administrators can access this endpoint.
    if (!isAdminAuthenticated()) {
        return 'Unauthorized';
    }

    try {
        mw()->option_manager->override('users', 'register_email_enabled', true);
        $send =  mw()->user_manager->register_email_send();
        if ($send) {
            $user = Auth::user();

            return 'Email is send successfully to <b>'.$user->email.'</b>.';
        }
    } catch (Exception $e) {
        // Handle exceptions gracefully.
        return "Error: " . $e->getMessage();
    }
});

api_expose('users/search_authors', function ($params = false) {
    // ... (other code)

    if ($kw) {
        $all_users_search['keyword'] = db_escape($kw); // Sanitize input to prevent SQL injection.
        $all_users_search['search_in_fields'] = 'id,username,first_name,last_name,email';
    }

    // ... (other code)
});

api_expose('users/verify_email_link', function ($params) {
    if (isset($params['key'])) {
        try {
            $decoded = mw()->format->decrypt($params['key']);
            if ($decoded) {
                // ... (other code)
            } else {
                return 'Invalid key';
            }
        } catch (Exception $e) {
            // Handle exceptions gracefully.
            return 'Error: ' . $e->getMessage();
        }
    }
});
