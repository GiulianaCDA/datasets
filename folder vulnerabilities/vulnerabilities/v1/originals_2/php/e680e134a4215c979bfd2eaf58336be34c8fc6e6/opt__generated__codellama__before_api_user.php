use MicroweberPackages\Backup\Exporters\JsonExport;

api_expose('user_social_login');

api_expose('logout');

api_expose('user_register', [
    'email' => ['required', 'unique:users,email'],
    'username' => ['required', 'unique:users,username'],
    'password' => ['required', 'confirmed']
]);

api_expose('social_login_process');

api_expose('user_reset_password_from_link', [
    'email' => ['required', 'exists:users,email'],
    'verification_key' => ['required', 'exists:users,verification_key']
]);

api_expose_admin('user_make_logged');

api_expose('user_login', [
    'email' => ['required', 'email', 'exists:users,email'],
    'password' => ['required', 'confirmed']
]);

api_expose('is_logged', function () {
    $user = Auth::user();
    if ($user) {
        return [
            'id' => $user->id,
            'username' => $user->username,
            'email' => $user->email,
            'is_admin' => $user->is_admin
        ];
    } else {
        return false;
    }
});

api_expose('user_send_forgot_password', [
    'email' => ['required', 'exists:users,email']
]);

api_expose_admin('users/register_email_send_test', function () {

    try {
        $user = Auth::user();
        mw()->option_manager->override('users', 'register_email_enabled', true);
        $send =  mw()->user_manager->register_email_send($user->id);
        if ($send) {
            return [
                'message' => 'Register email sent successfully'
            ];
        } else {
            return [
                'message' => 'Failed to send register email'
            ];
        }
    } catch (Exception $e) {
        echo 'Exception: ', $e->getMessage(), "\n";
    }
});

api_expose_admin('users/register_email_send', function ($params) {
    try {
        $user = Auth::user();
        mw()->option_manager->override('users', 'register_email_enabled', true);
        $send =  mw()->user_manager->register_email_send($user->id, $params['email']);
        if ($send) {
            return [
                'message' => 'Register email sent successfully'
            ];
        } else {
            return [
                'message' => 'Failed to send register email'
            ];
        }
    } catch (Exception $e) {
        echo 'Exception: ', $e->getMessage(), "\n";
    }
});
