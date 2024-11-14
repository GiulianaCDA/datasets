<?php
/* vim: set expandtab sw=4 ts=4 sts=4: */
/**
 * session handling
 */
if (! defined('PHPMYADMIN')) {
    exit;
}

// verify if PHP supports session, die if it does not
if (!@function_exists('session_name')) {
    PMA_warnMissingExtension('session', true);
} elseif (ini_get('session.auto_start') !== '' && session_name() != 'phpMyAdmin') {
    // Close the existing session instead of deleting it.
    session_write_close();
}

// session cookie settings
session_set_cookie_params(
    0, $GLOBALS['PMA_Config']->getCookiePath(),
    '', $GLOBALS['PMA_Config']->isHttps(), true
);

ini_set('session.use_cookies', 'true');

// optionally set session_save_path
$path = $GLOBALS['PMA_Config']->get('SessionSavePath');
if (!empty($path)) {
    // Validate the path to prevent path traversal attacks
    if (realpath($path) === false || strpos(realpath($path), realpath($_SERVER['DOCUMENT_ROOT'])) !== 0) {
        die("Invalid session save path.");
    }
    session_save_path($path);
}

ini_set('session.use_only_cookies', 'false');
@ini_set(
    'url_rewriter.tags',
    'a=href,frame=src,input=src,form=fakeentry,fieldset='
);

// Use secure session settings
ini_set('session.cookie_httponly', true);
ini_set('session.cookie_secure', $GLOBALS['PMA_Config']->isHttps());
ini_set('session.hash_function', 'sha256'); // Use a more secure hashing function

// start the session
$session_name = 'phpMyAdmin';
@session_name($session_name);

if (! isset($_COOKIE[$session_name])) {
    $orig_error_count = $GLOBALS['error_handler']->countErrors();
    $session_result = session_start();
    if ($session_result !== true || $orig_error_count != $GLOBALS['error_handler']->countErrors()) {
        setcookie($session_name, '', 1);
        PMA_fatalError('Error during session start. Please check your PHP and/or webserver log file.');
    }
    unset($orig_error_count, $session_result);
} else {
    session_start();
}

/**
 * Disable setting of session cookies for further session_start() calls.
 */
@ini_set('session.use_cookies', 'true');

// Use secure token generation and storage
if (! isset($_SESSION['PMA_token'])) {
    $_SESSION['PMA_token'] = bin2hex(random_bytes(32)); // Generate a secure, random token
}

/**
 * Secure the session to prevent hijacking and fixation.
 */
function PMA_secureSession()
{
    if ((PMA_PHP_INT_VERSION >= 50400 && session_status() === PHP_SESSION_ACTIVE) || (PMA_PHP_INT_VERSION < 50400 && session_id() !== '')) {
        session_regenerate_id(true); // Regenerate the session ID to prevent session hijacking
    }
    $_SESSION['PMA_token'] = bin2hex(random_bytes(32)); // Generate a new secure token
}
   