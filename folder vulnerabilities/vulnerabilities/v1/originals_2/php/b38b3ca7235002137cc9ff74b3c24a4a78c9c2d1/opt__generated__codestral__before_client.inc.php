<?php
/*********************************************************************
    client.inc.php

    File included on every client page

    Peter Rotich <peter@osticket.com>
    Copyright (c)  2006-2013 osTicket
    http://www.osticket.com

    Released under the GNU General Public License WITHOUT ANY WARRANTY.
    See LICENSE.TXT for details.

    vim: expandtab sw=4 ts=4 sts=4:
**********************************************************************/
if(!strcasecmp(basename($_SERVER['SCRIPT_NAME']),basename(__FILE__))) {
    die('kwaheri rafiki!');
}

$thisdir = str_replace('\\', '/', dirname(__FILE__)) . '/';
if(!file_exists($thisdir.'main.inc.php')) {
    error_log("Fatal Error: main.inc.php not found.");
    die('Internal Server Error');
}

require_once($thisdir.'main.inc.php');

if(!defined('INCLUDE_DIR')) {
    error_log("Fatal error: INCLUDE_DIR not defined.");
    die('Internal Server Error');
}

define('CLIENTINC_DIR', INCLUDE_DIR . 'client/');
define('OSTCLIENTINC', TRUE);

define('ASSETS_PATH', ROOT_PATH . 'assets/default/');

// Check the status of the HelpDesk.
if (!in_array(strtolower(basename($_SERVER['SCRIPT_NAME'])), array('logo.php')) && !(is_object($ost) && $ost->isSystemOnline())) {
    include(ROOT_DIR . 'offline.php');
    exit;
}

require_once(INCLUDE_DIR . 'class.client.php');
require_once(INCLUDE_DIR . 'class.ticket.php');
require_once(INCLUDE_DIR . 'class.dept.php');

$errors = array();
$msg = '';
$nav = null;

$thisclient = UserAuthenticationBackend::getUser();

if (isset($_GET['lang']) && $_GET['lang']) {
    if(preg_match('/^[a-z]{2}([-_][A-Z]{2})?$/', $_GET['lang'])) {
        $_SESSION['client:lang'] = $_GET['lang'];
    } else {
        error_log("Invalid language parameter.");
    }
}

TextDomain::configureForUser($thisclient);

if ($thisclient && $thisclient->getId() && $thisclient->isValid()) {
     $thisclient->refreshSession();
} else {
    $thisclient = null;
}

// Enforce CSRF protection for POSTS
if ($_POST  && !$ost->checkCSRFToken()) {
    error_log("Invalid CSRF token.");
    Http::redirect('index.php');
    exit;
}

$ost->addExtraHeader('<meta name="csrf_token" content="'.$ost->getCSRFToken().'" />');

define('PAGE_LIMIT', DEFAULT_PAGE_LIMIT);

$nav = new UserNav($thisclient, 'home');

$exempt = in_array(basename($_SERVER['SCRIPT_NAME']), array('logout.php', 'ajax.php', 'logs.php', 'upgrade.php'));

if (!$exempt && $thisclient && ($acct = $thisclient->getAccount()) && $acct->isPasswdResetForced()) {
    $warn = __('Password change required to continue');
    require_once('profile.php'); // Using require_once to avoid duplicate declarations of variables or functions
    exit;
}
?>
