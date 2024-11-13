<?php
//
// This file is part of « Webfinance »
//
// Copyright (c) 2004-2006 NBI SARL
// Author : Nicolas Bouthors <nbouthors@nbi.fr>
//
// You can use and redistribute this file under the term of the GNU GPL v2.0
//

// $Id: save_taxes.php 531 2007-06-13 12:32:31Z thierry $

require("../inc/main.php");
must_login();

if (isset($_GET['action']) && $_GET['action'] == "delete") {
  $id = (int) $_GET['id'];
  mysql_query("DELETE FROM webfinance_pref WHERE id_pref = $id");
  $_SESSION['message'] = _('Tax deleted');
  header("Location: preferences.php?tab=Taxes");
  exit;
}

foreach ($_POST['taxes'] as $id => $data) {
  if ($id == "new") {
    if (!empty($data['taxe'])) {
      $q = sprintf("INSERT INTO webfinance_pref SET type_pref='taxe_%s', value='%s'", $data['taxe'], mysql_real_escape_string($data['value']));
      $_SESSION['message'] = _('Tax added');
    }
  } else {
    $id = (int) $id;
    $q = sprintf("UPDATE webfinance_pref SET type_pref='taxe_%s', value='%s' WHERE id_pref = $id", $data['taxe'], mysql_real_escape_string($data['value']));
    $_SESSION['message'] = _('Tax updated');
  }
  if (isset($q) && !empty($q))
    mysql_query($q) or wf_mysqldie();
}

header("Location: preferences.php?tab=Taxes");
exit;


?>
