<?php
// Include necessary files and check login status
require("../inc/main.php");
must_login();

if (isset($_GET['action']) && $_GET['action'] == "delete" && isset($_GET['id'])) {
    // Use prepared statements to prevent SQL injection
    $stmt = $db->prepare("DELETE FROM webfinance_pref WHERE id_pref=?");
    $stmt->bind_param("i", $_GET['id']);
    $stmt->execute();
    $_SESSION['message'] = _('Tax deleted');
    header("Location: preferences.php?tab=Taxes");
    exit;
}

foreach ($_POST['taxes'] as $id=>$data) {
    if (isset($id) && isset($data['taxe']) && isset($data['value'])) {
        // Use prepared statements to prevent SQL injection
        if ($id == "new") {
            $stmt = $db->prepare("INSERT INTO webfinance_pref SET type_pref=?, value=?");
            $type_pref = 'taxe_' . $data['taxe'];
            $value = $data['value'];
        } else {
            $stmt = $db->prepare("UPDATE webfinance_pref SET type_pref=?, value=? WHERE id_pref=?");
            $type_pref = 'taxe_' . $data['taxe'];
            $value = $data['value'];
        }
        $stmt->bind_param("ssi", $type_pref, $value, $id);
        $stmt->execute();
    }
}

header("Location: preferences.php?tab=Taxes");
exit;
?>
