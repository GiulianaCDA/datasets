<?php
$integrity_errors = array();
$external = filter_var($_GET["external"], FILTER_VALIDATE_BOOLEAN);

// Get the form so we can walk through its fields
$form = BigTreeAutoModule::getForm(filter_var($_GET["form"], FILTER_SANITIZE_NUMBER_INT));

// Create a generic module class to get the decoded item data
$m = new BigTreeModule;
$m->Table = $form["table"];
$item = $m->get(filter_var($_GET["id"], FILTER_SANITIZE_NUMBER_INT));

// Loop through all the fields
foreach ($form["fields"] as $field => $resource) {
    if ($resource["type"] == "html") {
        $integrity_errors[$field] = $admin->checkHTML("", $item[$field], $external);
    } elseif ($resource["type"] == "text" && is_string($item[$field])) {
        $href = $item[$field];
        // External link
        if (strpos($href, "http") === 0 && strpos($href, WWW_ROOT) === false) {
            // Only check external links if we've requested them
            if ($external) {
                if (strpos($href, "#") !== false) {
                    $href = substr($href, 0, strpos($href, "#") - 1);
                }
                if (!$admin->urlExists($href)) {
                    $integrity_errors[$field] = array("a" => array($href));
                }
            }
            // Internal link
        } elseif (strpos($href, "http") === 0) {
            if (!$admin->urlExists($href)) {
                $integrity_errors[$field] = array("a" => array($href));
            }
        }
    }
}

// Only retrieve these if we have errors as we only need them for URL generation
if (count($integrity_errors)) {
    $action = $admin->getModuleActionForForm($form);
    $module = $admin->getModule($action["module"]);
}

foreach ($integrity_errors as $field => $error_types) {
    foreach ($error_types as $type => $errors) {
        foreach ($errors as $error) {
?>
<li>
    <section class="integrity_errors">
        <a href="<?=ADMIN_ROOT . $module["route"] . "/" . $action["route"] . "/" . $_GET["id"]?>/" target="_blank">Edit</a>
        <span class="icon_small icon_small_warning"></span>
        <p>Broken <?=(($type == "img") ? "Image" : "Link")?>: <?=$error?> in field &ldquo;<?=$form["fields"][$field]["title"]?>&rdquo;</p>
    </section>
</li>
<?php
        }
    }
}
?>
