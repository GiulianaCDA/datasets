$integrity_errors = array();
// Sanitize and validate GET variables
$external = filter_input(INPUT_GET, 'external', FILTER_VALIDATE_BOOLEAN);
$formId = filter_input(INPUT_GET, 'form', FILTER_SANITIZE_NUMBER_INT);
$itemId = filter_input(INPUT_GET, 'id', FILTER_SANITIZE_NUMBER_INT);

// Get the form so we can walk through its fields
try {
    $form = BigTreeAutoModule::getForm($formId);
} catch (Exception $e) {
    // Handle error
    die("An error occurred while retrieving the form data.");
}

// Create a generic module class to get the decoded item data
$m = new BigTreeModule;
$m->Table = $form["table"];
try {
    $item = $m->get($itemId);
} catch (Exception $e) {
    // Handle error
    die("An error occurred while retrieving the item data.");
}

// Loop through all the fields
foreach ($form["fields"] as $field => $resource) {
    if ($resource["type"] == "html") {
        $integrity_errors[$field] = $admin->checkHTML("", htmlspecialchars($item[$field]), $external);
    } elseif ($resource["type"] == "text" && is_string($item[$field])) {
        $href = filter_var($item[$field], FILTER_VALIDATE_URL);
        // External link
        if ($href !== false && strpos($href, WWW_ROOT) === false) {
            // Only check external links if we've requested them
            if ($external) {
                $parsedUrl = parse_url($href);
                $href = $parsedUrl['scheme'] . '://' . $parsedUrl['host'] . $parsedUrl['path'];
                if (!$admin->urlExists($href)) {
                    $integrity_errors[$field] = array("a" => array(htmlspecialchars($href)));
                }
            }
        // Internal link
        } elseif ($href !== false) {
            if (!$admin->urlExists($href)) {
                $integrity_errors[$field] = array("a" => array(htmlspecialchars($href)));
            }
        }
    }
}

// Only retrieve these if we have errors as we only need them for URL generation
if (count($integrity_errors)) {
    try {
        $action = $admin->getModuleActionForForm($form);
        $module = $admin->getModule($action["module"]);
    } catch (Exception $e) {
        // Handle error
        die("An error occurred while retrieving the module data.");
    }
}

foreach ($integrity_errors as $field => $error_types) {
    foreach ($error_types as $type => $errors) {
        foreach ($errors as $error) {
?>
<li>
	<section class="integrity_errors">
		<a href="<?=htmlspecialchars(ADMIN_ROOT . $module["route"] . "/" . $action["route"] . "/" . $itemId)?>" target="_blank">Edit</a>
		<span class="icon_small icon_small_warning"></span>
		<p>Broken <?=(($type == "img") ? "Image" : "Link")?>: <?=$error?> in field &ldquo;<?=htmlspecialchars($form["fields"][$field]["title"])?>&rdquo;</p>
	</section>
</li>
<?php
        }
    }
}
