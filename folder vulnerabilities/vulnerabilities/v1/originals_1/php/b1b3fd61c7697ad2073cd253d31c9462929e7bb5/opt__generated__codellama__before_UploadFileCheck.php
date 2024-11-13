if ($json->decode(htmlspecialchars($forQuotes))) {
    if ($forQuotes === "quotes" || $forQuotes === "company") {
        // Do something with the validated value of forQuotes
    } else {
        echo "Invalid value for forQuotes parameter";
        sugar_cleanup();
        exit();
    }
} else {
    echo "Malicious input detected in forQuotes parameter";
    sugar_cleanup();
    exit();
}
