<?php

ini_set('max_execution_time', 300);

?>

<html>
<head>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap-theme.min.css">
    <style>
        ul {
            margin: 0;
            padding: 0;
        }

        ul li {
            list-style-type: none;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="row">
        <div class="col-md-12">
            <h1>iubenda class test</h1>
            <form action="" method="POST">
                <strong>URL WEBSITE</strong><BR>
                <input type="text" name="url" value="<?php echo htmlspecialchars($_POST['url'] ?? ''); ?>">
                <input type="submit" class="btn" value="Analyze">
            </form>
        </div>

        <?php
        if (!empty($_POST['url'])) {
            $url = filter_var($_POST['url'], FILTER_SANITIZE_URL);
        } else {
            $url = '';
        }

        if ($url) {

            function print_stuff($array)
            {
                if (count($array)) {
                    echo "<ul>";
                    foreach ($array as $r) {
                        echo "<li><pre><code>" . htmlspecialchars($r) . "</code></pre></li>";
                    }
                    echo "</ul>";
                } else {
                    echo "<p>Nothing</p>";
                }
            }

            echo '<div class="col-md-12" style="padding-bottom:150px;"><h2>RESULTS</H2>';

            include_once('iubenda.class.php');

            $content = file_get_contents($url);

            $type = isset($_GET['type']) && in_array($_GET['type'], array('page', 'faster'), true) ? $_GET['type'] : 'page';
            $iubenda = new iubendaParser($content, array('type' => $type));
            $iubenda->parse();

            $iub_comments_detected = count($iubenda->iub_comments_detected);
            $scripts_detected = count($iubenda->scripts_detected);
            $iframes_detected = count($iubenda->iframes_detected);
            $iframes_converted = count($iubenda->iframes_converted);
            $script_inline_detected = count($iubenda->scripts_inline_detected);
            $script_inline_converted = count($iubenda->scripts_inline_converted);

            echo "<p>iubenda comments detected: $iub_comments_detected<br>
                 scripts detected: $scripts_detected<br>
                 scripts autoconverted: $script_converted<br>
                 script inline detected: $script_inline_detected<br>
                 script inline autoconverted: $script_inline_converted<br>
                 iframes detected: $iframes_detected<br>
                 iframes autoconverted: $iframes_converted</p>";

            echo "<H3>DETAILS</H3>";

            echo "<H4>iubenda comments stuff</h4>";
            print_stuff($iubenda->iub_comments_detected);
            echo "<H4>Script detected</h4>";
            print_stuff($iubenda->scripts_detected);
            echo "<H4>Script converted</h4>";
            print_stuff($iubenda->scripts_converted);
            echo "<H4>Script inline detected</h4>";
            print_stuff($iubenda->scripts_inline_detected);
            echo "<H4>Script inline converted</h4>";
            print_stuff($iubenda->scripts_inline_converted);
            echo "<H4>Iframe detected</h4>";
            print_stuff($iubenda->iframes_detected);
            echo "<H4>Iframe converted</h4>";
            print_stuff($iubenda->iframes_converted);

            echo "</div>";
        }
        ?>
    </div>
</div>

</body>
</html>
