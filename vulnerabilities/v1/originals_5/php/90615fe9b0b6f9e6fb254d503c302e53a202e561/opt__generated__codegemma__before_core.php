**Class Name:** wsBrokenLinkChecker

**Purpose:**

The `wsBrokenLinkChecker` class is a WordPress plugin that checks for broken links and sends email notifications.

**Key Methods:**

* `setup_cron_events()`: Installs or uninstalls Cron events based on plugin settings.
* `load_language()`: Loads the plugin's textdomain.
* `check_news()`: Checks for news notices related to the plugin.
* `promote_warnings_to_broken()`: Converts warning links to broken links.

**Properties:**

* `conf`: An instance of the `wsBrokenLinkChecker_Config` class.
* `is_textdomain_loaded`: A boolean indicating whether the plugin's textdomain has been loaded.

**Cron Events:**

* `blc_cron_check_links`: Checks for broken links hourly.
* `blc_cron_email_notifications`: Sends email notifications about broken links based on schedule.
* `blc_cron_database_maintenance`: Runs database maintenance every two weeks.
* `blc_cron_check_news`: Checks for news notices daily.

**Text Domain:**

The plugin's textdomain is `broken-link-checker`.

**Log Directory:**

The plugin logs to a file in the WordPress uploads directory, with the filename `blc-log.txt`.

**Usage:**

The plugin is automatically activated upon installation. The `setup_cron_events()` method is responsible for installing the Cron events.

**Additional Notes:**

* The `content_type()` method is not used in the provided code.
* The `get_default_log_directory()` and `get_default_log_basename()` methods are used to set the default log file location.
* The `check_news()` method retrieves news notices from a remote server and stores them in the plugin's configuration.
* The `promote_warnings_to_broken()` method is used to automatically convert warning links to broken links.