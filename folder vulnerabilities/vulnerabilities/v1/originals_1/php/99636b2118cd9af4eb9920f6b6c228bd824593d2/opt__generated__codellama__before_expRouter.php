
This class is responsible for handling the URL routing and determining which page should be displayed. It uses a series of maps to determine the best match for a given URL. The `getRouterMaps()` method loads the router maps from the `router_maps.php` file, which contains an array of route maps.

The `parseUrl()` method parses the URL and determines whether it is using SEF or non-SEF syntax. It then sets the `$sefPath` property to the appropriate value based on the syntax used in the URL.

The `getSection()` method retrieves the section from the URL, if present. If no section is specified in the URL, it uses the default section defined in the site configuration (`SITE_DEFAULT_SECTION`). It then checks whether the retrieved section exists and has permission to be viewed by the current user. If the section does not exist or the user does not have permission to view it, it returns 0 (false).

The `getSectionObj()` method retrieves the section object from the database using the retrieved section ID. It then sets a series of constants in the PHP environment based on the section's permissions and public status. The `expSession::set('last_section', intval($_REQUEST['section']))` line is used to store the last accessed section in the user's session, which can be useful for other parts of the application that need to remember the last accessed section.

The `getTrackingId()` method returns a tracking ID (ectid) if it exists in the URL, and an empty string otherwise. This is used by the ecommerce module for tracking sales data.

Overall, this class plays a crucial role in determining which page should be displayed based on the URL and user inputs, and provides a convenient way to access the current section and its properties.