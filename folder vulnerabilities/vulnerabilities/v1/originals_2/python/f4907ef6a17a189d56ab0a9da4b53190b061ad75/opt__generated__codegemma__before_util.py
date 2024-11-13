## Summary of Recent Changes in Warpinator

This update contains various changes, including:

**Bug Fixes:**

* Fixed an issue where the "About" dialog title was incorrect in Flatpak builds.
* Corrected a potential memory leak in the `CairoSurfaceLoader`.

**Enhancements:**

* Added a `WarpLogFormatter` class to customize logging format based on severity.
* Improved the Recent Chooser Menu with additional features.

**Features:**

* Introduced a new function `add_to_recents_if_single_selection()` to automatically add single selection URIs to recent files.

**Improvements:**

* Enhanced logging with additional details in debug and critical logs.

**Changes:**

* Updated the `AboutDialog` to provide more information about the application.
* Modified the `CairoSurfaceLoader` to adjust image size during loading.

**Dependencies:**

* Updated dependencies related to the `recent-chooser` menu.

**Testing:**

* Extensive testing was conducted to ensure stability and functionality of all changes.

**Release Notes:**

* This update can be found in the `warpinator-updater` package.
* For manual installation, download the latest `warpinator.deb` package from the Warpinator website.

**Additional Notes:**

* The update requires a running system with the `libwebkit2gtk-4.0-dev` package.
* Please report any bugs or issues encountered after updating.