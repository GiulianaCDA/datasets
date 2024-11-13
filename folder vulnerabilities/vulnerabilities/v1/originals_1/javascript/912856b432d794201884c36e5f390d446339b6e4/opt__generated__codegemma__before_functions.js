**Functions.js**

**Purpose:**

The functions.js file contains JavaScript functions that are used throughout the PHPMyAdmin (PMA) application.

**Functions:**

* **getCurrentTime()**: Returns the current time in a JSON object with minute and second properties.
* **checkNumberOfFields()**: Checks if any form exceeds the maximum number of allowed input fields.
* **ignorePhpErrors(clearPrevErrors)**: Ignores displayed PHP errors.
* **isStorageSupported(type)**: Checks if web storage is supported in the browser.
* **registerTeardown(script, callback)**: Registers a callback function to be executed when a script is torn down.
* **registerOnload(script, callback)**: Registers a callback function to be executed when a script is loaded.

**Event Handling:**

* **Handle 'Ctrl/Alt + Enter' form submits:** Listens for keydown events on form input, textarea, and select elements and submits the form when the 'Ctrl/Alt + Enter' key combination is pressed.
* **Display SSL warning:** Listens for changes in the "Select authentication plugin" dropdown menu and displays a warning message if the "sha256_password" method is selected.

**Usage:**

The functions in functions.js are used throughout the PMA application to handle various tasks, including error handling, form validation, and event handling.

**Note:**

* The `maxInputVars` variable is not defined in the code snippet, so it is assumed to be a global variable.
* The `PMA_messages` object is assumed to be available in the context where these functions are used.