**Description:**

The code snippet you provided is a PHP script that displays a sortable table of sub-languages. Each row in the table represents a sub-language, with columns for the original name, English name, LMS folder, properties, and actions.

**Features:**

* **Sortable:** The table is sortable by clicking on the header cells.
* **Visiblity:** Sub-languages can be made available or unavailable.
* **Edit:** The user can edit sub-languages.
* **Add Word:** The user can add words to a sub-language.
* **Delete:** The user can delete a sub-language.

**Code Breakdown:**

* **Header:** The header row contains the column titles.
* **Data:** Each row represents a sub-language, with columns for the sub-language name, English name, LMS folder, properties, and actions.
* **Actions:** The actions column includes links to make the sub-language available/unavailable, edit it, add words, and delete it.
* **SortableTableFromArrayConfig:** This class is used to create the sortable table.
* **Display::return_icon:** This function is used to display icons in the table.
* **Security::remove_XSS:** This function is used to sanitize user input.
* **Display::display_footer:** This function displays the footer of the page.

**Purpose:**

The script is used to manage sub-languages in a platform. It allows users to view, edit, add words, and delete sub-languages.

**Additional Notes:**

* The code assumes that the necessary data is available in variables called `$language_data`, `$row`, `$row_lang`, and `$all_information_of_sub_language`.
* The icons used in the table are defined in the `Display` class.
* The `SortableTableFromArrayConfig` class provides functionality for creating sortable tables from an array of data.
* The script is wrapped in