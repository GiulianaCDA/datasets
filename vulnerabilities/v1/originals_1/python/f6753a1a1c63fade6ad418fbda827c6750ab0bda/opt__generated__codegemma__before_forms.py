**AnnouncementForm**

- Form for creating announcements.
- Includes fields for message, category, expiry date, and notification settings.
- Uses the `MarkdownTextarea` widget for the message field.

**ChangesForm**

- Form for filtering changes.
- Includes fields for project, language, action, author username, start date, and end date.
- Dynamically populates the language and project choices based on the request object.

**LabelForm**

- Form for creating or editing labels.
- Includes fields for name and color.
- Uses the `ColorWidget` for the color field.

**ProjectTokenDeleteForm**

- Form for deleting project tokens.
- Includes a hidden field for the project token.

**ProjectTokenCreateForm**

- Form for creating project tokens.
- Includes fields for name, expiry date, and project.
- Validates the expiry date to ensure it is not in the past.

**ProjectGroupDeleteForm**

- Form for deleting project groups.
- Includes a hidden field for the project group.

**ProjectUserGroupForm**

- Form for managing user groups for a project.
- Includes a hidden field for the user and a multiple-choice field for groups.

**Additional Notes:**

- The form classes use a variety of widgets and field types to provide different input options.
- The `WeblateDateField` and `WeblateDateInput` widgets are custom widgets for date input.
- The `SortedSelectMultiple` widget is used for the action field.
- The `UsernameField` widget is used for the author username field.
- The forms are dynamically initialized with project-specific data.