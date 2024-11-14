**UserChangeEmailSendCode:**

* The user sends a request to send a code to their new email address.
* The request includes the user's ID, captcha ID, and captcha code.
* The user must be logged in.
* The user's email must be verified.
* The captcha code must be correct.
* The user's IP address is recorded in the action record.
* The API returns a success response if the code is sent successfully.

**UserChangeEmailVerify:**

* The user receives a code in their new email address.
* The user enters the code in the API request.
* The API verifies the code and returns a success response if the code is valid.

**UserRanking:**

* The API returns a list of users ranked by a specific metric.

**UserUnsubscribeEmailNotification:**

* The user receives a link to unsubscribe from email notifications.
* The user clicks the link and enters the code in the API request.
* The API unsubscribes the user from email notifications.

**Additional Notes:**

* The API uses the middleware to get the user's ID from the context.
* The API uses the translator to translate error messages.
* The API uses the action service to record user actions.
* The API uses the user service to perform user operations.
* The API uses the email service to send emails.