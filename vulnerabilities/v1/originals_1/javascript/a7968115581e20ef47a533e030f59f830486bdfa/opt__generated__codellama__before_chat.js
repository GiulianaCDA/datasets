This is a JavaScript module that defines a `Chat` object with several methods for interacting with the chat feature in a collaborative pad. The module exports an instance of the `Chat` class, which can be used to send and receive chat messages, as well as manage the chat history and mentions.

The `Chat` class has the following methods:

* `send()`: sends a new message to the pad and clears the input field.
* `show()`: displays the chat window and focuses on the input field.
* `hide()`: hides the chat window.
* `scrollDown()`: scrolls the chat history down to the most recent messages.
* `init(pad)`: initializes the chat feature with a reference to the pad object. This method is called automatically when the module is loaded.

The `Chat` class also has several event handlers for chat-related events, such as new messages being sent and received, and the input field being clicked or focused on. These event handlers update the chat history and mentions accordingly.