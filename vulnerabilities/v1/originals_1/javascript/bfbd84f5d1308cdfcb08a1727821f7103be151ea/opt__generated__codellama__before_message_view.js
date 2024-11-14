[PYTHON]
from typing import List

class Message:
    def __init__(self, text: str, attachments: List[str] = None) -> None:
        self.text = text
        self.attachments = attachments if attachments else []
    
    def get_text(self) -> str:
        return self.text
    
    def get_attachments(self) -> List[str]:
        return self.attachments

class MessageView:
    def __init__(self, message: Message):
        self.message = message
    
    def render(self) -> None:
        pass

# Example usage:
messages = [
    Message("Hello World!"),
    Message("How are you?", ["image1.jpg", "image2.jpg"]),
]
for message in messages:
    MessageView(message).render()
[/PYTHON]
