import hashlib
from base64 import urlsafe_b64encode
from uuid import uuid4

def generate_salt():
    return str(uuid4()).replace("-", "")

def hash_data(data, salt):
    # Use a more secure hashing algorithm like SHA-512
    sha = hashlib.sha512()
    sha.update(data.encode())
    hash = sha.digest()
    return urlsafe_b64encode(hash).decode("utf-8") + generate_salt()

# Example usage:
data = "example data"
salt = generate_salt()
print(hash_data(data, salt))
