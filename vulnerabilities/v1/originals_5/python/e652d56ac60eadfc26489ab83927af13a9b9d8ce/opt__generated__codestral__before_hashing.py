import hashlib, base64, uuid
from cryptography.hazmat.primitives import hashes
from typing import Any, final

class Hashing():
    def __init__(self) -> None:
        self.salt = None

    def generate_salt(self):
        return base64.urlsafe_b64encode(uuid.uuid4().bytes)

    def salted_hash(self, data: bytes, salt: bytes = None):
        if not salt:
            salt = self.generate_salt()
        hashed_data = hashlib.pbkdf2_hmac('sha512', data, salt, 100000)
        return base64.urlsafe_b64encode(hashed_data + salt)

    def SHA256(self, data: str):
        sha = hashlib.sha256(bytes(data.encode()))
        hash = sha.digest()
        return self.salted_hash(hash)

    def SHA512(self, data: str):
        sha = hashlib.sha512(bytes(data.encode()))
        hash = sha.digest()
        return self.salted_hash(hash)

    def SHA244(self, data: str):
        sha = hashlib.sha224(bytes(data.encode()))
        hash = sha.digest()
        return self.salted_hash(hash)

    def SHA384(self, data: str):
        sha = hashlib.sha384(bytes(data.encode()))
        hash = sha.digest()
        return self.salted_hash(hash)

    def BLAKE2(self, data: bytes):
        a = hashes.Hash(hashes.BLAKE2s(32))
        a.update(data)
        return self.salted_hash(a.finalize())
