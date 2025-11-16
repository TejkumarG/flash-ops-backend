"""
Encryption utilities for database credentials.
"""
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import binascii

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("encryption")


def decrypt_password(encrypted_password: str) -> str:
    """
    Decrypt database password.

    Format: iv:ciphertext (both hex encoded)

    Args:
        encrypted_password: Encrypted password string

    Returns:
        Decrypted password
    """
    try:
        # Split IV and ciphertext
        parts = encrypted_password.split(':')
        if len(parts) != 2:
            raise ValueError("Invalid encrypted password format. Expected 'iv:ciphertext'")

        iv_hex, ciphertext_hex = parts

        # Convert from hex
        iv = binascii.unhexlify(iv_hex)
        ciphertext = binascii.unhexlify(ciphertext_hex)

        # Get encryption key
        key = binascii.unhexlify(settings.ENCRYPTION_KEY)

        # Decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(ciphertext)

        # Remove padding
        decrypted = unpad(decrypted_padded, AES.block_size)

        return decrypted.decode('utf-8')

    except Exception as e:
        logger.error(f"Failed to decrypt password: {e}")
        raise ValueError(f"Password decryption failed: {e}")


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt API key.

    Format: iv:ciphertext (both hex encoded)

    Args:
        encrypted_key: Encrypted API key string

    Returns:
        Decrypted API key
    """
    try:
        # Split IV and ciphertext
        parts = encrypted_key.split(':')
        if len(parts) != 2:
            raise ValueError("Invalid encrypted API key format. Expected 'iv:ciphertext'")

        iv_hex, ciphertext_hex = parts

        # Convert from hex
        iv = binascii.unhexlify(iv_hex)
        ciphertext = binascii.unhexlify(ciphertext_hex)

        # Get encryption key
        key = binascii.unhexlify(settings.ENCRYPTION_KEY)

        # Decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(ciphertext)

        # Remove padding
        decrypted = unpad(decrypted_padded, AES.block_size)

        return decrypted.decode('utf-8')

    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        raise ValueError(f"API key decryption failed: {e}")
