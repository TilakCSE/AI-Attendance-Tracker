"""
security.py

Utilities for encryption key management and secure encryption/decryption
of face embeddings using Fernet (symmetric authenticated encryption).
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
from cryptography.fernet import Fernet


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _default_key_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    secrets_dir = os.path.join(base_dir, ".secrets")
    _ensure_dir(secrets_dir)
    return os.path.join(secrets_dir, "fernet.key")


def get_fernet(key_path: Optional[str] = None) -> Fernet:
    """
    Get or create a Fernet key and return a Fernet instance.
    The key is stored under ./.secrets/fernet.key by default.
    """
    key_file = key_path or _default_key_path()
    if not os.path.exists(key_file):
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
    else:
        with open(key_file, "rb") as f:
            key = f.read()
    return Fernet(key)


def encrypt_embedding_array(arr: np.ndarray, fernet: Optional[Fernet] = None) -> bytes:
    """
    Serialize a numpy array embedding and encrypt it with Fernet.
    """
    f = fernet or get_fernet()
    buf = io.BytesIO()
    # Use a compact and safe dtype for embeddings
    np.save(buf, arr.astype(np.float32), allow_pickle=False)
    data = buf.getvalue()
    return f.encrypt(data)


def decrypt_embedding_array(token: bytes, fernet: Optional[Fernet] = None) -> np.ndarray:
    """
    Decrypt a Fernet token and deserialize into a numpy array embedding.
    """
    f = fernet or get_fernet()
    data = f.decrypt(token)
    buf = io.BytesIO(data)
    arr = np.load(buf, allow_pickle=False)
    return arr
