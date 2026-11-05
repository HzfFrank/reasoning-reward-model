"""Explicit API endpoint and private key file; no user-home credential discovery."""
import os
from pathlib import Path
from urllib.parse import urlparse

def load_api_settings():
    url = os.environ.get('V7_API_BASE_URL', '')
    key = os.environ.get('V7_API_KEY_FILE', '')
    if urlparse(url).scheme not in ('https', 'http') or not urlparse(url).netloc:
        raise ValueError('set V7_API_BASE_URL explicitly')
    path = Path(key).expanduser()
    if not key or not path.is_file():
        raise ValueError('set V7_API_KEY_FILE to a private credential file')
    return {'base_url': url, 'auth': {'args': [str(path)]}}
