import re
import unicodedata


def normalized(text: str) -> str:
    # Keep case and punctuation, which may encode a name or change the meaning.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
