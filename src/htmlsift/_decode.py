"""Decode HTML bytes to str.

Resolution order: BOM, encoding= argument, <meta> declaration, strict utf-8, cp1252.
Undecodable bytes become U+FFFD.
"""
from w3lib.encoding import html_body_declared_encoding, read_bom, resolve_encoding, to_unicode


def _utf8_or_cp1252(data):
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "cp1252"
    return "utf-8"


def decode_html(data, encoding=None):
    """Decode HTML bytes to str. encoding= is a charset label, e.g. from an HTTP or WARC
    Content-Type header. Raises TypeError for non-bytes input."""
    if not isinstance(data, bytes):
        raise TypeError(f"decode_html expects bytes, got {type(data).__name__}")
    bom_enc, bom = read_bom(data)
    if bom_enc:
        return to_unicode(data[len(bom):], bom_enc)
    enc = resolve_encoding(encoding.strip()) if encoding else None
    if enc is None:
        enc = html_body_declared_encoding(data)
        # A <meta> readable as ASCII cannot truthfully declare utf-16/32.
        if enc and enc.startswith(("utf-16", "utf-32")):
            enc = "utf-8"
    if enc is None:
        enc = _utf8_or_cp1252(data)
    return to_unicode(data, enc)


def prepare_input(html, encoding=None):
    """Extractor input -> what parse() takes: str unchanged, bytes decoded and re-encoded
    as utf-8. Raises TypeError for other types, or for encoding= with str input."""
    if isinstance(html, str):
        if encoding is not None:
            raise TypeError("encoding= applies to bytes input; str is already decoded")
        return html
    if isinstance(html, bytes):
        return decode_html(html, encoding).encode("utf-8")
    raise TypeError(f"html must be str or bytes, got {type(html).__name__}")
