"""Make a page's links absolute: resolve the base URL, then rewrite against it."""
from urllib.parse import urljoin, urlsplit


def _is_absolute_http(url):
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _first_base_href(root):
    for el in root.iter("base"):
        href = (el.get("href") or "").strip()
        if href:
            return href
    return None


def effective_base(root, base_url=None):
    """The URL to resolve links against, or None to leave them unchanged.

    The first non-empty <base href> is resolved against base_url; a malformed or
    non-http(s) result falls back to base_url. Without base_url, the <base href> is
    used only if it is an absolute http(s) URL. A malformed base_url gives None."""
    href = _first_base_href(root)
    if base_url:
        try:
            urlsplit(base_url)
        except ValueError:
            return None
        if href is None:
            return base_url
        try:
            joined = urljoin(base_url, href)
        except ValueError:
            return base_url
        return joined if _is_absolute_http(joined) else base_url
    return href if href is not None and _is_absolute_http(href) else None


def make_absolute(root, base):
    """Rewrite root's links against base in place; malformed links are left as is.
    <base> tags in root are ignored; srcset is not rewritten."""
    def absolute(link):
        try:
            return urljoin(base, link.strip())
        except ValueError:
            return link

    root.rewrite_links(absolute, resolve_base_href=False)
