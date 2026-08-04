"""URL download, validation, and HTML extraction service.

Supports:
- Downloading PDF files from HTTP/HTTPS URLs with timeout and retry
- Extracting text from HTML pages via BeautifulSoup
- Parsing URL lists from TXT, CSV, and JSON files
- URL validation, deduplication, and safe filename derivation
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

# ── URL validation ────────────────────────────────────────────────────

# Private / internal IP ranges (IPv4)
_PRIVATE_NETS = [
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^0\."),
]

# Known internal hostnames
_INTERNAL_HOSTS = {"localhost", "0.0.0.0", "[::]", "::1", "[::1]"}


def validate_url(url: str) -> tuple[bool, str]:
    """Validate a URL for safety and correctness.

    Returns:
        ``(is_valid, error_message)`` — error_message is empty when valid.
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or not a string."

    url = url.strip()

    # Must start with http:// or https://
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme '{parsed.scheme}'. Only http and https are accepted."

    if not parsed.hostname:
        return False, "URL has no hostname."

    # Block internal / private IPs
    hostname = parsed.hostname.lower()

    if hostname in _INTERNAL_HOSTS:
        return False, f"Internal hostname '{hostname}' is not allowed."

    # Check IPv4 private ranges
    for pattern in _PRIVATE_NETS:
        if pattern.match(hostname):
            return False, f"Private/internal IP '{hostname}' is not allowed."

    return True, ""


# ── Content type detection ────────────────────────────────────────────

PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # some servers return this for PDF
}

HTML_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
}


def is_pdf_content(content_type: str, url: str = "") -> bool:
    """Determine whether a response is a PDF based on Content-Type or URL extension."""
    ct = content_type.lower().split(";")[0].strip()
    if ct in PDF_CONTENT_TYPES:
        # When Content-Type is octet-stream, also check the URL extension
        if ct == "application/octet-stream" and url:
            return Path(urlparse(url).path).suffix.lower() == ".pdf"
        return True
    return False


def is_html_content(content_type: str) -> bool:
    """Determine whether a response is HTML."""
    ct = content_type.lower().split(";")[0].strip()
    return ct in HTML_CONTENT_TYPES


# ── Filename derivation ───────────────────────────────────────────────

# Unsafe filename characters (Windows + Unix)
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_FILENAME_LEN = 200


def derive_filename(url: str, content_type: str = "") -> str:
    """Derive a safe filename from a URL.

    Strategy:
    1. Extract the path component of the URL.
    2. Use the last path segment as the base filename.
    3. If no meaningful filename, hash the URL.
    4. Append the correct extension based on Content-Type.
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        segments = path.split("/")
        name = segments[-1] if segments[-1] else segments[-2] if len(segments) >= 2 else ""
    else:
        name = ""

    # If name is empty or has no extension, use a hash
    name = _UNSAFE_FILENAME_RE.sub("_", name).strip()
    if not name or "." not in name:
        # Use the domain + hash for uniqueness
        domain = parsed.hostname or "unknown"
        hash_part = hashlib.md5(url.encode()).hexdigest()[:12]
        name = f"{domain}_{hash_part}"

    # Ensure reasonable length
    if len(name) > _MAX_FILENAME_LEN:
        base, ext = _splitext(name)
        name = base[:_MAX_FILENAME_LEN - len(ext)] + ext

    # Append correct extension based on Content-Type if missing
    ct = content_type.lower().split(";")[0].strip()
    if ct in PDF_CONTENT_TYPES and not name.lower().endswith(".pdf"):
        name += ".pdf"
    elif ct in HTML_CONTENT_TYPES and not name.lower().endswith((".html", ".htm")):
        name += ".html"

    # Ensure non-empty
    if not name:
        name = f"download_{int(time.time())}"

    return name


def _splitext(name: str) -> tuple[str, str]:
    """Split filename into (stem, extension)."""
    for i in range(len(name) - 1, -1, -1):
        if name[i] == ".":
            return name[:i], name[i:]
    return name, ""


# ── HTTP download ─────────────────────────────────────────────────────

def _build_client(timeout: int) -> httpx.Client:
    """Build an httpx Client with conservative settings."""
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=settings.url_fetch_connect_timeout,
            read=timeout,
            write=30.0,
            pool=10.0,
        ),
        follow_redirects=True,
        max_redirects=5,
        headers={"User-Agent": settings.user_agent},
    )


def download_file(
    url: str,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> tuple[bytes, str, str]:
    """Download a file from a URL with timeout and retry.

    Args:
        url: The URL to download.
        timeout: Read timeout in seconds.  Falls back to ``settings.url_fetch_timeout``.
        max_retries: Max retry attempts.  Falls back to ``settings.url_fetch_max_retries``.

    Returns:
        ``(content_bytes, content_type, final_url)`` — final_url may differ
        from the input URL due to redirects.

    Raises:
        ValueError: URL validation failed.
        httpx.HTTPError: Download failed after all retries.
    """
    timeout = timeout or settings.url_fetch_timeout
    max_retries = max_retries or settings.url_fetch_max_retries

    # 1) Validate
    is_valid, err = validate_url(url)
    if not is_valid:
        raise ValueError(f"URL validation failed: {err}")

    # 2) Check Content-Length first (HEAD request)
    with _build_client(timeout) as client:
        try:
            head_resp = client.head(url, follow_redirects=True)
            content_length = head_resp.headers.get("content-length")
            if content_length:
                cl = int(content_length)
                if cl > settings.url_max_file_size:
                    raise ValueError(
                        f"Content-Length {cl} exceeds max allowed size "
                        f"{settings.url_max_file_size} bytes. Refusing to download."
                    )
        except httpx.HTTPError:
            pass  # HEAD failed, proceed with GET anyway

    # 3) GET with retry
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with _build_client(timeout) as client:
                response = client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                # Enforce size limit on the actual body
                body = response.content
                if len(body) > settings.url_max_file_size:
                    raise ValueError(
                        f"Downloaded file size {len(body)} exceeds max "
                        f"{settings.url_max_file_size} bytes."
                    )

                return body, content_type, str(response.url)

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_error = exc
            if attempt < max_retries:
                delay = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Download attempt %d/%d for %s failed: %s. Retrying in %ds…",
                    attempt + 1, max_retries + 1, url, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error("Download exhausted retries for %s: %s", url, exc)

        except httpx.HTTPStatusError as exc:
            # Don't retry on 4xx (client errors)
            raise

    # All retries exhausted
    raise RuntimeError(
        f"Failed to download {url} after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


# ── HTML text extraction ──────────────────────────────────────────────

# Tags to remove before text extraction
_REMOVE_TAGS = [
    "script", "style", "noscript", "iframe", "svg",
    "nav", "footer", "header", "aside",
]


def _detect_html_encoding(
    html_bytes: bytes, content_type: str = "",
) -> str:
    """Detect the encoding of an HTML document.

    Priority:
    1. ``charset`` parameter in HTTP ``Content-Type`` header (if provided).
    2. ``<meta charset="...">`` in the HTML head.
    3. ``<meta http-equiv="Content-Type" content="...; charset=...">``.
    4. Heuristic: if the page contains bytes in the GBK lead-byte range
       (0x81-0xFE) and few lone bytes > 0x7F, prefer ``gb18030``.
    5. Fallback: ``utf-8`` (BeautifulSoup default).
    """
    import codecs

    # 1) HTTP Content-Type header hint
    if content_type:
        m = re.search(r"charset\s*=\s*([^\s;]+)", content_type, re.I)
        if m:
            hinted = m.group(1).strip().lower().strip("\"'")
            try:
                codecs.lookup(hinted)
                return hinted
            except LookupError:
                pass

    # 2–3) Inspect HTML <meta> tags with a quick regex (before full parse)
    head = html_bytes[:4096]  # only need the head
    # <meta charset="gbk">
    m = re.search(rb'<meta\s[^>]*charset\s*=\s*["\']?([a-zA-Z0-9_-]+)', head, re.I)
    if m:
        enc = m.group(1).decode("ascii").lower()
        try:
            codecs.lookup(enc)
            return enc
        except LookupError:
            pass
    # <meta http-equiv="Content-Type" content="text/html; charset=gbk">
    m = re.search(
        rb'<meta[^>]*content\s*=\s*["\'][^"\']*charset\s*=\s*([a-zA-Z0-9_-]+)',
        head, re.I,
    )
    if m:
        enc = m.group(1).decode("ascii").lower()
        try:
            codecs.lookup(enc)
            return enc
        except LookupError:
            pass

    # 4) Heuristic: if the raw bytes look like GBK (many double-byte sequences),
    #    prefer gb18030 over utf-8.  Count byte pairs in 0x81-0xFE range
    #    followed by 0x40-0xFE (GBK/GB18030 double-byte region).
    gbk_leads = 0
    for i in range(len(head) - 1):
        if 0x81 <= head[i] <= 0xFE and 0x40 <= head[i + 1] <= 0xFE:
            gbk_leads += 1
    if gbk_leads > 4:
        return "gb18030"

    # 5) Default
    return "utf-8"


def extract_html_text(
    html_bytes: bytes,
    content_type: str = "",
) -> str:
    """Extract readable text from an HTML document.

    Removes script/style/nav/footer/header elements, then extracts
    the plain text from the ``<body>`` using BeautifulSoup.

    Args:
        html_bytes: Raw HTML content as bytes.
        content_type: Optional HTTP ``Content-Type`` header value
            (e.g. ``"text/html; charset=gbk"``) to help encoding detection.

    Returns:
        Extracted text with paragraphs separated by newlines.
    """
    encoding = _detect_html_encoding(html_bytes, content_type)
    soup = BeautifulSoup(html_bytes, "lxml", from_encoding=encoding)

    # Remove unwanted elements
    for tag in _REMOVE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    # Also remove hidden elements
    for el in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        el.decompose()
    for el in soup.find_all(attrs={"aria-hidden": "true"}):
        el.decompose()

    # Extract text from body (or entire document if no body)
    body = soup.find("body")
    root = body if body else soup

    # Use get_text with separator to preserve paragraph structure
    text = root.get_text(separator="\n", strip=True)

    # Clean up: collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ── URL list parsing ──────────────────────────────────────────────────

def parse_url_list(content: str | bytes, ext: str) -> list[str]:
    """Parse URLs from a file's content.

    Supported formats:
    - **TXT**: one URL per line
    - **CSV**: URLs in the first column (skips header row if first cell
      doesn't start with ``http``)
    - **JSON**: array of URLs, or array of objects with a ``url`` key

    Args:
        content: File content as string or bytes.
        ext: File extension without dot (e.g. ``"txt"``, ``"csv"``, ``"json"``).

    Returns:
        List of URL strings (not yet validated or deduplicated).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    ext = ext.lower().strip(".")

    if ext in ("txt", "text"):
        return _parse_txt(content)

    if ext == "csv":
        return _parse_csv(content)

    if ext == "json":
        return _parse_json(content)

    raise ValueError(f"Unsupported URL list file format: .{ext}")


def _parse_txt(content: str) -> list[str]:
    """Extract URLs from plain text (one per line)."""
    urls: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip comment lines
        if line.startswith("#"):
            continue
        # Extract URL from the line (may have surrounding text)
        url_match = re.search(r"https?://\S+", line)
        if url_match:
            urls.append(url_match.group(0).rstrip(".,;:!?\"'）)】]"))
        elif line.startswith("http"):
            urls.append(line)
    return urls


def _parse_csv(content: str) -> list[str]:
    """Extract URLs from CSV (first column)."""
    urls: list[str] = []
    reader = csv.reader(io.StringIO(content))
    for row in reader:
        if not row or not row[0].strip():
            continue
        cell = row[0].strip()
        # Skip header row
        if not cell.startswith("http"):
            continue
        if re.match(r"^https?://", cell):
            urls.append(cell)
    return urls


def _parse_json(content: str) -> list[str]:
    """Extract URLs from JSON."""
    data = json.loads(content)

    if isinstance(data, list):
        urls: list[str] = []
        for item in data:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict):
                # Try common keys: url, link, href, download_url
                for key in ("url", "link", "href", "download_url", "pdf_url"):
                    val = item.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        urls.append(val)
                        break
        return urls

    if isinstance(data, dict):
        urls: list[str] = []
        for val in data.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.startswith("http"):
                        urls.append(item)
            elif isinstance(val, str) and val.startswith("http"):
                urls.append(val)
        return urls

    raise ValueError("JSON content is neither an array nor an object.")


# ── URL deduplication ─────────────────────────────────────────────────

def deduplicate_urls(urls: list[str]) -> list[str]:
    """Remove duplicate URLs, preserving first-occurrence order.

    Normalises URLs by stripping trailing slashes and fragments before
    comparison, but returns the original form.
    """
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if not isinstance(url, str):
            continue
        url = url.strip()
        if not url:
            continue

        # Normalise for comparison: lowercase scheme+host, remove fragment
        try:
            parsed = urlparse(url)
            norm = (
                parsed.scheme.lower()
                + "://"
                + (parsed.hostname or "").lower()
                + parsed.path.rstrip("/")
                + ("?" + parsed.query if parsed.query else "")
            )
        except Exception:
            norm = url.lower()

        if norm not in seen:
            seen.add(norm)
            result.append(url)

    return result


# ── Convenience: download and auto-detect ─────────────────────────────

class DownloadResult:
    """Result of a URL download attempt."""

    def __init__(
        self,
        url: str,
        success: bool,
        source_type: str = "",
        content_bytes: bytes = b"",
        text: str = "",
        filename: str = "",
        content_type: str = "",
        error: str = "",
    ):
        self.url = url
        self.success = success
        self.source_type = source_type  # "http_pdf" | "http_html"
        self.content_bytes = content_bytes
        self.text = text
        self.filename = filename
        self.content_type = content_type
        self.error = error


def fetch_and_detect(url: str) -> DownloadResult:
    """Download a URL and auto-detect its type (PDF or HTML).

    This is the main entry point for single-URL processing.
    On success, ``content_bytes`` is populated for PDFs, ``text`` for HTML.

    Returns:
        ``DownloadResult`` with all metadata populated.
    """
    try:
        body, ct, final_url = download_file(url)
        filename = derive_filename(final_url, ct)

        if is_pdf_content(ct, final_url):
            return DownloadResult(
                url=url,
                success=True,
                source_type="http_pdf",
                content_bytes=body,
                filename=filename,
                content_type=ct,
            )

        if is_html_content(ct):
            if settings.html_extraction_mode == "skip":
                return DownloadResult(
                    url=url,
                    success=False,
                    source_type="http_html",
                    filename=filename,
                    content_type=ct,
                    error="HTML extraction is disabled (html_extraction_mode=skip).",
                )
            try:
                text = extract_html_text(body, content_type=ct)
                if not text.strip():
                    return DownloadResult(
                        url=url,
                        success=False,
                        source_type="http_html",
                        filename=filename,
                        content_type=ct,
                        error="HTML page contains no extractable text.",
                    )
                return DownloadResult(
                    url=url,
                    success=True,
                    source_type="http_html",
                    text=text,
                    filename=filename,
                    content_type=ct,
                )
            except Exception as exc:
                return DownloadResult(
                    url=url,
                    success=False,
                    source_type="http_html",
                    filename=filename,
                    content_type=ct,
                    error=f"HTML text extraction failed: {exc}",
                )

        # Unknown content type
        return DownloadResult(
            url=url,
            success=False,
            filename=filename,
            content_type=ct,
            error=f"Unsupported content type '{ct}'. Only PDF and HTML are accepted.",
        )

    except ValueError as exc:
        return DownloadResult(url=url, success=False, error=str(exc))
    except httpx.HTTPStatusError as exc:
        return DownloadResult(
            url=url,
            success=False,
            error=f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}",
        )
    except Exception as exc:
        return DownloadResult(url=url, success=False, error=str(exc))
