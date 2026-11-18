"""Resume text extraction for (almost) any file format.

Security notes:
- Resume content is UNTRUSTED. This module only extracts text; it never executes content.
- Office/ODF files are read with zipfile + regex (no XML parser) to avoid XML-bomb issues,
  and every archive member read is size-capped to defend against zip bombs.
- Nothing is written to disk. Files are processed in memory and discarded.
"""
from __future__ import annotations

import html
import io
import re
import zipfile
from dataclasses import dataclass

MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_TEXT_CHARS = 60_000
MIN_USEFUL_CHARS = 40

# Image files (screenshots, photos) are not supported: mock mode cannot read them and
# live mode is not a supported input path for this demo.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".svg"}
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*")
IMAGE_REJECTED_MESSAGE = "Image files (screenshots, photos) are not supported. Upload a PDF, DOCX or TXT file instead."


class ResumeParseError(ValueError):
    """Raised when a resume cannot be turned into usable text."""


@dataclass
class ParsedResume:
    filename: str
    text: str = ""
    detected_format: str = "text"

    @property
    def chars(self) -> int:
        return len(self.text)


def _ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""


def _strip_tags(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", markup)
    markup = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|text:p|text:h|a:p)>|<br\s*/?>|<text:line-break\s*/?>", "\n", markup)
    markup = re.sub(r"<[^>]+>", " ", markup)
    return html.unescape(markup)


def _read_member(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as fh:
        data = fh.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ResumeParseError("Archive member is too large to process safely.")
    return data.decode("utf-8", errors="replace")


def _parse_zip_office(data: bytes, ext: str) -> str:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ResumeParseError("File looks corrupted or is not a valid Office document.") from exc
    names = zf.namelist()
    parts: list[str] = []
    if ext == ".docx" or "word/document.xml" in names:
        targets = [n for n in names if n in ("word/document.xml",) or re.match(r"word/(header|footer)\d*\.xml", n)]
    elif ext == ".odt" or "content.xml" in names:
        targets = ["content.xml"] if "content.xml" in names else []
    elif ext == ".pptx" or any(n.startswith("ppt/slides/") for n in names):
        targets = sorted(n for n in names if re.match(r"ppt/slides/slide\d+\.xml", n))
    elif ext == ".xlsx" or "xl/sharedStrings.xml" in names:
        targets = [n for n in ("xl/sharedStrings.xml",) if n in names]
    else:
        targets = []
    for name in targets:
        parts.append(_strip_tags(_read_member(zf, name)))
    return "\n".join(parts)


def _parse_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # imported lazily so other formats work without it
    except ImportError as exc:  # pragma: no cover
        raise ResumeParseError("PDF support requires the 'pypdf' package.") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ResumeParseError("This PDF is password-protected. Please upload an unlocked copy.")
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n".join((page.extract_text() or "") for page in pages)
    except ResumeParseError:
        raise
    except Exception as exc:
        raise ResumeParseError("Could not read this PDF. It may be corrupted.") from exc


def _parse_rtf(data: bytes) -> str:
    raw = data.decode("latin-1", errors="replace")
    raw = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), raw)
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = re.sub(r"[{}]", "", raw)
    return raw


def _best_effort_strings(data: bytes) -> str:
    """Last resort for unknown/legacy binary formats (e.g. .doc): pull readable runs."""
    chunks = re.findall(rb"[\x20-\x7e\t\r\n]{6,}", data)
    return "\n".join(c.decode("ascii", errors="ignore") for c in chunks)


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127 or b >= 128)
    return printable / len(sample) > 0.95


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_image(ext: str, data: bytes, content_type: str) -> bool:
    return (
        ext in IMAGE_EXTENSIONS
        or content_type.lower().startswith("image/")
        or data.startswith(IMAGE_MAGIC)
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        or (data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"mif1"))
    )


def parse_resume(filename: str, data: bytes, content_type: str = "") -> ParsedResume:
    """Turn uploaded bytes into text. Image files are rejected."""
    if not data:
        raise ResumeParseError("The uploaded file is empty.")
    filename = (filename or "resume").strip()
    ext = _ext(filename)

    if _is_image(ext, data, content_type):
        raise ResumeParseError(IMAGE_REJECTED_MESSAGE)

    detected = ext.lstrip(".") or "unknown"
    if data[:4] == b"%PDF" or ext == ".pdf":
        text, detected = _parse_pdf(data), "pdf"
    elif data[:2] == b"PK" or ext in (".docx", ".odt", ".pptx", ".xlsx"):
        text = _parse_zip_office(data, ext)
        detected = ext.lstrip(".") or "office"
    elif data[:5] == b"{\\rtf" or ext == ".rtf":
        text, detected = _parse_rtf(data), "rtf"
    elif ext in (".html", ".htm", ".xml") or data[:200].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        text, detected = _strip_tags(data.decode("utf-8", errors="replace")), "html"
    elif _looks_like_text(data):
        text, detected = data.decode("utf-8", errors="replace"), ext.lstrip(".") or "text"
    else:
        text, detected = _best_effort_strings(data), f"{detected} (best-effort)"

    text = _normalize(text)[:MAX_TEXT_CHARS]
    if len(text) < MIN_USEFUL_CHARS:
        raise ResumeParseError(
            "Could not find readable text in this file. If it is a scanned document, "
            "upload a text-based copy (PDF with a text layer, DOCX or TXT)."
        )
    return ParsedResume(filename=filename, text=text, detected_format=detected)
