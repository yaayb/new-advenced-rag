"""Text extraction for uploaded files (PDF, DOCX, TXT, MD)."""

import io
import re
import zipfile
from xml.etree import ElementTree

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_text(filename: str, raw: bytes) -> str:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("le fichier dépasse la limite de 50 Mo")
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            raise ValueError(f"lecture du PDF impossible : {e}") from e
    elif name.endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                root = ElementTree.fromstring(z.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
            raise ValueError(f"lecture du DOCX impossible : {e}") from e
        paras = ["".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p")]
        text = "\n\n".join(p for p in paras if p.strip())
    elif name.endswith((".txt", ".md", ".markdown", ".csv")):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("type de fichier non pris en charge (PDF, DOCX, TXT ou MD)")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise ValueError("aucun texte extractible (les PDF scannés nécessitent un OCR)")
    return text
