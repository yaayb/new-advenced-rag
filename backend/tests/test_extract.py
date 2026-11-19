import io
import zipfile

import pytest

from evidencelab.extract import extract_text


def _docx(paragraphs: list[str]) -> bytes:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_docx_and_txt_extraction() -> None:
    assert "Hello world" in extract_text("a.docx", _docx(["Hello world", "Second paragraph"]))
    assert extract_text("a.md", b"# Title\nbody") .startswith("# Title")


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        extract_text("a.exe", b"x")
    with pytest.raises(ValueError):
        extract_text("a.docx", b"not a zip")
    with pytest.raises(ValueError):
        extract_text("a.pdf", b"not a pdf")
    with pytest.raises(ValueError):
        extract_text("a.txt", b"   ")
