"""Discover uploaded blank forms and the fields needed for a draft."""
import hashlib
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from olefile import OleFileIO

ROOT = Path(__file__).resolve().parent.parent
FORMS_DIR = Path(os.getenv("POLICY_FORMS_DIR", str(ROOT / "data" / "forms")))
EXTENSIONS = {".hwp", ".hwpx", ".docx"}
MAX_FORM_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 60 * 1024 * 1024
PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")

# These labels are only requested when they actually appear in the uploaded form.
KNOWN_FIELDS = (
    ("student_name", "성명", ("성명", "이름")),
    ("student_id", "학번", ("학번",)),
    ("birth_date", "생년월일", ("생년월일",)),
    ("phone", "연락처", ("연락처", "전화번호")),
    ("department", "소속 학부(과)", ("소속학부(과)", "소속학과", "소속학부")),
    ("completed_semesters", "이수학기", ("이수학기",)),
    ("current_major", "제1전공", ("제1전공", "주전공")),
    ("requested_department", "신청 학부(과)", ("부전공신청학부(과)", "신청학부(과)")),
    ("requested_major", "신청 전공", ("부전공명", "신청전공")),
    ("earned_credits", "총 취득학점", ("총취득학점",)),
    ("gpa", "평균 평점", ("평균평점",)),
    ("course_plan", "신청 과목", ("신청과목명", "신청과목")),
)


def _safe_zip(path: Path) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(path)
    if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED_BYTES:
        archive.close()
        raise ValueError("양식 압축 파일이 너무 큽니다.")
    return archive


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".hwp":
        with OleFileIO(path) as ole:
            if not ole.exists("PrvText"):
                return ""
            return ole.openstream("PrvText").read().decode("utf-16le", errors="replace")
    with _safe_zip(path) as archive:
        names = archive.namelist()
        prefix = "word/" if path.suffix.lower() == ".docx" else "Contents/"
        texts = []
        for name in names:
            if not name.startswith(prefix) or not name.endswith(".xml"):
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError:
                continue
            texts.extend(node.text or "" for node in root.iter()
                         if node.tag.rsplit("}", 1)[-1] in {"t", "text"})
        return "\n".join(texts)


def _read_manifest(path: Path) -> dict:
    sidecar = path.with_name(path.name + ".form.json")
    if not sidecar.exists():
        return {}
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("양식 설정 파일은 JSON 객체여야 합니다.")
    return value


def _fields(path: Path, text: str, manifest: dict) -> list[dict]:
    configured = manifest.get("fields")
    if configured is not None:
        if not isinstance(configured, list):
            raise ValueError("fields는 배열이어야 합니다.")
        return [{"key": str(item["key"]), "label": str(item.get("label", item["key"])),
                 "required": bool(item.get("required", True)),
                 "multiline": bool(item.get("multiline", False))} for item in configured]
    placeholders = set(PLACEHOLDER.findall(text))
    condensed = re.sub(r"[\s<>]+", "", text)
    fields = []
    for key, label, aliases in KNOWN_FIELDS:
        if key in placeholders or any(re.sub(r"\s+", "", alias) in condensed for alias in aliases):
            fields.append({"key": key, "label": label, "required": True,
                           "multiline": key == "course_plan"})
    for key in sorted(placeholders - {field["key"] for field in fields}):
        fields.append({"key": key, "label": key.replace("_", " "), "required": True,
                       "multiline": False})
    return fields


def _replaceable_placeholders(path: Path) -> set[str]:
    if path.suffix.lower() not in {".hwpx", ".docx"}:
        return set()
    markers = set()
    with _safe_zip(path) as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                markers.update(PLACEHOLDER.findall(archive.read(name).decode("utf-8", errors="replace")))
    return markers


def describe_form(path: Path, root: Path | None = None) -> dict:
    root = root or FORMS_DIR
    path = path.resolve()
    relative = path.relative_to(root.resolve())
    if len(relative.parts) < 2 or relative.parts[0] not in {"academic", "corporate"}:
        raise ValueError("양식은 academic 또는 corporate 폴더에 있어야 합니다.")
    if path.suffix.lower() not in EXTENSIONS or path.stat().st_size > MAX_FORM_BYTES:
        raise ValueError("지원하지 않는 양식이거나 파일이 너무 큽니다.")
    text = _extract_text(path)
    manifest = _read_manifest(path)
    fields = _fields(path, text, manifest)
    placeholders = _replaceable_placeholders(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    title = str(manifest.get("title") or path.stem)
    return {"id": hashlib.sha256(str(relative).encode()).hexdigest()[:20],
            "title": title, "filename": path.name, "domain": relative.parts[0],
            "format": path.suffix.lower()[1:], "source": str(relative),
            "source_url": str(manifest.get("source_url") or ""),
            "version": str(manifest.get("version") or ""),
            "content_hash": digest, "fields": fields,
            "draft_format": path.suffix.lower()[1:] if fields and
                            all(field["key"] in placeholders for field in fields) and
                            path.suffix.lower() in {".hwpx", ".docx"} else "docx",
            "fills_original_layout": bool(fields and
                                          all(field["key"] in placeholders for field in fields) and
                                          path.suffix.lower() in {".hwpx", ".docx"})}


def list_forms(domain: str | None = None, root: Path | None = None) -> list[dict]:
    root = root or FORMS_DIR
    if domain is not None and domain not in {"academic", "corporate"}:
        raise ValueError("지원하지 않는 분야입니다.")
    domains = (domain,) if domain else ("academic", "corporate")
    forms = []
    for name in domains:
        for path in sorted((root / name).rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                try:
                    forms.append(describe_form(path, root))
                except (ValueError, OSError, zipfile.BadZipFile):
                    continue
    return forms


def search_forms(query: str, domain: str = "academic", root: Path | None = None) -> list[dict]:
    root = root or FORMS_DIR
    normalized_query = unicodedata.normalize("NFC", query).lower()
    words = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", normalized_query))
    for word in tuple(words):
        stem = re.sub(r"(?:에서는|에게는|으로는|에서|에게|으로|은|는|이|가|을|를|의|도|에|로)$", "", word)
        if len(stem) >= 2:
            words.add(stem)
    matches = []
    for form in list_forms(domain, root):
        title = unicodedata.normalize("NFC", form["title"]).lower()
        score = sum(2 for word in words if word in title)
        if "부전공" in normalized_query and "부전공" in title:
            score += 4
        if "복수전공" in normalized_query and "복수전공" in title:
            score += 4
        if score:
            matches.append((score, form))
    return [form for _, form in sorted(matches, key=lambda pair: (-pair[0], pair[1]["title"]))[:5]]


def form_path(form_id: str, root: Path | None = None) -> Path:
    root = root or FORMS_DIR
    for form in list_forms(root=root):
        if form["id"] == form_id:
            return (root / form["source"]).resolve()
    raise KeyError(form_id)
