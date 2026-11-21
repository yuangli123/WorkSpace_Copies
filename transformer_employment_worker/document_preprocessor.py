"""把图片、PDF、DOC/DOCX 统一转换为 Qwen3-VL 可读取的逐页 PNG。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
WORD_SUFFIXES = {".doc", ".docx"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES | WORD_SUFFIXES


@dataclass(frozen=True)
class PageTask:
    work_id: str
    source_id: str
    source_file: Path
    source_relative_path: str
    source_format: str
    page_number: int
    page_count: int
    image_path: Path
    from_cache: bool = False


@dataclass(frozen=True)
class PreprocessOptions:
    pdf_dpi: int
    max_pages_per_document: int
    pdf_password: str
    pdftoppm_bin: str
    libreoffice_bin: str


def numeric_page_key(path: Path) -> tuple[int, str]:
    digits = "".join(char for char in path.stem if char.isdigit())
    return (int(digits) if digits else 0, path.name)


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_files(input_path: Path, output_dir: Path) -> tuple[list[Path], list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_SUFFIXES:
            return [input_path], []
        return [], [input_path]
    supported: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or is_inside(path, output_dir):
            continue
        (supported if path.suffix.lower() in SUPPORTED_SUFFIXES else skipped).append(path)
    return supported, skipped


def load_password_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--pdf-password-file 必须是 JSON 对象")
    return {str(key): str(password) for key, password in value.items()}


def resolve_pdf_password(
    source: Path,
    relative_path: str,
    fallback: str,
    mapping: dict[str, str],
) -> str:
    for key in (str(source.resolve()), relative_path, source.name, source.stem):
        if key in mapping:
            return mapping[key]
    return fallback


def source_fingerprint(path: Path, relative_path: str, signature: str) -> str:
    stat = path.stat()
    payload = (
        f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}|{signature}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def normalize_image_pages(source: Path, page_dir: Path, max_pages: int) -> list[Path]:
    page_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    with Image.open(source) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if max_pages:
            frame_count = min(frame_count, max_pages)
        orientation = image.getexif().get(274, 1)
        if (
            frame_count == 1
            and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and orientation in (None, 1)
        ):
            # 常规单帧图片直接交给视觉处理器，避免重复编码为体积更大的PNG。
            return [source.resolve()]
        for index in range(frame_count):
            image.seek(index)
            frame = ImageOps.exif_transpose(image.copy()).convert("RGB")
            output = page_dir / f"page-{index + 1:04d}.png"
            frame.save(output, format="PNG")
            pages.append(output)
    return pages


def render_pdf_with_pdftoppm(
    source: Path,
    page_dir: Path,
    dpi: int,
    password: str,
    max_pages: int,
    executable: str,
) -> list[Path]:
    resolved = executable if Path(executable).is_file() else shutil.which(executable)
    if not resolved:
        return []
    page_dir.mkdir(parents=True, exist_ok=True)
    command = [str(resolved), "-png", "-r", str(dpi)]
    if password:
        command.extend(["-upw", password])
    if max_pages:
        command.extend(["-f", "1", "-l", str(max_pages)])
    command.extend([str(source), str(page_dir / "page")])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"pdftoppm 渲染失败：{message}")
    return sorted(page_dir.glob("page-*.png"), key=numeric_page_key)


def render_pdf_with_pymupdf(
    source: Path,
    page_dir: Path,
    dpi: int,
    password: str,
    max_pages: int,
) -> list[Path]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return []
    page_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(source)
    try:
        if document.needs_pass and not document.authenticate(password):
            raise RuntimeError("PDF 已加密且密码无效或未提供")
        page_count = min(len(document), max_pages) if max_pages else len(document)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pages: list[Path] = []
        for index in range(page_count):
            output = page_dir / f"page-{index + 1:04d}.png"
            document[index].get_pixmap(matrix=matrix, alpha=False).save(output)
            pages.append(output)
        return pages
    finally:
        document.close()


def render_pdf(
    source: Path,
    page_dir: Path,
    options: PreprocessOptions,
    password: str,
) -> list[Path]:
    pages = render_pdf_with_pdftoppm(
        source,
        page_dir,
        options.pdf_dpi,
        password,
        options.max_pages_per_document,
        options.pdftoppm_bin,
    )
    if not pages:
        pages = render_pdf_with_pymupdf(
            source,
            page_dir,
            options.pdf_dpi,
            password,
            options.max_pages_per_document,
        )
    if not pages:
        raise RuntimeError("缺少 PDF 渲染器：请安装 pdftoppm 或 PyMuPDF")
    return pages


def find_libreoffice(explicit: str) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else shutil.which(explicit)
    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_word_to_pdf(source: Path, work_dir: Path, explicit_bin: str) -> Path:
    executable = find_libreoffice(explicit_bin)
    if not executable:
        raise RuntimeError("处理 DOC/DOCX 需要 LibreOffice/soffice，但当前未找到")
    convert_dir = work_dir / "converted_pdf"
    profile_dir = work_dir / "libreoffice_profile"
    convert_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(convert_dir),
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"LibreOffice 转 PDF 失败：{message}")
    expected = convert_dir / f"{source.stem}.pdf"
    if expected.exists():
        return expected
    candidates = list(convert_dir.glob("*.pdf"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError("LibreOffice 执行结束，但未找到转换后的 PDF")


def relative_source_path(source: Path, input_path: Path) -> str:
    return source.name if input_path.is_file() else str(source.relative_to(input_path))


def prepare_source(
    source: Path,
    input_path: Path,
    render_root: Path,
    options: PreprocessOptions,
    password_map: dict[str, str],
) -> list[PageTask]:
    relative_path = relative_source_path(source, input_path)
    signature = (
        f"simple-worker-v1|dpi={options.pdf_dpi}|"
        f"max_pages={options.max_pages_per_document}"
    )
    source_id = source_fingerprint(source, relative_path, signature)
    source_dir = render_root / source_id
    page_dir = source_dir / "pages"
    marker = source_dir / "PREPROCESS_COMPLETE.json"
    pages: list[Path] = []
    if marker.exists():
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(marker_data, dict):
                raise ValueError("缓存标记不是JSON对象")
            cached_paths = [
                Path(value) for value in marker_data.get("page_paths", [])
            ]
            if cached_paths and all(path.exists() for path in cached_paths):
                pages = cached_paths
        except (OSError, ValueError, TypeError):
            pages = []
        if not pages:
            pages = sorted(page_dir.glob("page-*.png"), key=numeric_page_key)
    cache_hit = bool(pages)

    if not pages:
        page_dir.mkdir(parents=True, exist_ok=True)
        for partial in page_dir.glob("page-*.png"):
            partial.unlink()
        marker.unlink(missing_ok=True)
        suffix = source.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            pages = normalize_image_pages(
                source, page_dir, options.max_pages_per_document
            )
        elif suffix in PDF_SUFFIXES:
            password = resolve_pdf_password(
                source, relative_path, options.pdf_password, password_map
            )
            pages = render_pdf(source, page_dir, options, password)
        elif suffix in WORD_SUFFIXES:
            pdf = convert_word_to_pdf(source, source_dir, options.libreoffice_bin)
            pages = render_pdf(source=pdf, page_dir=page_dir, options=options, password="")
        else:
            raise RuntimeError(f"不支持的文件格式：{suffix}")

    if not pages:
        raise RuntimeError("文档未生成任何可推理页面")
    marker.write_text(
        json.dumps(
            {
                "source_file": str(source.resolve()),
                "page_count": len(pages),
                "page_paths": [str(page.resolve()) for page in pages],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    page_count = len(pages)
    return [
        PageTask(
            work_id=f"{source_id}:page:{index}",
            source_id=source_id,
            source_file=source.resolve(),
            source_relative_path=relative_path,
            source_format=source.suffix.lower().lstrip("."),
            page_number=index,
            page_count=page_count,
            image_path=page,
            from_cache=cache_hit,
        )
        for index, page in enumerate(pages, start=1)
    ]


def dependency_status(options: PreprocessOptions) -> dict[str, Any]:
    pdftoppm = (
        options.pdftoppm_bin
        if Path(options.pdftoppm_bin).is_file()
        else shutil.which(options.pdftoppm_bin)
    )
    try:
        import fitz  # type: ignore  # noqa: F401
        pymupdf = True
    except ImportError:
        pymupdf = False
    libreoffice = find_libreoffice(options.libreoffice_bin)
    return {
        "pdftoppm": pdftoppm,
        "pymupdf": pymupdf,
        "libreoffice": libreoffice,
        "pdf_supported": bool(pdftoppm or pymupdf),
        "docx_supported": bool(libreoffice and (pdftoppm or pymupdf)),
    }
