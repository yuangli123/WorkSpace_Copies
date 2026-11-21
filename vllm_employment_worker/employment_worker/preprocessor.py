from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image, ImageOps

from .types import PageImage, PreparedDocument


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
WORD_SUFFIXES = {".doc", ".docx"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES | WORD_SUFFIXES


def _secure(path: Path, directory: bool = False) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        pass


@dataclass(frozen=True)
class PreprocessOptions:
    cache_dir: Path
    pdf_dpi: int = 180
    max_pages_per_document: int = 0
    pdftoppm_bin: str = "pdftoppm"
    libreoffice_bin: str = ""
    pdf_password: str = ""
    pdf_password_file: str = ""
    workers: int = 4
    show_progress: bool = True


class PreprocessProgress:
    """适用于交互终端和重定向日志的轻量预处理进度条。"""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self.completed = 0
        self.cache_hits = 0
        self.failures = 0
        self.started = time.perf_counter()
        self.is_tty = sys.stderr.isatty()
        self.log_interval = max(1, total // 20) if total else 1

    def update(self, relative_path: str, cache_hit: bool, failed: bool) -> None:
        if not self.enabled:
            return
        self.completed += 1
        self.cache_hits += int(cache_hit)
        self.failures += int(failed)
        elapsed = max(time.perf_counter() - self.started, 0.001)
        rate = self.completed / elapsed
        generated = self.completed - self.cache_hits - self.failures
        if self.is_tty:
            width = 28
            filled = round(width * self.completed / self.total)
            bar = "█" * filled + "░" * (width - filled)
            name = relative_path.replace("\n", " ")[-42:]
            line = (
                f"\r预处理 [{bar}] {self.completed}/{self.total} "
                f"新建:{generated} 命中:{self.cache_hits} 失败:{self.failures} "
                f"{rate:.2f}文件/s  {name:<42}"
            )
            print(line, end="", file=sys.stderr, flush=True)
        elif self.completed % self.log_interval == 0 or self.completed == self.total:
            print(
                f"预处理 [{self.completed}/{self.total}] 新建:{generated} "
                f"命中:{self.cache_hits} 失败:{self.failures} {rate:.2f}文件/s",
                file=sys.stderr,
                flush=True,
            )

    def finish(self) -> None:
        if self.enabled and self.is_tty:
            print(file=sys.stderr, flush=True)


def _numeric_page_key(path: Path) -> tuple[int, str]:
    digits = "".join(char for char in path.stem if char.isdigit())
    return (int(digits) if digits else 0, path.name)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_files(input_path: Path, output_dir: Path) -> tuple[list[Path], list[Path]]:
    if input_path.is_file():
        return ([input_path], []) if input_path.suffix.lower() in SUPPORTED_SUFFIXES else ([], [input_path])
    supported: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or _is_inside(path, output_dir):
            continue
        (supported if path.suffix.lower() in SUPPORTED_SUFFIXES else skipped).append(path)
    return supported, skipped


def dependency_status(options: PreprocessOptions) -> dict[str, Any]:
    pdftoppm = _resolve_executable(options.pdftoppm_bin)
    libreoffice = _find_libreoffice(options.libreoffice_bin)
    try:
        import fitz  # type: ignore  # noqa: F401

        pymupdf = True
    except (ImportError, OSError):
        pymupdf = False
    return {
        "pdftoppm": pdftoppm,
        "pymupdf": pymupdf,
        "libreoffice": libreoffice,
        "pdf_supported": bool(pdftoppm or pymupdf),
        "doc_docx_supported": bool(libreoffice and (pdftoppm or pymupdf)),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
    }


def _resolve_executable(value: str) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve()) if Path(value).is_file() else shutil.which(value)


def _find_libreoffice(explicit: str) -> str | None:
    return _resolve_executable(explicit) or shutil.which("soffice") or shutil.which("libreoffice")


def _relative_path(source: Path, input_path: Path) -> str:
    return source.name if input_path.is_file() else source.relative_to(input_path).as_posix()


def _source_id(source: Path, relative_path: str, signature: str) -> str:
    stat = source.stat()
    content = f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}|{signature}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def _load_password_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pdf_password_file 必须是 JSON 对象")
    return {str(key): str(item) for key, item in value.items()}


class DocumentPreprocessor:
    def __init__(self, options: PreprocessOptions) -> None:
        self.options = options
        self.options.cache_dir.mkdir(parents=True, exist_ok=True)
        _secure(self.options.cache_dir, directory=True)
        self._word_lock = Lock()
        self._passwords = _load_password_map(options.pdf_password_file)

    def prepare_many(
        self,
        sources: list[Path],
        input_path: Path,
    ) -> tuple[list[PreparedDocument], list[dict[str, str]]]:
        documents: list[PreparedDocument] = []
        errors: list[dict[str, str]] = []
        progress = PreprocessProgress(len(sources), self.options.show_progress)
        with ThreadPoolExecutor(max_workers=self.options.workers) as executor:
            future_to_source = {
                executor.submit(self.prepare, source, input_path): source
                for source in sources
            }
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    document = future.result()
                    documents.append(document)
                    progress.update(
                        document.relative_path,
                        cache_hit=document.from_cache,
                        failed=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "source_file": str(source.resolve()),
                            "stage": "preprocess",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    progress.update(str(source), cache_hit=False, failed=True)
        progress.finish()
        documents.sort(key=lambda item: item.relative_path)
        errors.sort(key=lambda item: item["source_file"])
        self._write_archive_index(documents)
        return documents, errors

    def prepare(self, source: Path, input_path: Path) -> PreparedDocument:
        source = source.resolve()
        relative_path = _relative_path(source, input_path.resolve())
        signature = (
            f"vllm-worker-v2-archive|dpi={self.options.pdf_dpi}|"
            f"max_pages={self.options.max_pages_per_document}"
        )
        source_id = _source_id(source, relative_path, signature)
        source_dir = self.options.cache_dir / source_id
        page_dir = source_dir / "pages"
        marker = source_dir / "PREPROCESS_COMPLETE.json"
        cached_pages = self._read_cache(marker)
        if cached_pages:
            return self._make_document(source, relative_path, source_id, cached_pages, True)

        page_dir.mkdir(parents=True, exist_ok=True)
        _secure(source_dir, directory=True)
        _secure(page_dir, directory=True)
        for partial in page_dir.glob("page-*"):
            if partial.is_file():
                partial.unlink()
        if marker.exists():
            marker.unlink()

        suffix = source.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            pages = self._normalize_image(source, page_dir)
        elif suffix in PDF_SUFFIXES:
            pages = self._render_pdf(source, page_dir, self._password_for(source, relative_path))
        elif suffix in WORD_SUFFIXES:
            with self._word_lock:
                converted_pdf = self._convert_word(source, source_dir)
            pages = self._render_pdf(converted_pdf, page_dir, "")
        else:
            raise RuntimeError(f"不支持的文件格式：{suffix}")
        if not pages:
            raise RuntimeError("没有生成任何可推理页面")

        for page in pages:
            if _is_inside(page, self.options.cache_dir):
                _secure(page)

        marker.write_text(
            json.dumps(
                {
                    "source_file": str(source),
                    "relative_path": relative_path,
                    "source_id": source_id,
                    "page_paths": [str(path.resolve()) for path in pages],
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _secure(marker)
        return self._make_document(source, relative_path, source_id, pages, False)

    @staticmethod
    def _read_cache(marker: Path) -> list[Path]:
        if not marker.exists():
            return []
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            pages = [Path(item) for item in value.get("page_paths", [])]
            return pages if pages and all(path.exists() for path in pages) else []
        except (OSError, ValueError, TypeError):
            return []

    def _write_archive_index(self, documents: list[PreparedDocument]) -> None:
        """维护可审计的全局索引；每个 source_id 的页面和完成标记才是缓存真值。"""
        index_path = self.options.cache_dir / "archive_index.json"
        entries: dict[str, dict[str, Any]] = {}
        if index_path.exists():
            try:
                old = json.loads(index_path.read_text(encoding="utf-8"))
                entries = {
                    str(item["source_id"]): item
                    for item in old.get("entries", [])
                    if isinstance(item, dict) and item.get("source_id")
                }
            except (OSError, ValueError, TypeError):
                entries = {}
        for document in documents:
            entries[document.source_id] = {
                "source_id": document.source_id,
                "source_file": str(document.source_file),
                "relative_path": document.relative_path,
                "format": document.source_format,
                "page_count": len(document.pages),
                "page_paths": [str(page.image_path) for page in document.pages],
                "last_accessed_at": datetime.now().isoformat(timespec="seconds"),
            }
        payload = {
            "archive_version": 1,
            "archive_dir": str(self.options.cache_dir.resolve()),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": sorted(
                entries.values(), key=lambda item: str(item.get("relative_path", ""))
            ),
        }
        temporary = index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(index_path)
        _secure(index_path)

    @staticmethod
    def _make_document(
        source: Path,
        relative_path: str,
        source_id: str,
        pages: list[Path],
        from_cache: bool,
    ) -> PreparedDocument:
        return PreparedDocument(
            source_id=source_id,
            source_file=source,
            relative_path=relative_path,
            source_format=source.suffix.lower().lstrip("."),
            pages=tuple(
                PageImage(page_number=index, image_path=path.resolve())
                for index, path in enumerate(pages, start=1)
            ),
            from_cache=from_cache,
        )

    def _password_for(self, source: Path, relative_path: str) -> str:
        for key in (str(source), relative_path, source.name, source.stem):
            if key in self._passwords:
                return self._passwords[key]
        return self.options.pdf_password

    def _normalize_image(self, source: Path, page_dir: Path) -> list[Path]:
        pages: list[Path] = []
        with Image.open(source) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            if self.options.max_pages_per_document:
                frame_count = min(frame_count, self.options.max_pages_per_document)
            orientation = image.getexif().get(274, 1)
            direct_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
            if frame_count == 1 and source.suffix.lower() in direct_suffixes and orientation in (None, 1):
                output = page_dir / f"page-0001{source.suffix.lower()}"
                shutil.copy2(source, output)
                return [output]
            for index in range(frame_count):
                image.seek(index)
                output = page_dir / f"page-{index + 1:04d}.png"
                ImageOps.exif_transpose(image.copy()).convert("RGB").save(output, "PNG")
                pages.append(output)
        return pages

    def _render_pdf(self, source: Path, page_dir: Path, password: str) -> list[Path]:
        pages = self._render_pdf_pdftoppm(source, page_dir, password)
        if not pages:
            pages = self._render_pdf_pymupdf(source, page_dir, password)
        if not pages:
            raise RuntimeError("缺少 PDF 渲染器，请安装 pdftoppm 或 PyMuPDF")
        return pages

    def _render_pdf_pdftoppm(self, source: Path, page_dir: Path, password: str) -> list[Path]:
        executable = _resolve_executable(self.options.pdftoppm_bin)
        if not executable:
            return []
        command = [executable, "-png", "-r", str(self.options.pdf_dpi)]
        if password:
            command.extend(["-upw", password])
        if self.options.max_pages_per_document:
            command.extend(["-f", "1", "-l", str(self.options.max_pages_per_document)])
        command.extend([str(source), str(page_dir / "page")])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"pdftoppm 渲染失败：{message}")
        return sorted(page_dir.glob("page-*.png"), key=_numeric_page_key)

    def _render_pdf_pymupdf(self, source: Path, page_dir: Path, password: str) -> list[Path]:
        try:
            import fitz  # type: ignore
        except (ImportError, OSError):
            return []
        document = fitz.open(source)
        try:
            if document.needs_pass and not document.authenticate(password):
                raise RuntimeError("PDF 已加密，密码无效或未提供")
            count = len(document)
            if self.options.max_pages_per_document:
                count = min(count, self.options.max_pages_per_document)
            matrix = fitz.Matrix(self.options.pdf_dpi / 72.0, self.options.pdf_dpi / 72.0)
            pages = []
            for index in range(count):
                output = page_dir / f"page-{index + 1:04d}.png"
                document[index].get_pixmap(matrix=matrix, alpha=False).save(output)
                pages.append(output)
            return pages
        finally:
            document.close()

    def _convert_word(self, source: Path, source_dir: Path) -> Path:
        executable = _find_libreoffice(self.options.libreoffice_bin)
        if not executable:
            raise RuntimeError("处理 DOC/DOCX 需要 LibreOffice/soffice，但当前未找到")
        convert_dir = source_dir / "converted_pdf"
        profile_dir = source_dir / "libreoffice_profile"
        convert_dir.mkdir(parents=True, exist_ok=True)
        profile_dir.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
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
        raise RuntimeError("LibreOffice 已结束，但没有找到转换后的 PDF")
