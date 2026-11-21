from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REQUIRED_SOURCE_TABLES = {"graduates", "evidence_files", "graduate_evidence"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mask_identity(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "*" * max(0, len(text) - 4) + text[-4:]


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def _chunks(values: list[int], size: int = 500) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SourceRepository:
    """以只读 URI 访问毕业生业务数据库。"""

    def __init__(self, database: Path, evidence_root: Path) -> None:
        self.database = database.resolve()
        self.evidence_root = evidence_root.resolve()
        self.verify_schema()
        self._graduate_columns = self._table_columns("graduates")
        self._province_column = next(
            (
                name
                for name in (
                    "school_province",
                    "xxszsf",
                    "province",
                    "sfmc",
                    "shengfen",
                )
                if name in self._graduate_columns
            ),
            None,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def verify_schema(self) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing = sorted(REQUIRED_SOURCE_TABLES - tables)
            if missing:
                raise RuntimeError(f"业务数据库缺少表：{', '.join(missing)}")
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(f"业务数据库完整性检查失败：{integrity}")
            foreign_key_issues = int(
                connection.execute("SELECT COUNT(*) FROM pragma_foreign_key_check").fetchone()[0]
            )
            if foreign_key_issues:
                raise RuntimeError(f"业务数据库存在 {foreign_key_issues} 条外键异常")
            return {
                "sqlite_version": sqlite3.sqlite_version,
                "tables": sorted(tables),
                "integrity": integrity,
                "foreign_key_issues": foreign_key_issues,
            }

    def _table_columns(self, table: str) -> set[str]:
        if table not in REQUIRED_SOURCE_TABLES:
            raise ValueError(f"不允许读取未知表结构：{table}")
        with closing(self._connect()) as connection, connection:
            return {
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }

    def existing_graduate_ids(self, ids: list[int]) -> set[int]:
        if not ids:
            return set()
        with closing(self._connect()) as connection, connection:
            found: set[int] = set()
            for values in _chunks(ids):
                rows = connection.execute(
                    f"SELECT id FROM graduates WHERE id IN ({_placeholders(values)})",
                    values,
                )
                found.update(int(row[0]) for row in rows)
            return found

    def existing_evidence_ids(self, ids: list[int]) -> set[int]:
        if not ids:
            return set()
        with closing(self._connect()) as connection, connection:
            found: set[int] = set()
            for values in _chunks(ids):
                rows = connection.execute(
                    f"SELECT id FROM evidence_files WHERE id IN ({_placeholders(values)})",
                    values,
                )
                found.update(int(row[0]) for row in rows)
            return found

    def dashboard_overview(self) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            graduate_count = int(
                connection.execute("SELECT COUNT(*) FROM graduates").fetchone()[0]
            )
            evidence_count = int(
                connection.execute("SELECT COUNT(*) FROM evidence_files").fetchone()[0]
            )
            linked_count = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT evidence_file_id) FROM graduate_evidence"
                ).fetchone()[0]
            )
            graduates_with_evidence = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT graduate_id) FROM graduate_evidence"
                ).fetchone()[0]
            )
            province_count = (
                int(
                    connection.execute(
                        f"""
                        SELECT COUNT(DISTINCT {self._province_column}) FROM graduates
                        WHERE {self._province_column} IS NOT NULL
                          AND TRIM({self._province_column})<>''
                        """
                    ).fetchone()[0]
                )
                if self._province_column
                else 0
            )
            school_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT xxmc) FROM graduates
                    WHERE xxmc IS NOT NULL AND TRIM(xxmc)<>''
                    """
                ).fetchone()[0]
            )
            graduation_cohort_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT bynf) FROM graduates
                    WHERE bynf IS NOT NULL AND TRIM(bynf)<>''
                    """
                ).fetchone()[0]
            )
            categories = [
                {"name": str(row[0]), "count": int(row[1])}
                for row in connection.execute(
                    """
                    SELECT category,COUNT(*) FROM evidence_files
                    GROUP BY category ORDER BY COUNT(*) DESC,category
                    """
                )
            ]
            formats = [
                {"name": str(row[0] or "未知").lstrip(".").upper(), "count": int(row[1])}
                for row in connection.execute(
                    """
                    SELECT extension,COUNT(*) FROM evidence_files
                    GROUP BY extension ORDER BY COUNT(*) DESC,extension
                    """
                )
            ]
            return {
                "graduate_count": graduate_count,
                "evidence_count": evidence_count,
                "linked_evidence_count": linked_count,
                "unmatched_evidence_count": evidence_count - linked_count,
                "graduates_with_evidence_count": graduates_with_evidence,
                "graduates_without_evidence_count": graduate_count - graduates_with_evidence,
                "province_count": province_count,
                "province_available": bool(self._province_column),
                "school_count": school_count,
                "graduation_cohort_count": graduation_cohort_count,
                "categories": categories,
                "formats": formats,
            }

    def graduate_filter_options(self) -> dict[str, Any]:
        def distinct_values(connection: sqlite3.Connection, column: str) -> list[str]:
            if column not in self._graduate_columns:
                return []
            return [
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT {column} FROM graduates
                    WHERE {column} IS NOT NULL AND TRIM({column})<>''
                    ORDER BY {column}
                    """
                )
            ]

        with closing(self._connect()) as connection, connection:
            provinces = (
                distinct_values(connection, self._province_column)
                if self._province_column
                else []
            )
            destinations = distinct_values(connection, "byqxlb")
            return {
                "province_available": bool(self._province_column),
                "province_source_column": self._province_column or "",
                "provinces": provinces,
                "graduation_years": distinct_values(connection, "bynf"),
                "schools": distinct_values(connection, "xxmc"),
                "colleges": distinct_values(connection, "xymc"),
                "destinations": destinations,
                # 兼容旧前端；实际值已经改为毕业生表中的毕业去向。
                "material_categories": destinations,
            }

    def list_graduates(
        self,
        page: int,
        page_size: int,
        province: str = "",
        graduation_year: str = "",
        school: str = "",
        college: str = "",
        name: str = "",
        student_id: str = "",
        destination: str = "",
        evidence_status: str = "all",
        verification_status: str = "all",
        verification_statuses: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        filters: list[str] = []
        params: list[Any] = []

        def exact(column: str, value: str) -> None:
            value = value.strip()
            if value:
                filters.append(f"g.{column}=?")
                params.append(value)

        exact("bynf", graduation_year)
        exact("xxmc", school)
        exact("xymc", college)
        if province.strip() and self._province_column:
            filters.append(f"g.{self._province_column}=?")
            params.append(province.strip())
        if name.strip():
            filters.append("COALESCE(g.xm,'') LIKE ?")
            params.append(f"%{name.strip()}%")
        if student_id.strip():
            filters.append("COALESCE(g.xh,'') LIKE ?")
            params.append(f"%{student_id.strip()}%")
        exact("byqxlb", destination)
        if evidence_status == "with":
            filters.append(
                "EXISTS(SELECT 1 FROM graduate_evidence gew WHERE gew.graduate_id=g.id)"
            )
        elif evidence_status == "without":
            filters.append(
                "NOT EXISTS(SELECT 1 FROM graduate_evidence gew WHERE gew.graduate_id=g.id)"
            )
        status_map = verification_statuses or {}
        if verification_status in {"pass", "review", "fail"}:
            matching_ids = sorted(
                graduate_id
                for graduate_id, value in status_map.items()
                if value == verification_status
            )
            filters.append(
                f"g.id IN ({','.join(str(value) for value in matching_ids)})"
                if matching_ids
                else "0=1"
            )
        elif verification_status == "unverified":
            checked_ids = sorted(status_map)
            if checked_ids:
                filters.append(
                    f"g.id NOT IN ({','.join(str(value) for value in checked_ids)})"
                )

        clause = " AND ".join(filters) if filters else "1=1"
        province_select = (
            f"COALESCE(g.{self._province_column},'')"
            if self._province_column
            else "''"
        )
        base_sql = f"""
            FROM graduates g
            LEFT JOIN (
                SELECT graduate_id,COUNT(DISTINCT evidence_file_id) AS evidence_count
                FROM graduate_evidence GROUP BY graduate_id
            ) ec ON ec.graduate_id=g.id
            WHERE {clause}
        """

        with closing(self._connect()) as connection, connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT g.id AS graduate_id,g.xh,g.xm,g.xxmc,g.xymc,g.zymc,
                       g.bynf,g.byqxlb,{province_select} AS province,
                       COALESCE(ec.evidence_count,0) AS evidence_count
                {base_sql}
                ORDER BY CASE WHEN COALESCE(ec.evidence_count,0)>0 THEN 0 ELSE 1 END,
                         g.xxmc,g.xymc,g.xh,g.id
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            items = [
                {
                    "graduate_id": int(row["graduate_id"]),
                    "xh": str(row["xh"] or ""),
                    "xm": str(row["xm"] or ""),
                    "province": str(row["province"] or ""),
                    "xxmc": str(row["xxmc"] or ""),
                    "xymc": str(row["xymc"] or ""),
                    "zymc": str(row["zymc"] or ""),
                    "bynf": str(row["bynf"] or ""),
                    "byqxlb": str(row["byqxlb"] or ""),
                    "evidence_count": int(row["evidence_count"] or 0),
                    "verification_status": status_map.get(
                        int(row["graduate_id"]), "unverified"
                    ),
                    "evidences": [],
                }
                for row in rows
            ]
            graduate_ids = [item["graduate_id"] for item in items]
            if graduate_ids:
                evidence_rows = connection.execute(
                    f"""
                    SELECT ge.graduate_id,ef.id AS evidence_file_id,ef.file_name,
                           ef.category,ef.extension,ef.file_size
                    FROM graduate_evidence ge
                    JOIN evidence_files ef ON ef.id=ge.evidence_file_id
                    WHERE ge.graduate_id IN ({_placeholders(graduate_ids)})
                    ORDER BY ge.graduate_id,ef.id
                    """,
                    graduate_ids,
                ).fetchall()
                by_graduate = {graduate_id: [] for graduate_id in graduate_ids}
                for row in evidence_rows:
                    extension = str(row["extension"] or "")
                    by_graduate[int(row["graduate_id"])].append(
                        {
                            "evidence_file_id": int(row["evidence_file_id"]),
                            "file_name": str(row["file_name"] or ""),
                            "category": str(row["category"] or ""),
                            "extension": extension,
                            "file_size": int(row["file_size"] or 0),
                            "preview_supported": extension.lower().lstrip(".")
                            in {
                                "jpg", "jpeg", "png", "webp", "bmp",
                                "tif", "tiff", "pdf",
                            },
                        }
                    )
                for item in items:
                    item["evidences"] = by_graduate[item["graduate_id"]]

            return {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size,
                "items": items,
                "province_available": bool(self._province_column),
            }

    def list_graduate_selection(
        self,
        province: str = "",
        graduation_year: str = "",
        school: str = "",
        college: str = "",
        name: str = "",
        student_id: str = "",
        destination: str = "",
        evidence_status: str = "all",
        verification_status: str = "all",
        verification_statuses: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """一次返回创建任务所需的轻量毕业生数据，避免前端逐页抓取。"""
        filters: list[str] = []
        params: list[Any] = []

        def exact(column: str, value: str) -> None:
            if value.strip():
                filters.append(f"g.{column}=?")
                params.append(value.strip())

        exact("bynf", graduation_year)
        exact("xxmc", school)
        exact("xymc", college)
        exact("byqxlb", destination)
        if province.strip() and self._province_column:
            filters.append(f"g.{self._province_column}=?")
            params.append(province.strip())
        if name.strip():
            filters.append("COALESCE(g.xm,'') LIKE ?")
            params.append(f"%{name.strip()}%")
        if student_id.strip():
            filters.append("COALESCE(g.xh,'') LIKE ?")
            params.append(f"%{student_id.strip()}%")
        if evidence_status == "with":
            filters.append(
                "EXISTS(SELECT 1 FROM graduate_evidence gew WHERE gew.graduate_id=g.id)"
            )
        elif evidence_status == "without":
            filters.append(
                "NOT EXISTS(SELECT 1 FROM graduate_evidence gew WHERE gew.graduate_id=g.id)"
            )

        status_map = verification_statuses or {}
        if verification_status in {"pass", "review", "fail"}:
            matching_ids = sorted(
                graduate_id
                for graduate_id, value in status_map.items()
                if value == verification_status
            )
            filters.append(
                f"g.id IN ({','.join(str(value) for value in matching_ids)})"
                if matching_ids
                else "0=1"
            )
        elif verification_status == "unverified" and status_map:
            filters.append(
                f"g.id NOT IN ({','.join(str(value) for value in sorted(status_map))})"
            )

        clause = " AND ".join(filters) if filters else "1=1"
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                f"""
                SELECT g.id AS graduate_id,g.xh,g.xm,g.xxmc,g.xymc,g.zymc,
                       g.bynf,g.byqxlb,
                       EXISTS(
                           SELECT 1 FROM graduate_evidence ge
                           WHERE ge.graduate_id=g.id
                       ) AS has_evidence
                FROM graduates g
                WHERE {clause}
                ORDER BY g.xxmc,g.xymc,g.xh,g.id
                """,
                params,
            ).fetchall()
        items = [
            {
                "graduate_id": int(row["graduate_id"]),
                "xh": str(row["xh"] or ""),
                "xm": str(row["xm"] or ""),
                "xxmc": str(row["xxmc"] or ""),
                "xymc": str(row["xymc"] or ""),
                "zymc": str(row["zymc"] or ""),
                "bynf": str(row["bynf"] or ""),
                "byqxlb": str(row["byqxlb"] or ""),
                "evidence_count": 1 if row["has_evidence"] else 0,
                "verification_status": status_map.get(
                    int(row["graduate_id"]), "unverified"
                ),
                "evidences": [],
            }
            for row in rows
        ]
        return {"total": len(items), "items": items}

    def list_evidences(
        self,
        page: int,
        page_size: int,
        keyword: str = "",
        category: str = "",
        link_status: str = "all",
    ) -> dict[str, Any]:
        where: list[str] = []
        params: list[Any] = []
        keyword = keyword.strip()
        category = category.strip()
        if keyword:
            pattern = f"%{keyword}%"
            where.append(
                """
                (ef.file_name LIKE ? OR ef.category LIKE ? OR
                 COALESCE(g.xh,'') LIKE ? OR COALESCE(g.xm,'') LIKE ? OR
                 COALESCE(g.byqxlb,'') LIKE ?)
                """
            )
            params.extend([pattern] * 5)
        if category:
            where.append("ef.category=?")
            params.append(category)
        if link_status == "linked":
            where.append("COALESCE(ls.link_count,0)>0")
        elif link_status == "unmatched":
            where.append("COALESCE(ls.link_count,0)=0")
        clause = " AND ".join(where) if where else "1=1"
        base_sql = f"""
            FROM evidence_files ef
            LEFT JOIN (
                SELECT evidence_file_id,COUNT(*) AS link_count,
                       MIN(graduate_id) AS first_graduate_id
                FROM graduate_evidence GROUP BY evidence_file_id
            ) ls ON ls.evidence_file_id=ef.id
            LEFT JOIN graduates g ON g.id=ls.first_graduate_id
            WHERE {clause}
        """
        with closing(self._connect()) as connection, connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT ef.id AS evidence_file_id,ef.file_name,ef.category,
                       ef.extension,ef.mime_group,ef.file_size,
                       COALESCE(ls.link_count,0) AS link_count,
                       g.id AS graduate_id,g.xh,g.xm,g.byqxlb
                {base_sql}
                ORDER BY ef.id
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            items = [
                {
                    "evidence_file_id": int(row["evidence_file_id"]),
                    "file_name": str(row["file_name"]),
                    "category": str(row["category"]),
                    "extension": str(row["extension"] or ""),
                    "mime_group": str(row["mime_group"] or ""),
                    "file_size": int(row["file_size"] or 0),
                    "link_status": "linked" if int(row["link_count"]) else "unmatched",
                    "link_count": int(row["link_count"]),
                    "graduate_id": row["graduate_id"],
                    "xh": str(row["xh"] or ""),
                    "xm": str(row["xm"] or ""),
                    "byqxlb": str(row["byqxlb"] or ""),
                    "preview_supported": str(row["extension"] or "").lower().lstrip(".")
                    in {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "pdf"},
                }
                for row in rows
            ]
            return {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size,
                "items": items,
            }

    def evidence_file(self, evidence_id: int) -> dict[str, Any] | None:
        values = self.evidences([evidence_id])
        return values[0] if values else None

    def selected_contexts(
        self,
        graduate_ids: list[int],
        evidence_file_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not graduate_ids and not evidence_file_ids:
            return []
        graduate_groups = list(_chunks(graduate_ids)) if graduate_ids else [[]]
        evidence_groups = list(_chunks(evidence_file_ids)) if evidence_file_ids else [[]]
        values: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        with closing(self._connect()) as connection, connection:
            for graduate_group in graduate_groups:
                for evidence_group in evidence_groups:
                    where = []
                    params: list[Any] = []
                    if graduate_group:
                        where.append(
                            f"g.id IN ({_placeholders(graduate_group)})"
                        )
                        params.extend(graduate_group)
                    if evidence_group:
                        where.append(
                            f"ef.id IN ({_placeholders(evidence_group)})"
                        )
                        params.extend(evidence_group)
                    clause = " AND ".join(where)
                    sql = f"""
            SELECT
                g.id AS graduate_id, g.xh, g.xm, g.sfzh, g.xxmc, g.xymc,
                g.zymc, g.bynf, g.byrq, g.byqxlb, g.dwmc, g.zzjgdm,
                g.dwxz, g.hylb, g.dwszd, g.dwdz, g.dwlxr, g.dwdh,
                g.qysj, g.bdzqfrq,
                ef.id AS evidence_file_id, ef.file_name, ef.relative_path,
                ef.category, ef.mime_group, ef.extension, ef.md5,
                ge.match_field, ge.match_value, ge.match_category,
                ge.match_confidence, ge.match_note
            FROM graduate_evidence ge
            JOIN graduates g ON g.id=ge.graduate_id
            JOIN evidence_files ef ON ef.id=ge.evidence_file_id
            WHERE {clause}
            ORDER BY g.id, ef.id
                    """
                    for row in connection.execute(sql, params):
                        key = (int(row["graduate_id"]), int(row["evidence_file_id"]))
                        if key not in seen:
                            seen.add(key)
                            values.append(self._context_from_row(row))
        values.sort(
            key=lambda item: (
                int(item["graduate_id"]),
                int(item["evidence"]["evidence_file_id"]),
            )
        )
        return values

    def low_confidence_contexts(
        self,
        graduate_ids: list[int],
        evidence_file_ids: list[int],
    ) -> list[dict[str, Any]]:
        """读取 review 表中明确带 graduate_id 的候选；无法确定身份的候选不猜测。"""
        with closing(self._connect()) as connection, connection:
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_match_review'"
            ).fetchone()
            if not table_exists:
                return []
            rows = connection.execute(
                """
                SELECT evidence_file_id,candidate_json,note
                FROM evidence_match_review
                WHERE candidate_count>0 AND candidate_json IS NOT NULL
                """
            ).fetchall()
            pairs: dict[tuple[int, int], dict[str, Any]] = {}
            for row in rows:
                evidence_id = int(row["evidence_file_id"])
                if evidence_file_ids and evidence_id not in evidence_file_ids:
                    continue
                try:
                    candidates = json.loads(row["candidate_json"])
                except (TypeError, ValueError):
                    continue
                if isinstance(candidates, dict):
                    candidates = candidates.get("candidates", [])
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    raw_id = candidate.get("graduate_id") or candidate.get("id")
                    try:
                        graduate_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if graduate_ids and graduate_id not in graduate_ids:
                        continue
                    pairs[(graduate_id, evidence_id)] = {
                        "match_value": str(candidate.get("match_value", "")),
                        "match_confidence": str(candidate.get("confidence", "low")),
                        "match_note": str(row["note"] or "低置信候选关联"),
                    }

            values: list[dict[str, Any]] = []
            for (graduate_id, evidence_id), match in sorted(pairs.items()):
                row = connection.execute(
                    """
                    SELECT
                        g.id AS graduate_id, g.xh, g.xm, g.sfzh, g.xxmc, g.xymc,
                        g.zymc, g.bynf, g.byrq, g.byqxlb, g.dwmc, g.zzjgdm,
                        g.dwxz, g.hylb, g.dwszd, g.dwdz, g.dwlxr, g.dwdh,
                        g.qysj, g.bdzqfrq,
                        ef.id AS evidence_file_id, ef.file_name, ef.relative_path,
                        ef.category, ef.mime_group, ef.extension, ef.md5
                    FROM graduates g CROSS JOIN evidence_files ef
                    WHERE g.id=? AND ef.id=?
                    """,
                    (graduate_id, evidence_id),
                ).fetchone()
                if row is None:
                    continue
                raw = dict(row)
                raw.update(
                    {
                        "match_field": "candidate_json",
                        "match_category": "low_confidence_candidate",
                        **match,
                    }
                )
                values.append(self._context_from_row(raw))
            return values

    def graduates(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows: list[sqlite3.Row] = []
        with closing(self._connect()) as connection, connection:
            for values in _chunks(ids):
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT id AS graduate_id, xh, xm, sfzh, xxmc, xymc, zymc,
                               bynf, byrq, byqxlb, dwmc, zzjgdm, dwxz, hylb,
                               dwszd, dwdz, dwlxr, dwdh, qysj, bdzqfrq
                        FROM graduates
                        WHERE id IN ({_placeholders(values)}) ORDER BY id
                        """,
                        values,
                    ).fetchall()
                )
        rows.sort(key=lambda row: int(row["graduate_id"]))
        return [self._graduate_context(dict(row)) for row in rows]

    def evidences(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows: list[sqlite3.Row] = []
        with closing(self._connect()) as connection, connection:
            for values in _chunks(ids):
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT id AS evidence_file_id, file_name, relative_path,
                               category, mime_group, extension, md5
                        FROM evidence_files
                        WHERE id IN ({_placeholders(values)}) ORDER BY id
                        """,
                        values,
                    ).fetchall()
                )
        rows.sort(key=lambda row: int(row["evidence_file_id"]))
        return [self._evidence_context(dict(row)) for row in rows]

    def _context_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        raw = dict(row)
        context = self._graduate_context(raw)
        context["evidence"] = self._evidence_context(raw)
        match_field = str(raw.get("match_field") or "")
        match_value = str(raw.get("match_value") or "")
        if match_field.lower() in {"sfzh", "id_card", "identity_card"}:
            match_value = mask_identity(match_value)
        context["evidence"].update(
            {
                "match_field": match_field,
                "match_value": match_value,
                "match_category": raw.get("match_category") or "",
                "match_confidence": raw.get("match_confidence") or "",
                "match_note": raw.get("match_note") or "",
            }
        )
        return context

    @staticmethod
    def _graduate_context(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "graduate_id": int(raw["graduate_id"]),
            "student": {
                "xh": raw.get("xh") or "",
                "xm": raw.get("xm") or "",
                "sfzh_masked": mask_identity(raw.get("sfzh")),
                "xxmc": raw.get("xxmc") or "",
                "xymc": raw.get("xymc") or "",
                "zymc": raw.get("zymc") or "",
                "bynf": raw.get("bynf") or "",
                "byrq": raw.get("byrq") or "",
                "byqxlb": raw.get("byqxlb") or "",
            },
            "employment": {
                "dwmc": raw.get("dwmc") or "",
                "zzjgdm": raw.get("zzjgdm") or "",
                "dwxz": raw.get("dwxz") or "",
                "hylb": raw.get("hylb") or "",
                "dwszd": raw.get("dwszd") or "",
                "dwdz": raw.get("dwdz") or "",
                "dwlxr": raw.get("dwlxr") or "",
                "dwdh": raw.get("dwdh") or "",
                "qysj": raw.get("qysj") or "",
                "bdzqfrq": raw.get("bdzqfrq") or "",
            },
        }

    def _evidence_context(self, raw: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(raw.get("relative_path") or "")
        relative = Path(relative_path)
        candidates = [(self.evidence_root / relative).resolve()]
        # 兼容 evidence_root 指向 data 根目录，或直接指向材料目录两种部署方式。
        if relative.parts and relative.parts[0] == self.evidence_root.name:
            candidates.insert(
                0,
                (self.evidence_root.joinpath(*relative.parts[1:])).resolve(),
            )
        file_path = next((path for path in candidates if path.is_file()), candidates[0])
        try:
            file_path.relative_to(self.evidence_root)
        except ValueError as exc:
            raise RuntimeError(f"材料路径越出根目录：{relative_path}") from exc
        return {
            "evidence_file_id": int(raw["evidence_file_id"]),
            "file_name": raw.get("file_name") or "",
            "relative_path": relative_path,
            "file_path": str(file_path),
            "category": raw.get("category") or "",
            "mime_group": raw.get("mime_group") or "",
            "extension": raw.get("extension") or "",
            "md5": raw.get("md5") or "",
        }


class JobRepository:
    """独立保存任务和结果，不修改业务源库。"""

    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self) -> None:
        with self._write_lock, closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS check_jobs (
                    job_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    template_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS check_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES check_jobs(job_id) ON DELETE CASCADE,
                    graduate_id INTEGER,
                    evidence_file_id INTEGER,
                    route_id TEXT,
                    status TEXT NOT NULL,
                    cache_key TEXT,
                    context_json TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS check_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL UNIQUE REFERENCES check_items(id) ON DELETE CASCADE,
                    result_json TEXT NOT NULL,
                    raw_metrics_json TEXT,
                    completed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_check_items_job_status
                    ON check_items(job_id, status);
                CREATE INDEX IF NOT EXISTS idx_check_items_cache
                    ON check_items(cache_key, status);
                """
            )
            job_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(check_jobs)")
            }
            if "task_name" not in job_columns:
                connection.execute(
                    "ALTER TABLE check_jobs ADD COLUMN task_name TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "UPDATE check_jobs SET task_name=job_id WHERE TRIM(task_name)=''"
            )

    def recover_incomplete(self) -> list[str]:
        with self._write_lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE check_items SET status='queued', updated_at=? WHERE status='running'",
                (now_iso(),),
            )
            rows = connection.execute(
                "SELECT DISTINCT job_id FROM check_items WHERE status='queued'"
            ).fetchall()
            job_ids = [str(row[0]) for row in rows]
            if job_ids:
                connection.executemany(
                    "UPDATE check_jobs SET status='queued', updated_at=? WHERE job_id=?",
                    [(now_iso(), job_id) for job_id in job_ids],
                )
            return job_ids

    def cached_result(self, cache_key: str) -> dict[str, Any] | None:
        if not cache_key:
            return None
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT cr.result_json FROM check_results cr
                JOIN check_items ci ON ci.id=cr.item_id
                WHERE ci.cache_key=? AND ci.status='completed'
                ORDER BY cr.id DESC LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def create_job(
        self,
        request: dict[str, Any],
        model: str,
        template_version: str,
        items: list[dict[str, Any]],
    ) -> str:
        job_id = f"chk_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        timestamp = now_iso()
        with self._write_lock, closing(self._connect()) as connection, connection:
            task_name = str(request.get("task_name") or "").strip() or (
                f"就业材料核验-{datetime.now():%Y%m%d-%H%M%S}"
            )
            connection.execute(
                """
                INSERT INTO check_jobs(
                    job_id,task_name,status,request_json,model,template_version,
                    created_at,updated_at
                ) VALUES (?,?,'queued',?,?,?,?,?)
                """,
                (
                    job_id,
                    task_name,
                    json.dumps(request, ensure_ascii=False),
                    model,
                    template_version,
                    timestamp,
                    timestamp,
                ),
            )
            for item in items:
                cursor = connection.execute(
                    """
                    INSERT INTO check_items(
                        job_id, graduate_id, evidence_file_id, route_id, status,
                        cache_key, context_json, error_code, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        item.get("graduate_id"),
                        item.get("evidence_file_id"),
                        item.get("route_id"),
                        item.get("status", "queued"),
                        item.get("cache_key"),
                        json.dumps(item.get("context", {}), ensure_ascii=False),
                        item.get("error_code"),
                        item.get("error_message"),
                        timestamp,
                        timestamp,
                    ),
                )
                if item.get("result") is not None:
                    connection.execute(
                        "INSERT INTO check_results(item_id,result_json,raw_metrics_json,completed_at) VALUES(?,?,?,?)",
                        (
                            cursor.lastrowid,
                            json.dumps(item["result"], ensure_ascii=False),
                            "{}",
                            timestamp,
                        ),
                    )
            self._refresh_job_locked(connection, job_id)
        return job_id

    def queued_items(self, job_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM check_items WHERE job_id=? AND status='queued' ORDER BY id",
                (job_id,),
            )
            return [self._item(row) for row in rows]

    def mark_running(self, item_id: int) -> bool:
        with self._write_lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT ci.job_id,ci.status AS item_status,cj.status AS job_status
                FROM check_items ci JOIN check_jobs cj ON cj.job_id=ci.job_id
                WHERE ci.id=?
                """,
                (item_id,),
            ).fetchone()
            if not row or row["item_status"] != "queued" or row["job_status"] == "cancelled":
                return False
            connection.execute(
                "UPDATE check_items SET status='running',updated_at=? WHERE id=?",
                (now_iso(), item_id),
            )
            connection.execute(
                "UPDATE check_jobs SET status='running',updated_at=? WHERE job_id=?",
                (now_iso(), str(row[0])),
            )
            return True

    def mark_completed(
        self,
        item_id: int,
        result: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        with self._write_lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT job_id,status FROM check_items WHERE id=?", (item_id,)
            ).fetchone()
            if not row or row["status"] == "cancelled":
                return
            timestamp = now_iso()
            connection.execute(
                "UPDATE check_items SET status='completed',updated_at=?,error_code=NULL,error_message=NULL WHERE id=?",
                (timestamp, item_id),
            )
            connection.execute(
                "INSERT OR REPLACE INTO check_results(item_id,result_json,raw_metrics_json,completed_at) VALUES(?,?,?,?)",
                (
                    item_id,
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    timestamp,
                ),
            )
            self._refresh_job_locked(connection, str(row[0]))

    def mark_failed(self, item_id: int, code: str, message: str) -> None:
        with self._write_lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT job_id,status FROM check_items WHERE id=?", (item_id,)
            ).fetchone()
            if not row or row["status"] == "cancelled":
                return
            connection.execute(
                "UPDATE check_items SET status='failed',error_code=?,error_message=?,updated_at=? WHERE id=?",
                (code, message, now_iso(), item_id),
            )
            self._refresh_job_locked(connection, str(row[0]))

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection, connection:
            job = connection.execute(
                "SELECT * FROM check_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not job:
                return None
            counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM check_items WHERE job_id=? GROUP BY status",
                    (job_id,),
                )
            }
            total = sum(counts.values())
            return {
                "job_id": job_id,
                "task_name": str(job["task_name"] or job_id),
                "status": str(job["status"]),
                "model": str(job["model"]),
                "total_items": total,
                "queued_items": counts.get("queued", 0),
                "running_items": counts.get("running", 0),
                "completed_items": counts.get("completed", 0) + counts.get("unsupported", 0),
                "failed_items": counts.get("failed", 0),
                "cancelled_items": counts.get("cancelled", 0),
                "created_at": str(job["created_at"]),
                "updated_at": str(job["updated_at"]),
            }

    def list_jobs(self, limit: int = 20, keyword: str = "") -> list[dict[str, Any]]:
        keyword = keyword.strip()
        where = "WHERE j.task_name LIKE ? OR j.job_id LIKE ?" if keyword else ""
        params: list[Any] = (
            [f"%{keyword}%", f"%{keyword}%", limit] if keyword else [limit]
        )
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                f"""
                SELECT j.job_id,j.task_name,j.status,j.model,j.created_at,j.updated_at,
                       COUNT(ci.id) AS total_items,
                       SUM(CASE WHEN ci.status='queued' THEN 1 ELSE 0 END) AS queued_items,
                       SUM(CASE WHEN ci.status='running' THEN 1 ELSE 0 END) AS running_items,
                       SUM(CASE WHEN ci.status IN ('completed','unsupported') THEN 1 ELSE 0 END) AS completed_items,
                       SUM(CASE WHEN ci.status='failed' THEN 1 ELSE 0 END) AS failed_items,
                       SUM(CASE WHEN ci.status='cancelled' THEN 1 ELSE 0 END) AS cancelled_items
                FROM check_jobs j
                LEFT JOIN check_items ci ON ci.job_id=j.job_id
                {where}
                GROUP BY j.job_id
                ORDER BY j.created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                {
                    "job_id": str(row["job_id"]),
                    "task_name": str(row["task_name"] or row["job_id"]),
                    "status": str(row["status"]),
                    "model": str(row["model"]),
                    "total_items": int(row["total_items"] or 0),
                    "queued_items": int(row["queued_items"] or 0),
                    "running_items": int(row["running_items"] or 0),
                    "completed_items": int(row["completed_items"] or 0),
                    "failed_items": int(row["failed_items"] or 0),
                    "cancelled_items": int(row["cancelled_items"] or 0),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                }
                for row in rows
            ]

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """强制取消等待中或运行中的任务，保留已经完成的结果。"""
        with self._write_lock, closing(self._connect()) as connection, connection:
            job = connection.execute(
                "SELECT job_id,task_name,status FROM check_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if job is None:
                return None
            if str(job["status"]) not in {"queued", "running"}:
                raise RuntimeError("仅等待中或运行中的任务可以取消")
            timestamp = now_iso()
            cursor = connection.execute(
                """
                UPDATE check_items SET status='cancelled',updated_at=?
                WHERE job_id=? AND status IN ('queued','running')
                """,
                (timestamp, job_id),
            )
            connection.execute(
                "UPDATE check_jobs SET status='cancelled',updated_at=? WHERE job_id=?",
                (timestamp, job_id),
            )
            return {
                "job_id": str(job["job_id"]),
                "task_name": str(job["task_name"] or job["job_id"]),
                "status": "cancelled",
                "cancelled_items": int(cursor.rowcount or 0),
            }

    def delete_job(self, job_id: str) -> dict[str, Any] | None:
        """删除已结束任务及其队列项和结果；业务源库与原始材料不受影响。"""
        with self._write_lock, closing(self._connect()) as connection, connection:
            job = connection.execute(
                "SELECT job_id,task_name,status FROM check_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if job is None:
                return None
            job_status = str(job["status"])
            if job_status in {"queued", "running"}:
                raise RuntimeError("运行中或等待中的任务不能删除")
            item_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM check_items WHERE job_id=?", (job_id,)
                ).fetchone()[0]
            )
            result_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM check_results cr
                    JOIN check_items ci ON ci.id=cr.item_id
                    WHERE ci.job_id=?
                    """,
                    (job_id,),
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM check_jobs WHERE job_id=?", (job_id,))
            return {
                "job_id": str(job["job_id"]),
                "task_name": str(job["task_name"] or job["job_id"]),
                "deleted_items": item_count,
                "deleted_results": result_count,
            }

    def dashboard_counts(self) -> dict[str, int]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT status,COUNT(*) FROM check_jobs GROUP BY status"
            ).fetchall()
            counts = {str(row[0]): int(row[1]) for row in rows}
            return {
                "total_jobs": sum(counts.values()),
                "queued_jobs": counts.get("queued", 0),
                "running_jobs": counts.get("running", 0),
                "completed_jobs": counts.get("completed", 0),
                "failed_jobs": counts.get("failed", 0),
                "cancelled_jobs": counts.get("cancelled", 0),
            }

    def job_results(self, job_id: str) -> list[dict[str, Any]] | None:
        with closing(self._connect()) as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM check_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not exists:
                return None
            rows = connection.execute(
                """
                SELECT ci.*, cr.result_json, cr.raw_metrics_json
                FROM check_items ci LEFT JOIN check_results cr ON cr.item_id=ci.id
                WHERE ci.job_id=? ORDER BY ci.graduate_id,ci.evidence_file_id,ci.id
                """,
                (job_id,),
            )
            values = []
            for row in rows:
                item = self._item(row)
                item["result"] = json.loads(row["result_json"]) if row["result_json"] else None
                item["metrics"] = json.loads(row["raw_metrics_json"]) if row["raw_metrics_json"] else {}
                values.append(item)
            return values

    def checked_student_items(self) -> list[dict[str, Any]]:
        """读取跨任务学生结果，供服务层生成按学生去重的核验汇总。"""
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT
                    ci.*,
                    j.task_name,
                    j.status AS job_status,
                    j.created_at AS job_created_at,
                    j.updated_at AS job_updated_at,
                    cr.result_json,
                    cr.raw_metrics_json,
                    cr.completed_at
                FROM check_items ci
                JOIN check_jobs j ON j.job_id=ci.job_id
                LEFT JOIN check_results cr ON cr.item_id=ci.id
                WHERE ci.graduate_id IS NOT NULL
                ORDER BY
                    j.updated_at DESC,
                    j.created_at DESC,
                    j.job_id DESC,
                    ci.graduate_id,
                    ci.evidence_file_id,
                    ci.id
                """
            )
            values = []
            for row in rows:
                item = self._item(row)
                item.update(
                    {
                        "task_name": str(row["task_name"] or row["job_id"]),
                        "job_status": str(row["job_status"]),
                        "job_created_at": str(row["job_created_at"]),
                        "job_updated_at": str(row["job_updated_at"]),
                        "completed_at": str(row["completed_at"] or ""),
                        "result": (
                            json.loads(row["result_json"])
                            if row["result_json"]
                            else None
                        ),
                        "metrics": (
                            json.loads(row["raw_metrics_json"])
                            if row["raw_metrics_json"]
                            else {}
                        ),
                    }
                )
                values.append(item)
            return values

    @staticmethod
    def _item(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "item_id": int(row["id"]),
            "job_id": str(row["job_id"]),
            "graduate_id": row["graduate_id"],
            "evidence_file_id": row["evidence_file_id"],
            "route_id": row["route_id"],
            "status": str(row["status"]),
            "cache_key": row["cache_key"],
            "context": json.loads(row["context_json"]),
            "error_code": row["error_code"],
            "error_message": row["error_message"],
        }

    @staticmethod
    def _refresh_job_locked(connection: sqlite3.Connection, job_id: str) -> None:
        current = connection.execute(
            "SELECT status FROM check_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if current and str(current[0]) == "cancelled":
            return
        counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT status,COUNT(*) FROM check_items WHERE job_id=? GROUP BY status",
                (job_id,),
            )
        }
        if counts.get("running", 0):
            status = "running"
        elif counts.get("queued", 0):
            status = "queued"
        elif counts.get("failed", 0) and not (
            counts.get("completed", 0) or counts.get("unsupported", 0)
        ):
            status = "failed"
        else:
            status = "completed"
        connection.execute(
            "UPDATE check_jobs SET status=?,updated_at=? WHERE job_id=?",
            (status, now_iso(), job_id),
        )
