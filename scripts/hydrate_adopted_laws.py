#!/usr/bin/env python3
"""
Download adopted-law PDFs and extract structured law text into dep_act_laws.

The script is incremental by default:
- rows with final adopted-law identifiers or parliament-adopted status are considered;
- cached PDFs in outputs/pdfs/adopted_laws are reused;
- existing adopted_law_text_json values are not recomputed unless --force-extract is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crawl_deputy_activity import (
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_USER_AGENT,
    Fetcher,
    _clean_text,
    _extract_pdf_text_direct,
    _extract_pdf_text_ocr,
    _fetch_senat_search_results_html,
    _fold,
    _hostname_is_senat_ro,
    _law_search_number_and_year,
    _normalize_law_source_url,
    _parse_rows,
    _safe_filename_component,
    _write_bytes_atomic,
    _looks_like_senat_legislation_search_page,
)
from init_db import DEFAULT_DB_PATH, init_db


DEFAULT_ADOPTED_LAW_PDF_DIR = Path("outputs/pdfs/adopted_laws")
DEFAULT_ADOPTED_LAW_OCR_PAGES = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_pdf_filename(adopted_law_identifier: str, law_id: str) -> str:
    cache_key = _safe_filename_component(adopted_law_identifier or law_id)
    return f"adopted_law_{cache_key}.pdf"


def _find_cached_pdf(
    pdf_dir: Path,
    *,
    adopted_law_identifier: str,
    law_id: str,
    stored_filename: str | None,
) -> Path | None:
    candidates: list[Path] = []
    if stored_filename:
        candidates.append(pdf_dir / stored_filename)
    expected = pdf_dir / _safe_pdf_filename(adopted_law_identifier, law_id)
    candidates.append(expected)

    safe_id = _safe_filename_component(adopted_law_identifier)
    candidates.extend(sorted(pdf_dir.glob(f"*{safe_id}*.pdf")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def list_adopted_law_pdf_urls(html_text: str, source_url: str) -> list[str]:
    """Find adopted/final law PDF candidates in priority order.

    CDEP sometimes exposes a "Forma adoptata de Camera" link that later 404s,
    while "Forma pentru promulgare" is still available. Try the adopted form
    first, then the promulgation form as the closest final-text fallback.
    """
    urls: list[str] = []

    def add_url(href: str) -> None:
        url = urllib.parse.urljoin(source_url, href.replace("\\", "/"))
        if url not in urls:
            urls.append(url)

    def row_matches(text: str, *, adopted: bool = False, promulgare: bool = False) -> bool:
        folded = _fold(text)
        if "forma" not in folded:
            return False
        if adopted and "adoptat" in folded:
            return True
        if promulgare and ("promulgare" in folded or "promulgat" in folded):
            return True
        return False

    for row in _parse_rows(html_text):
        if not row_matches(row.text, adopted=True):
            continue
        for link in row.links:
            href = (link.href or "").strip()
            if ".pdf" in href.casefold():
                add_url(href)

    for row in _parse_rows(html_text):
        for link in row.links:
            href = (link.href or "").strip()
            if ".pdf" in href.casefold() and row_matches(link.text, adopted=True):
                add_url(href)

    for match in re.finditer(
        r"""<a\s+[^>]*href\s*=\s*["']([^"']+\.pdf[^"']*)["'][^>]*>(.*?)</a>""",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = match.group(1)
        label = _fold(re.sub(r"<[^>]+>", " ", match.group(2)))
        if "forma" in label and "adoptat" in label:
            add_url(href)

    for row in _parse_rows(html_text):
        if not row_matches(row.text, promulgare=True):
            continue
        for link in row.links:
            href = (link.href or "").strip()
            if ".pdf" in href.casefold():
                add_url(href)

    for row in _parse_rows(html_text):
        for link in row.links:
            href = (link.href or "").strip()
            if ".pdf" in href.casefold() and row_matches(link.text, promulgare=True):
                add_url(href)

    for match in re.finditer(
        r"""<a\s+[^>]*href\s*=\s*["']([^"']+\.pdf[^"']*)["'][^>]*>(.*?)</a>""",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = match.group(1)
        label = _fold(re.sub(r"<[^>]+>", " ", match.group(2)))
        if "forma" in label and "promulgare" in label:
            add_url(href)
    return urls


def parse_adopted_law_pdf_url(html_text: str, source_url: str) -> str | None:
    """Find the first PDF linked from "Forma adoptata" or final fallback rows."""
    urls = list_adopted_law_pdf_urls(html_text, source_url)
    return urls[0] if urls else None


def _list_senat_legislative_file_urls(html_text: str, source_url: str) -> list[str]:
    urls: list[str] = []
    for row in _parse_rows(html_text):
        for link in row.links:
            href = (link.href or "").strip()
            if "senat.ro/legis/lista.aspx" not in href.casefold():
                continue
            url = urllib.parse.urljoin(source_url, href)
            if url not in urls:
                urls.append(url)
    return urls


def discover_adopted_law_pdf_urls(
    fetcher: Fetcher,
    html_text: str,
    source_url: str,
) -> list[str]:
    urls = list_adopted_law_pdf_urls(html_text, source_url)
    for senat_url in _list_senat_legislative_file_urls(html_text, source_url):
        try:
            senat_html = fetcher.fetch(senat_url)
        except Exception:
            continue
        for pdf_url in list_adopted_law_pdf_urls(senat_html, senat_url):
            if pdf_url not in urls:
                urls.append(pdf_url)
    return urls


def fetch_law_page(fetcher: Fetcher, source_url: str, identifier: str | None) -> tuple[str, str]:
    url = _normalize_law_source_url(source_url)
    html_text = fetcher.fetch(url)
    if _hostname_is_senat_ro(url) and _looks_like_senat_legislation_search_page(html_text):
        search_number, search_year = _law_search_number_and_year(url, identifier)
        if search_number and search_year:
            html_text = _fetch_senat_search_results_html(
                fetcher,
                url,
                search_number=search_number,
                search_year=search_year,
            )
    return html_text, url


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()


def _extract_pdf_text(pdf_bytes: bytes, *, ocr_pages: int, ocr_language: str) -> tuple[str, str]:
    direct_text = _extract_pdf_text_direct(pdf_bytes)
    if len(_clean_text(direct_text)) >= 120:
        return direct_text, "pypdf"
    if shutil.which("tesseract") is None:
        return direct_text, "pypdf_empty_no_tesseract"
    try:
        return (
            _extract_pdf_text_ocr(
                pdf_bytes,
                language=ocr_language,
                max_pages=ocr_pages,
                from_tail=False,
            ),
            "ocr",
        )
    except Exception:
        if direct_text:
            return direct_text, "pypdf_ocr_failed"
        raise


_ARTICLE_RE = re.compile(
    r"(?im)(?:^|\n)\s*Art\.?\s*([0-9]+|[IVXLCDM]+)\.?\s*[-–.]?\s*"
)
_CHAPTER_RE = re.compile(
    r"(?im)(?:^|\n)\s*((?:CAPITOLUL|Capitolul)\s+[0-9IVXLCDM]+[^\n]*)"
)


def _extract_articles(text: str) -> list[dict[str, str]]:
    matches = list(_ARTICLE_RE.finditer(text))
    articles: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        articles.append(
            {
                "number": match.group(1).strip(),
                "text": body,
            }
        )
    return articles


def _extract_chapters(text: str) -> list[dict[str, Any]]:
    matches = list(_CHAPTER_RE.finditer(text))
    chapters: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        chapters.append(
            {
                "title": _clean_text(match.group(1)),
                "text": section_text,
                "articles": _extract_articles(section_text),
            }
        )
    return chapters


def _extract_law_title(text: str, fallback_title: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    useful: list[str] = []
    for line in lines[:30]:
        folded = _fold(line)
        if folded.startswith(("parlamentul romaniei", "camera deputatilor", "senat")):
            continue
        if folded.startswith(("lege", "ordonanta", "hotarare")) or "privind" in folded:
            useful.append(line)
        if useful and len(" ".join(useful)) > 40:
            break
    return _clean_text(" ".join(useful)) or fallback_title


def structure_law_text(
    *,
    text: str,
    title: str,
    adopted_law_identifier: str,
    pdf_filename: str,
    pdf_url: str | None,
    extraction_method: str,
) -> dict[str, Any]:
    normalized = _normalize_pdf_text(text)
    articles = _extract_articles(normalized)
    chapters = _extract_chapters(normalized)
    first_articles = articles[:2] if articles else []
    plain_excerpt = _clean_text(" ".join(article["text"] for article in first_articles))[:900]
    return {
        "schema_version": 1,
        "adopted_law_identifier": adopted_law_identifier,
        "pdf_filename": pdf_filename,
        "pdf_url": pdf_url,
        "title": _extract_law_title(normalized, title),
        "article_count": len(articles),
        "chapter_count": len(chapters),
        "plain_text_excerpt": plain_excerpt,
        "full_text": normalized,
        "chapters": chapters,
        "articles": articles,
        "extraction_method": extraction_method,
        "extracted_at": _now_iso(),
    }


def _row_adopted_law_cache_label(row: sqlite3.Row) -> str:
    return str(row["adopted_law_identifier"] or row["identifier"] or row["law_id"])


def _iter_adopted_laws(conn: sqlite3.Connection, *, force_extract: bool, limit: int | None) -> list[sqlite3.Row]:
    where = [
        """(
            (adopted_law_identifier IS NOT NULL AND TRIM(adopted_law_identifier) <> '')
            OR law_status IN ('adoptata', 'adoptata_in_parlament')
        )"""
    ]
    if not force_extract:
        where.append("(adopted_law_text_json IS NULL OR TRIM(adopted_law_text_json) = '')")
    sql = f"""
        SELECT law_id, source_url, identifier, title, adopted_law_identifier,
               law_status,
               adopted_law_pdf_filename, adopted_law_pdf_url, adopted_law_text_json
        FROM dep_act_laws
        WHERE {' AND '.join(where)}
        ORDER BY law_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (limit,)).fetchall()
    return conn.execute(sql).fetchall()


def hydrate_adopted_laws(
    *,
    db_path: Path,
    pdf_dir: Path,
    force_extract: bool,
    limit: int | None,
    skip_download: bool,
    extract_only: bool,
    ocr_pages: int,
    ocr_language: str,
) -> int:
    init_db(db_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(user_agent=DEFAULT_USER_AGENT)

    downloaded = 0
    extracted = 0
    skipped = 0
    failed = 0
    unavailable = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _iter_adopted_laws(conn, force_extract=force_extract, limit=limit)
        print(f"Found {len(rows)} adopted/adopted-in-parliament laws to hydrate.")

        for row in rows:
            law_id = str(row["law_id"])
            adopted_law_identifier = _row_adopted_law_cache_label(row)
            expected_filename = _safe_pdf_filename(adopted_law_identifier, law_id)
            cache_path = _find_cached_pdf(
                pdf_dir,
                adopted_law_identifier=adopted_law_identifier,
                law_id=law_id,
                stored_filename=row["adopted_law_pdf_filename"],
            )
            pdf_url = row["adopted_law_pdf_url"]

            try:
                if cache_path is None and not skip_download and not extract_only:
                    html_text, final_source_url = fetch_law_page(
                        fetcher,
                        str(row["source_url"]),
                        row["identifier"],
                    )
                    pdf_urls = discover_adopted_law_pdf_urls(
                        fetcher,
                        html_text,
                        final_source_url,
                    )
                    if not pdf_urls:
                        unavailable += 1
                        conn.execute(
                            """
                            UPDATE dep_act_laws
                            SET adopted_law_parse_error = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE law_id = ?
                            """,
                            ("Adopted/final law PDF link not published yet", law_id),
                        )
                        conn.commit()
                        print(f"  SKIP {adopted_law_identifier}: adopted/final PDF link not published yet")
                        continue
                    last_download_error: Exception | None = None
                    pdf_bytes: bytes | None = None
                    for candidate_pdf_url in pdf_urls:
                        try:
                            pdf_bytes = fetcher.fetch_bytes(candidate_pdf_url)
                            pdf_url = candidate_pdf_url
                            break
                        except Exception as exc:
                            last_download_error = exc
                    if pdf_bytes is None:
                        assert last_download_error is not None
                        raise ValueError(
                            "All adopted-law PDF candidates failed; "
                            f"last error: {last_download_error}; candidates={pdf_urls}"
                        )
                    cache_path = pdf_dir / expected_filename
                    _write_bytes_atomic(cache_path, pdf_bytes)
                    downloaded += 1
                    conn.execute(
                        """
                        UPDATE dep_act_laws
                        SET adopted_law_pdf_filename = ?,
                            adopted_law_pdf_url = ?,
                            adopted_law_parse_error = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE law_id = ?
                        """,
                        (cache_path.name, pdf_url, law_id),
                    )
                    conn.commit()

                if cache_path is None:
                    skipped += 1
                    conn.execute(
                        """
                        UPDATE dep_act_laws
                        SET adopted_law_parse_error = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE law_id = ?
                        """,
                        ("Adopted law PDF is not cached and download is disabled", law_id),
                    )
                    conn.commit()
                    continue

                if (
                    row["adopted_law_text_json"]
                    and str(row["adopted_law_text_json"]).strip()
                    and not force_extract
                ):
                    skipped += 1
                    continue

                pdf_bytes = cache_path.read_bytes()
                raw_text, extraction_method = _extract_pdf_text(
                    pdf_bytes,
                    ocr_pages=ocr_pages,
                    ocr_language=ocr_language,
                )
                structured = structure_law_text(
                    text=raw_text,
                    title=str(row["title"]),
                    adopted_law_identifier=adopted_law_identifier,
                    pdf_filename=cache_path.name,
                    pdf_url=pdf_url,
                    extraction_method=extraction_method,
                )
                conn.execute(
                    """
                    UPDATE dep_act_laws
                    SET adopted_law_pdf_filename = ?,
                        adopted_law_pdf_url = COALESCE(?, adopted_law_pdf_url),
                        adopted_law_text_json = ?,
                        adopted_law_text_extracted_at = ?,
                        adopted_law_parse_error = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE law_id = ?
                    """,
                    (
                        cache_path.name,
                        pdf_url,
                        json.dumps(structured, ensure_ascii=False, sort_keys=True),
                        structured["extracted_at"],
                        law_id,
                    ),
                )
                conn.commit()
                extracted += 1
                print(f"  OK {adopted_law_identifier}: {cache_path.name}")
            except Exception as exc:
                failed += 1
                conn.execute(
                    """
                    UPDATE dep_act_laws
                    SET adopted_law_parse_error = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE law_id = ?
                    """,
                    (str(exc)[:1000], law_id),
                )
                conn.commit()
                print(f"  ERROR {adopted_law_identifier}: {exc}", file=sys.stderr)
            time.sleep(0.05)

    print(
        "Adopted-law hydration complete: "
        f"{downloaded} downloaded, {extracted} extracted, {skipped} skipped, "
        f"{unavailable} unavailable, {failed} failed."
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download adopted-law PDFs and store structured extracted law text."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument(
        "--pdf-dir",
        default=str(DEFAULT_ADOPTED_LAW_PDF_DIR),
        help=f"Adopted-law PDF cache dir (default: {DEFAULT_ADOPTED_LAW_PDF_DIR})",
    )
    parser.add_argument("--force-extract", action="store_true", help="Re-extract text even when adopted_law_text_json already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N adopted laws.")
    parser.add_argument("--skip-download", action="store_true", help="Only extract PDFs that are already cached locally.")
    parser.add_argument("--extract-only", action="store_true", help="Alias for --skip-download, kept explicit for reruns.")
    parser.add_argument("--ocr-pages", type=int, default=DEFAULT_ADOPTED_LAW_OCR_PAGES, help=f"Maximum pages to OCR on image PDFs (default: {DEFAULT_ADOPTED_LAW_OCR_PAGES}).")
    parser.add_argument("--ocr-language", default=DEFAULT_OCR_LANGUAGE, help=f"Tesseract language(s), default {DEFAULT_OCR_LANGUAGE}.")
    args = parser.parse_args()

    return hydrate_adopted_laws(
        db_path=Path(args.db_path),
        pdf_dir=Path(args.pdf_dir),
        force_extract=args.force_extract,
        limit=args.limit,
        skip_download=args.skip_download or args.extract_only,
        extract_only=args.extract_only,
        ocr_pages=args.ocr_pages,
        ocr_language=args.ocr_language,
    )


if __name__ == "__main__":
    raise SystemExit(main())
