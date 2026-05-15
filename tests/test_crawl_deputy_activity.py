from __future__ import annotations

import contextlib
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_deputy_activity import (  # noqa: E402
    _fuzzy_member_matches_initiator_lines,
    _INITIATOR_BEFORE_SUSTINATORI_NOTE,
    _embedded_pdf_missing_initiator_name_signals,
    _fetch_senat_search_results_html,
    _canonicalize_senat_url,
    count_law_initiator_pdf_cache_gaps,
    _law_search_number_and_year,
    _is_transient_network_exception,
    _law_initiator_pdf_cache_path,
    _write_bytes_atomic,
    _looks_like_senat_legislation_search_page,
    _normalize_law_source_url,
    ensure_activity_schema,
    hydrate_law_initiators_for_records,
    ListingRecord,
    PoliticalDeclarationRecord,
    crawl_member,
    extract_initiators_section,
    extract_initiators_section_with_source,
    hydrate_political_declaration_records,
    hydrate_question_records,
    load_law_records_for_initiator_hydration,
    match_initiator_member_ids,
    _law_initiator_match_log_suffix,
    list_senat_law_initiator_pdf_urls,
    parse_law_initiator_pdf_url,
    parse_motive_pdf_url,
    parse_senat_adresa_inaintare_pdf_url,
    parse_senat_expunere_motive_pdf_url,
    parse_senat_forma_initiatorului_pdf_url,
    parse_political_declaration_detail,
    parse_political_declaration_text_page,
    parse_question_detail_recipient,
    parse_listing_records,
    parse_profile_activity,
    run_law_initiator_hydration_phase,
    single_signature_fuzzy_match,
    store_records,
    update_member_activity_crawl,
    update_member_activity_error,
)
import urllib.error  # noqa: E402
from init_db import init_db  # noqa: E402


def insert_test_member(conn: sqlite3.Connection, member_id: str = "deputat_1") -> None:
    conn.execute(
        """
        INSERT INTO members (
            member_id, source_member_id, chamber, name, normalized_name,
            party_id, profile_url, circumscriptie
        )
        VALUES (?, ?, 'deputat', 'Deputat Test', 'deputat test', 'TST',
                'https://example.test/profile', NULL)
        """,
        (member_id, member_id.removeprefix("deputat_")),
    )


class TestDeputyActivityCrawler(unittest.TestCase):
    def test_parse_profile_activity_links_and_counts(self):
        html = """
        <table>
          <tr>
            <td align="right">Propuneri legislative iniţiate: </td>
            <td><a href="/pls/parlam/structura2015.mp?idm=1&pag=2">12</a>,
                din care 2 promulgate legi</td>
          </tr>
          <tr>
            <td align="right">Proiecte de hotarâre iniţiate: </td>
            <td><a href="structura2015.mp?idm=1&pag=3">4</a></td>
          </tr>
          <tr>
            <td align="right">Întrebari şi interpelări: </td>
            <td><a href="structura2015.mp?idm=1&pag=7">9</a></td>
          </tr>
          <tr>
            <td align="right">Moţiuni: </td>
            <td><a href="structura2015.mp?idm=1&pag=8">1</a></td>
          </tr>
          <tr>
            <td align="right">Declaraţii politice depuse în scris: </td>
            <td><a href="structura2015.mp?idm=1&pag=9">3</a></td>
          </tr>
        </table>
        """
        activity = parse_profile_activity(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&cam=2&leg=2024",
        )

        self.assertEqual(activity["legislative_proposals"].count, 12)
        self.assertEqual(activity["legislative_proposals"].promulgated_count, 2)
        self.assertEqual(activity["decision_projects"].count, 4)
        self.assertEqual(activity["questions"].count, 9)
        self.assertEqual(activity["motions"].count, 1)
        self.assertEqual(activity["political_declarations"].count, 3)
        self.assertEqual(
            activity["legislative_proposals"].url,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=2",
        )

    def test_parse_profile_activity_current_motion_link(self):
        html = """
        <table>
          <tr valign="top">
            <td align="right">Moţiuni: </td>
            <td>
              <a href="/pls/parlam/structura2015.mp?idm=3&amp;cam=2&amp;leg=2024&amp;pag=11&amp;idl=1&amp;prn=0&amp;par=">
                <b>3</b>
              </a>
            </td>
          </tr>
        </table>
        """
        activity = parse_profile_activity(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024",
        )

        self.assertEqual(activity["motions"].count, 3)
        self.assertEqual(
            activity["motions"].url,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024&pag=11&idl=1&prn=0&par=",
        )

    def test_parse_profile_activity_nested_activity_table_not_clobbered_by_outer_row(self):
        activity_table = """
        <table>
          <tr valign="top">
            <td align="right">Moţiuni: </td>
            <td><a href="/pls/parlam/structura2015.mp?idm=3&amp;pag=11"><b>3</b></a></td>
          </tr>
        </table>
        """
        html = f"""
        <table>
          <tr>
            <td>Activitate parlamentară {activity_table}</td>
            <td>Alte date fără număr relevant</td>
          </tr>
        </table>
        """

        activity = parse_profile_activity(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024",
        )

        self.assertEqual(activity["motions"].count, 3)
        self.assertEqual(
            activity["motions"].url,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&pag=11",
        )

    def test_parse_motion_listing_uses_motion_idm_as_record_id(self):
        html = """
        <table>
          <tr>
            <td>1.</td>
            <td>
              <a href="/pls/parlam/parlament.motiuni2015.detalii?leg=2024&amp;cam=0&amp;idm=1582">
                ”România nu este de vânzare - fără progresişti la guvernare”
              </a>
            </td>
            <td>Iniţiată de 118 deputaţi şi senatori</td>
            <td>7 / 05-12-2025</td>
            <td>Respinsă</td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024&pag=11",
            "motions",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "motion:cdep:1582")
        self.assertEqual(
            records[0].source_url,
            "https://www.cdep.ro/pls/parlam/parlament.motiuni2015.detalii?leg=2024&cam=0&idm=1582",
        )

    def test_parse_listing_records_deduplicates_by_url(self):
        html = """
        <table>
          <tr><th>Nr.</th><th>Initiativa</th><th>Obiect</th></tr>
          <tr>
            <td>1</td>
            <td><a href="/pls/proiecte/upl_pck2015.proiect?idp=22110">PL-x 44/2025</a></td>
            <td>privind testarea parserului</td>
          </tr>
          <tr>
            <td>2</td>
            <td><a href="/pls/proiecte/upl_pck2015.proiect?idp=22110">PL-x 44/2025</a></td>
            <td>duplicat pe aceeasi pagina</td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=2",
            "laws",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "law:cdep:22110")
        self.assertEqual(records[0].identifier, "PL-x 44/2025")
        self.assertEqual(
            records[0].source_url,
            "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?cam=2&idp=22110",
        )

    def test_parse_law_listing_strips_member_specific_prefix_and_extracts_identifier(self):
        html = """
        <table>
          <tr><th>Nr.</th><th>Data</th><th>Initiativa</th><th>Stadiu</th></tr>
          <tr>
            <td>3.</td>
            <td>102/07.04.2025</td>
            <td>
              <a href="/pls/proiecte/upl_pck2015.proiect?idp=33125">B165/2025</a>
              Propunere legislativă privind înfiinţarea Institutului de Formare
              în Sectorul Public procedura legislativa încetata
            </td>
            <td>Lege 233/2025</td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=2",
            "laws",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].identifier, "B165/2025")
        self.assertEqual(
            records[0].details_text,
            "Propunere legislativă privind înfiinţarea Institutului de Formare "
            "în Sectorul Public procedura legislativa încetata",
        )
        self.assertEqual(records[0].title, records[0].details_text)
        self.assertEqual(records[0].adopted_law_identifier, "Lege 233/2025")

    def test_parse_law_adoption_uses_stadiu_column_only(self):
        html = """
        <table>
          <tr><th>Nr.</th><th>Data</th><th>Initiativa</th><th>Stadiu</th></tr>
          <tr>
            <td>4.</td>
            <td>174/02.06.2025</td>
            <td>
              <a href="/pls/proiecte/upl_pck2015.proiect?idp=22386">L78/2025</a>
              Propunere legislativă pentru stimularea investiţiilor şi modificarea
              art.291 alin.(2) din Legea 227/2015 Codul Fiscal
            </td>
            <td>la comisii</td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=2",
            "laws",
        )

        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].adopted_law_identifier)
        self.assertIn("Legea 227/2015 Codul Fiscal", records[0].details_text)
        self.assertNotIn("la comisii", records[0].details_text)

    def test_parse_motive_pdf_url(self):
        html = """
        <table>
          <tr valign="top">
            <td width="20">
              <a href="/proiecte/2025/100/60/1/em178.pdf" target="PDF">
                <img src="/img/icon_pdf_small.gif" border="0">
              </a>
            </td>
            <td colspan="3" width="100%">Expunerea de motive</td>
          </tr>
        </table>
        """

        self.assertEqual(
            parse_motive_pdf_url(
                html,
                "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?idp=22374",
            ),
            "https://www.cdep.ro/proiecte/2025/100/60/1/em178.pdf",
        )

    def test_senat_initiator_ocr_uses_only_expunerea_de_motive(self):
        html = """
        <table>
          <tr>
            <td>
 <a href="/legis/PDF/2026/26b131FG.PDF?nocache=true">
 forma inițiatorului
              </a>
              <a href="/legis/PDF/2026/26b131EM.PDF?nocache=true">
                expunerea de motive
              </a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b131&an_cls=2026"
        em = "https://senat.ro/legis/PDF/2026/26b131EM.PDF?nocache=true"
        self.assertEqual(parse_senat_expunere_motive_pdf_url(html, base), em)
        self.assertEqual(parse_law_initiator_pdf_url(html, base), em)
        self.assertEqual(list_senat_law_initiator_pdf_urls(html, base), [em])

    def test_parse_law_initiator_pdf_url_senat_falls_back_to_expunere(self):
        html = """
        <table>
          <tr valign="top">
            <td>
              <a href="/legis/PDF/2026/26b131EM.PDF?nocache=true">expunerea de motive</a>
            </td>
          </tr>
        </table>
        """
        base = "https://www.senat.ro/legis/lista.aspx?nr_cls=b131&an_cls=2026"
        self.assertEqual(
            parse_law_initiator_pdf_url(html, base),
            "https://www.senat.ro/legis/PDF/2026/26b131EM.PDF?nocache=true",
        )

    def test_parse_law_initiator_pdf_url_cdep_unchanged(self):
        html = """
        <table>
          <tr valign="top">
            <td width="20">
              <a href="/proiecte/2025/100/60/1/em178.pdf" target="PDF">
                <img src="/img/icon_pdf_small.gif" border="0">
              </a>
            </td>
            <td colspan="3" width="100%">Expunerea de motive</td>
          </tr>
        </table>
        """
        self.assertEqual(
            parse_law_initiator_pdf_url(
                html,
                "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?idp=22374",
            ),
            "https://www.cdep.ro/proiecte/2025/100/60/1/em178.pdf",
        )

    def test_law_search_number_and_year_prefers_source_url(self):
        self.assertEqual(
            _law_search_number_and_year(
                "https://senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
                "L 257/2026",
            ),
            ("b135", "2026"),
        )

    def test_canonicalize_senat_url_promotes_www_host(self):
        self.assertEqual(
            _canonicalize_senat_url(
                "https://senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026"
            ),
            "https://www.senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
        )
        self.assertEqual(
            _canonicalize_senat_url(
                "https://www.senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026"
            ),
            "https://www.senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
        )

    def test_looks_like_senat_legislation_search_page(self):
        html = """
        <html><body>
          <input name="ctl00$B_Center$Lista$txtNri" />
          <select name="ctl00$B_Center$Lista$ddAni"></select>
          <h4>Rezultate Cautare</h4>
        </body></html>
        """
        self.assertTrue(_looks_like_senat_legislation_search_page(html))
        self.assertFalse(_looks_like_senat_legislation_search_page("<html>plain</html>"))

    def test_fetch_senat_search_results_html_submits_search_postback(self):
        initial_html = """
        <html><body>
          <form method="post" action="./lista.aspx?nr_cls=b135&amp;an_cls=2026" id="aspnetForm">
            <input type="hidden" name="__VIEWSTATE" value="vs123" />
            <input type="hidden" name="__VIEWSTATEGENERATOR" value="gen456" />
            <input name="ctl00$B_Center$Lista$txtNri" />
            <select name="ctl00$B_Center$Lista$ddAni"><option selected value="2026">2026</option></select>
            <a id="ctl00_B_Center_Lista_btnCauta2" href="javascript:__doPostBack('ctl00$B_Center$Lista$btnCauta2','')">Caută</a>
            <h4>Rezultate Cautare</h4>
            <table><tr><td>Nu au fost găsite înregistrări.</td></tr></table>
          </form>
        </body></html>
        """
        results_html = """
        <html><body>
          <a href="/legis/PDF/2026/26L257EM.PDF?nocache=true">expunerea de motive</a>
        </body></html>
        """

        class _Headers:
            @staticmethod
            def get_content_charset():
                return "utf-8"

        class _Response:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")
                self.headers = _Headers()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self._body

        class _FakeOpener:
            def __init__(self):
                self.requests = []

            def open(self, request, timeout=30):
                self.requests.append(request)
                if len(self.requests) == 1:
                    return _Response(initial_html)
                return _Response(results_html)

        fake_opener = _FakeOpener()
        fetcher = MagicMock()
        fetcher.retries = 0
        fetcher.sleep_seconds = 0
        fetcher.timeout = 30
        fetcher.user_agent = "votez-test-agent"

        with patch("crawl_deputy_activity.urllib.request.build_opener", return_value=fake_opener):
            html = _fetch_senat_search_results_html(
                fetcher,
                "https://senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
                search_number="b135",
                search_year="2026",
            )

        self.assertIn("26L257EM.PDF", html)
        self.assertEqual(len(fake_opener.requests), 2)
        post_request = fake_opener.requests[1]
        self.assertEqual(post_request.get_method(), "POST")
        self.assertEqual(
            post_request.full_url,
            "https://www.senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
        )
        payload = post_request.data.decode("utf-8")
        self.assertIn("__EVENTTARGET=ctl00%24B_Center%24Lista%24btnCauta2", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24txtNri=b135", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24ddAni=2026", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24ddStadiu=", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24ddTipInitiativa=", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24ddIni=", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24txtIns=", payload)
        self.assertIn("ctl00%24B_Center%24Lista%24ddPri=", payload)

    def test_law_initiator_match_log_suffix_chamber_breakdown(self):
        text = "Inițiatori\nX\n"
        suffix = _law_initiator_match_log_suffix(
            text,
            matched_member_ids=["deputat_1", "senator_2"],
            deputy_rows=[("deputat_1", "A", "a")],
            senator_rows=[("senator_2", "B", "b")],
        )
        self.assertIn("1 deputati", suffix)
        self.assertIn("1 senatori", suffix)

    def test_law_initiator_match_log_suffix_no_one(self):
        text = "Inițiatori\nXyzzyplugh Abcdef PSD\n"
        suffix = _law_initiator_match_log_suffix(
            text,
            matched_member_ids=[],
            deputy_rows=[("deputat_1", "Ion Popescu", "ion popescu")],
            senator_rows=[("sen_1", "Ion Popescu", "ion popescu")],
        )
        self.assertIn("not matched to any deputat/senator", suffix)
        self.assertIn("Xyzzyplugh", suffix)

    def test_law_initiator_match_log_suffix_no_name_lines(self):
        suffix = _law_initiator_match_log_suffix(
            "Inițiatori\n",
            matched_member_ids=[],
            deputy_rows=[("deputat_1", "Ion Popescu", "ion popescu")],
            senator_rows=[],
        )
        self.assertIn("no initiator names found", suffix)

    def test_hydrate_law_initiators_skips_when_is_initiator_linked(self):
        """With skip_if_initiator_linked, do not fetch/OCR if dep_act_member_laws has is_initiator=1."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:test:init_skip",
                        "https://example.test/law/init-skip",
                        "PL 1/2026",
                        "Title",
                        "Details",
                        "[]",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO dep_act_member_laws (member_id, law_id, is_initiator)
                    VALUES ('deputat_1', 'law:test:init_skip', 1)
                    """,
                )
                conn.commit()
            fetcher = MagicMock()
            fetcher.fetch.side_effect = RuntimeError(
                "fetch should not run when initiator linked"
            )
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                hydrate_law_initiators_for_records(
                    conn,
                    records=[
                        ListingRecord(
                            record_id="law:test:init_skip",
                            source_url="https://example.test/law/init-skip",
                            identifier=None,
                            adopted_law_identifier=None,
                            title="Title",
                            details_text="Details",
                            columns=[],
                        )
                    ],
                    fetcher=fetcher,
                    member_rows=[("deputat_1", "Test", "test")],
                    force=True,
                    skip_if_initiator_linked=True,
                )
            fetcher.fetch.assert_not_called()

    def test_hydrate_law_initiators_uses_cached_pdf_without_fetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            cache_path = _law_initiator_pdf_cache_path(
                cache_dir,
                law_id="law:test:cached",
                identifier="PL-x 1/2026",
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"%PDF-1.4 cached")

            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:test:cached",
                        "https://example.test/law/cached",
                        "PL-x 1/2026",
                        "Title",
                        "Details",
                        "[]",
                    ),
                )
                conn.commit()

            fetcher = MagicMock()
            fetcher.fetch.side_effect = AssertionError("law page fetch must not run")
            fetcher.fetch_bytes.side_effect = AssertionError("pdf download must not run")

            with patch(
                "crawl_deputy_activity.extract_pdf_text_local",
                return_value="Inițiatori:\nDeputat Test\nExpunerea de motive",
            ):
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    hydrate_law_initiators_for_records(
                        conn,
                        records=[
                            ListingRecord(
                                record_id="law:test:cached",
                                source_url="https://example.test/law/cached",
                                identifier="PL-x 1/2026",
                                adopted_law_identifier=None,
                                title="Title",
                                details_text="Details",
                                columns=[],
                            )
                        ],
                        fetcher=fetcher,
                        member_rows=[("deputat_1", "Deputat Test", "deputat test")],
                        force=True,
                        skip_if_initiator_linked=False,
                        law_initiator_pdf_dir=cache_dir,
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT initiators_text FROM dep_act_laws WHERE law_id = ?",
                        ("law:test:cached",),
                    ).fetchone()
                    self.assertIn("Deputat Test", row[0] or "")
            fetcher.fetch.assert_not_called()
            fetcher.fetch_bytes.assert_not_called()

    def test_hydrate_law_initiators_downloads_pdf_into_local_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            cache_path = _law_initiator_pdf_cache_path(
                cache_dir,
                law_id="law:test:download",
                identifier="PL-x 2/2026",
            )
            pdf_bytes = b"%PDF-1.4 downloaded"
            law_page_html = """
            <table>
              <tr>
                <td>Expunerea de motive</td>
                <td><a href="/pdfs/init.pdf">PDF</a></td>
              </tr>
            </table>
            """

            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:test:download",
                        "https://example.test/law/download",
                        "PL-x 2/2026",
                        "Title",
                        "Details",
                        "[]",
                    ),
                )
                conn.commit()

            fetcher = MagicMock()
            fetcher.fetch.return_value = law_page_html
            fetcher.fetch_bytes.return_value = pdf_bytes

            with patch(
                "crawl_deputy_activity.extract_pdf_text_local",
                return_value="Inițiatori:\nDeputat Test\nExpunerea de motive",
            ):
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    hydrate_law_initiators_for_records(
                        conn,
                        records=[
                            ListingRecord(
                                record_id="law:test:download",
                                source_url="https://example.test/law/download",
                                identifier="PL-x 2/2026",
                                adopted_law_identifier=None,
                                title="Title",
                                details_text="Details",
                                columns=[],
                            )
                        ],
                        fetcher=fetcher,
                        member_rows=[("deputat_1", "Deputat Test", "deputat test")],
                        force=True,
                        skip_if_initiator_linked=False,
                        law_initiator_pdf_dir=cache_dir,
                    )
                    conn.commit()

            self.assertTrue(cache_path.exists())
            self.assertEqual(cache_path.read_bytes(), pdf_bytes)
            fetcher.fetch.assert_called_once_with("https://example.test/law/download")
            fetcher.fetch_bytes.assert_called_once_with("https://example.test/pdfs/init.pdf")

    def test_extract_and_match_law_initiators_text(self):
        text = """
        FIŞĂ ACT NORMATIV
        Iniţiatori:
        Alecsandru Marius-Nicolae
        Bende Sándor
        Expunerea de motive
        Restul documentului.
        """
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("deputat_3", "Alecsandru Marius-Nicolae", "alecsandru marius nicolae"),
                ("deputat_25", "Bende Sándor", "bende sandor"),
                ("deputat_99", "Alt Deputat", "alt deputat"),
            ],
        )

        self.assertIn("Alecsandru Marius-Nicolae", section)
        self.assertEqual(matched, ["deputat_3", "deputat_25"])

    def test_extract_initiators_initiatorilor_phrase(self):
        text = """
        În numele initiatorilor, Stefan-Ovidiu Popa, deputat PSD
        Expunerea de motive
        Corpul documentului.
        """
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("deputat_1", "Stefan-Ovidiu Popa", "stefan ovidiu popa"),
                ("deputat_2", "Alt Deputat", "alt deputat"),
            ],
        )
        self.assertIn("Stefan-Ovidiu Popa", section)
        self.assertEqual(matched, ["deputat_1"])

    def test_extract_initiators_numbered_table_rows_skip_party_suffix(self):
        text = """
        INITIATORI Numele și prenumele Grupul parlamentar Semnătura
        1 Ciunt Ionel PSD
        2 Chilat Crina-Fiorela PSD
        Expunerea de motive
        Urmează textul.
        """
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("deputat_10", "Ciunt Ionel", "ciunt ionel"),
                ("deputat_11", "Chilat Crina-Fiorela", "chilat crina fiorela"),
            ],
        )
        self.assertEqual(matched, ["deputat_10", "deputat_11"])

    def test_extract_initiators_excludes_handwritten_supporter_noise(self):
        text = """
        INITIATORI
        1 Popescu Ion PSD
        Ga h dnoweache PN | Noise xx
        Expunerea de motive
        """
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("deputat_a", "Popescu Ion", "popescu ion"),
                ("deputat_b", "Noise", "noise"),
            ],
        )
        self.assertEqual(matched, ["deputat_a"])
        self.assertNotIn("dnoweache", section)

    def test_extract_initiators_handwritten_like_before_sustinatori_zone(self):
        """OCR that looks handwritten is accepted only above Susținători, outside that table."""
        text = """
        Inițiatori
        Text scurt înainte de lista de susținere.
        AbCdEfGhIj KlMnOpQrSt
        Susținători
        1 Semnatar Unu PSD
        Expunerea de motive
        """
        section = extract_initiators_section(text)
        self.assertIn(_INITIATOR_BEFORE_SUSTINATORI_NOTE, section)
        self.assertIn("AbCdEfGhIj KlMnOpQrSt", section)
        matched = match_initiator_member_ids(
            section,
            [
                ("deputat_hw", "AbCdEfGhIj KlMnOpQrSt", "abcdefghij klmnopqrst"),
                ("deputat_sig", "Semnatar Unu", "semnatar unu"),
            ],
        )
        self.assertEqual(matched, ["deputat_hw"])
        self.assertNotIn("Semnatar", section)

    def test_extract_initiators_numbered_row_before_sustinatori_relaxed_ocr(self):
        """Number + party row before Susținători may carry a name that fails printed-table OCR."""
        text = """
        INITIATORI
1 AbCdEfGhIj KlMnOp PSD
        Susținători
        2 AltcinevaUSR
        Expunerea de motive
        """
        section = extract_initiators_section(text)
        self.assertIn(_INITIATOR_BEFORE_SUSTINATORI_NOTE, section)
        self.assertIn("AbCdEfGhIj KlMnOp", section)
        matched = match_initiator_member_ids(
            section,
            [
                ("dep_row", "AbCdEfGhIj KlMnOp", "abcdefghij klmnop"),
 ],
        )
        self.assertEqual(matched, ["dep_row"])

    def test_extract_initiators_senat_deputat_party_prefix_one_line_ocr(self):
        """Senat EM: 'Deputat/Senator PARTY Nume' with OCR row noise and '|' separators."""
        text = (
            "INITIATORI: "
            "Deputat AUR Florin-Cornel Popovici 4 "
            "Deputat AUR Doru Lucian Musat _ a "
            "2 Senator AUR Petrisor-Gabriel Peiu _ | "
            "Deputat AUR Veronica Grosu i "
            "Deputat AUR Andra-Claudia Constantinescu t "
            "LISTA Sustinatorilor propunere Expunerea de motive"
        )
        section = extract_initiators_section(text)
        for name in (
            "Florin-Cornel Popovici",
            "Doru Lucian Musat",
            "Petrisor-Gabriel Peiu",
            "Veronica Grosu",
            "Andra-Claudia Constantinescu",
        ):
            self.assertIn(name, section)

    def test_extract_initiators_senat_numele_initiatorilor_surname_first(self):
        """Senat EM: prose header plus SURNAME Given-Names (not Deputat PARTY …), before Lista susținătorilor."""
        text = """
 În numele inițiatorilor,
        Senator RUS Vasile-Ciprian ; Deputat BERESCU Monica-Elena ;
        BULEARCA Marius-Felix ; Deputat MARUSSIGeorge-Nicolae
        LISTA SUSTINATORILOR propunere legislative
        Pagina 6 din 6
        Expunerea de motive
        """
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("m1", "RUS Vasile-Ciprian", "rus vasile ciprian"),
                ("m2", "BERESCU Monica-Elena", "berescu monica elena"),
                ("m3", "BULEARCA Marius-Felix", "bulearca marius felix"),
                ("m4", "MARUSSI George-Nicolae", "marussi george nicolae"),
            ],
        )
        self.assertEqual(
            matched,
            ["m1", "m2", "m3", "m4"],
            msg=section,
        )
        self.assertNotIn("Pagina", section)

    def test_extract_initiators_skips_initiatorii_exprima_body_picks_signature_block(self):
        """EM body may say 'Inițiatorii își exprimă…'; real list is under 'În numele inițiatorilor'."""
        text = (
            "8. Concluzii. Inițiatorii își exprimă susținerea pentru prezentul proiect de lege "
            "și semnează prezentul punct. "
            "În numele inițiatorilor, Senator RUS Vasile-Ciprian ; Deputat BERESCU Monica-Elena ; "
            "BULEARCA Marius-Felix ; Deputat MARUSSI George-Nicolae "
            "LISTA SUSTINATORILOR propunere Expunerea de motive"
        )
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("m1", "RUS Vasile-Ciprian", "rus vasile ciprian"),
                ("m2", "BERESCU Monica-Elena", "berescu monica elena"),
                ("m3", "BULEARCA Marius-Felix", "bulearca marius felix"),
                ("m4", "MARUSSI George-Nicolae", "marussi george nicolae"),
            ],
        )
        self.assertEqual(matched, ["m1", "m2", "m3", "m4"], msg=section)
        self.assertNotIn("exprim", section.lower())

    def test_extract_initiators_numele_initiatorilor_without_in_when_ocrd(self):
        """If 'În' is lost in OCR, 'numele inițiatorilor' still anchors the signature block."""
        text = (
            "Inițiatorii își exprimă susținerea pentru prezentul proiect de lege. "
            "numele inițiatorilor, Senator RUS Vasile-Ciprian ; Deputat BERESCU Monica-Elena ; "
            "LISTA SUSTINATORILOR Expunerea de motive"
        )
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("m1", "RUS Vasile-Ciprian", "rus vasile ciprian"),
                ("m2", "BERESCU Monica-Elena", "berescu monica elena"),
            ],
        )
        self.assertEqual(matched, ["m1", "m2"], msg=section)

    def test_extract_initiators_skips_initiatorii_propunerii_legislative_paragraph(self):
        """Do not anchor on 'initiatorii propunerii legislative…' in EM body."""
        text = (
            "initiatorii propunerii legislative pentru modificarea art. 1 din Lege. "
            "În numele inițiatorilor, Deputat AUR Ion-Popescu ; "
            "LISTA SUSTINATORILOR Expunerea de motive"
        )
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [("d1", "Ion-Popescu", "ion popescu")],
        )
        self.assertEqual(matched, ["d1"], msg=section)
        self.assertNotIn("propunerii", section.lower())

    def test_extract_initiators_skips_in_numele_then_lista_without_names(self):
        """OCR-only 'în numele… Lista' block before the real signature is ignored."""
        text = (
            "in numele initiatorilor, 6 Lista iniSatorilor Propunerii legislative - X. "
            "În numele inițiatorilor, Senator RUS Vasile-Ciprian ; Deputat BERESCU Monica-Elena ; "
            "LISTA SUSTINATORILOR Expunerea de motive"
        )
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("m1", "RUS Vasile-Ciprian", "rus vasile ciprian"),
                ("m2", "BERESCU Monica-Elena", "berescu monica elena"),
            ],
        )
        self.assertEqual(matched, ["m1", "m2"], msg=section)

    def test_extract_initiators_numele_ocrd_camel_case_header(self):
        """OCR garbling like 'numeLe initiatoriLor' still matches the numele anchor."""
        text = (
            "numeLe initiatoriLor; Senator RUS Vasile-Ciprian ; Deputat MARUSSI George-Nicolae ; "
            "LISTA SUSTINATORILOR Expunerea de motive"
        )
        section = extract_initiators_section(text)
        self.assertIn("RUS Vasile-Ciprian", section)
        self.assertIn("MARUSSI George-Nicolae", section)

    def test_match_initiator_member_ids_fuzzy_ocr_surname(self):
        """OCR drops trailing letters (e.g. Musat → Musa) but fuzzy match still links."""
        section = (
            "În numele inițiatorilor\n"
            "Deputat AUR Doru-Lucian Musa\n"
        )
        matched = match_initiator_member_ids(
            section,
            [
                ("d1", "Doru-Lucian Musat", "doru lucian musat"),
                ("d2", "Alt Deputat", "alt deputat"),
            ],
        )
        self.assertEqual(matched, ["d1"])
        self.assertTrue(
            _fuzzy_member_matches_initiator_lines(
                ["deputat aur doru-lucian musa"],
                "doru lucian musat",
            )
        )

    def test_embedded_pdf_missing_initiator_name_signals_hollow(self):
        """Initiator header in PDF text but no plausible name lines (OCR fallback)."""
        filler = "Propunere legislativa cu mult text de corp. " * 5
        direct = filler + " INITIATORI: LISTA Sustin5torilor propunere fara nume tiparite."
        self.assertTrue(_embedded_pdf_missing_initiator_name_signals(direct))

    def test_embedded_pdf_expunere_without_initiator_header_is_hollow(self):
        """Body text embedded (Expunere de motive) but initiator block image-only → OCR."""
        filler = "Actualul context economic international si alte paragrafe. " * 12
        direct = filler + " EXPUNERE DE MOTIVE continut. LISTA Sustin5torilor fara INITIATOR."
        self.assertTrue(_embedded_pdf_missing_initiator_name_signals(direct))

    def test_embedded_pdf_missing_initiator_name_signals_not_hollow_when_deputat_line(
        self,
    ):
        filler = "Text de umplutura din PDF ca sa depaseasca pragul de caractere. " * 3
        direct = (
            filler
            + " INITIATORI: Deputat AUR Ion Popescu "
            "LISTA Sustinatorilor"
        )
        self.assertFalse(_embedded_pdf_missing_initiator_name_signals(direct))

    def test_extract_initiators_finds_header_after_long_lead_in(self):
        """Inițiatori may appear after cover text from earlier PDF pages."""
        filler = ("Pagina de gardă cu mult text. " * 400) + " INITIATORI "
        text = (
            filler
            + "\n2 Ionescu Maria PSD\nExpunerea de motive\n"
        )
        section = extract_initiators_section(text)
        matched = match_initiator_member_ids(
            section,
            [
                ("dep_x", "Ionescu Maria", "ionescu maria"),
            ],
        )
        self.assertEqual(matched, ["dep_x"])

    def test_parse_question_listing_extracts_identifier_and_clean_text(self):
        html = """
        <table>
          <tr><th>Nr.</th><th>Interpelare</th></tr>
          <tr>
            <td>1.</td>
            <td>
              <a href="/pls/parlam/interpelari2015.detalii?id=819">
                Interpelarea nr.819B/25-11-2025
              </a>
              Stadiul analizării propunerilor de instituire a zonelor de protecție strictă în județul Suceava
            </td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=7",
            "questions",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].identifier, "nr.819B/25-11-2025")
        self.assertEqual(
            records[0].details_text,
            "Stadiul analizării propunerilor de instituire a zonelor de protecție strictă în județul Suceava",
        )
        self.assertEqual(records[0].title, records[0].details_text)

    def test_parse_question_detail_recipient(self):
        html = """
        <table>
          <tr valign="top">
            <td>Destinatar:</td>
            <td>
              <b>Ministerul Mediului, Apelor şi Pădurilor</b>
              <br>în atenţia:
              doamnei
              <b>Diana-Anda Buzoianu</b>
              - Ministru
            </td>
          </tr>
        </table>
        """

        self.assertEqual(
            parse_question_detail_recipient(html),
            "Ministerul Mediului, Apelor şi Pădurilor în atenţia: doamnei Diana-Anda Buzoianu - Ministru",
        )

    def test_parse_question_detail_multiple_recipients(self):
        html = """
        <table>
          <tr>
            <td>Destinatari:</td>
            <td>Ministerul Investițiilor si Proiectelor Europene
            <br>în atenţia: domnului <b>Dragoş Nicolae Pîslaru</b> - Ministru</td>
          </tr>
          <tr>
            <td></td>
            <td>Ministerul Economiei, Digitalizării, Antreprenoriatului și Turismului
            <br>în atenţia: domnului <b>Ambrozie-Irineu Darău</b> - Ministru</td>
          </tr>
          <tr><td>Textul intervenţiei:</td><td>fişier PDF</td></tr>
        </table>
        """

        self.assertEqual(
            parse_question_detail_recipient(html),
            "Ministerul Investițiilor si Proiectelor Europene în atenţia: domnului Dragoş Nicolae Pîslaru - Ministru | "
            "Ministerul Economiei, Digitalizării, Antreprenoriatului și Turismului în atenţia: domnului Ambrozie-Irineu Darău - Ministru",
        )

    def test_hydrate_question_records_fetches_recipient(self):
        records = [
            ListingRecord(
                record_id="question:cdep:819",
                source_url="https://example.test/question?id=819",
                identifier="nr.819B/25-11-2025",
                title="Stadiul analizării propunerilor",
                details_text="Stadiul analizării propunerilor",
                columns=["1", "Interpelarea nr.819B/25-11-2025"],
            )
        ]

        class FakeFetcher:
            def fetch(self, url: str) -> str:
                self.url = url
                return """
                <table>
                  <tr><td>Obiectul întrebării:</td><td>Stadiul analizării propunerilor</td></tr>
                  <tr>
                    <td>Destinatar:</td>
                    <td><b>Ministerul Mediului, Apelor şi Pădurilor</b><br>în atenţia:
                    doamnei <b>Diana-Anda Buzoianu</b> - Ministru</td>
                  </tr>
                </table>
                """

        hydrated = hydrate_question_records(
            listing_records=records,
            fetcher=FakeFetcher(),
        )

        self.assertEqual(len(hydrated), 1)
        self.assertEqual(
            hydrated[0].recipient,
            "Ministerul Mediului, Apelor şi Pădurilor în atenţia: doamnei Diana-Anda Buzoianu - Ministru",
        )

    def test_parse_political_declaration_listing_and_detail_text_link(self):
        listing_html = """
        <table>
          <tr><th>Data</th><th>Titlu</th></tr>
          <tr>
            <td>2025-02-12</td>
            <td>
              <a href="/pls/steno/stenograma_scris?ids=98&idm=1,035&idl=1">
                <img src="/img/icon_go1.gif" border="0" align="right">
              </a>
              <span>
                declaraţie politică referitoare la "Necesitatea intervenţiei statului"
              </span>
            </td>
          </tr>
        </table>
        """
        records = parse_listing_records(
            listing_html,
            "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=1&pag=9",
            "political_declarations",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "political_declaration:cdep:98:1_035")
        self.assertEqual(
            records[0].source_url,
            "https://www.cdep.ro/pls/steno/stenograma_scris?ids=98&idm=1,035&idl=1",
        )

        class FakeFetcher:
            def fetch(self, url: str) -> str:
                self.url = url
                return """
                <html><body>
                  <p align="justify">"Necesitatea intervenţiei statului"</p>
                  <p align="justify">Textul integral al declaraţiei politice.</p>
                </body></html>
                """

        detail_html = """
        <html><body>
          <table>
            <tr>
              <td>Text declaraţie politică</td>
              <td><a href="/pls/steno/steno2015.text?id=77">Afişare text</a></td>
            </tr>
          </table>
        </body></html>
        """
        text_url, title, full_text = parse_political_declaration_detail(
            detail_html,
            "https://www.cdep.ro/pls/steno/steno2015.detalii?id=77",
            fetcher=FakeFetcher(),
        )

        self.assertEqual(
            text_url,
            "https://www.cdep.ro/pls/steno/steno2015.text?id=77",
        )
        self.assertEqual(title, "Necesitatea intervenţiei statului")
        self.assertIn("Textul integral al declaraţiei politice", full_text)

    def test_parse_political_declaration_nested_cdep_listing_title_and_link(self):
        listing_html = """
        <tbody>
          <tr valign="top">
            <td align="center"><b><a href="steno2024.sumar_scris?ids=96&amp;idl=1">2</a>.</b></td>
            <td colspan="3" bgcolor="#fef9c2">
              <table class="innertable"><tbody><tr>
                <td><b>Sedinta Camerei Deputatilor din 11 martie 2026</b></td>
              </tr></tbody></table>
            </td>
            <td><a href="steno2024.stenograma?ids=96&amp;idv=4699&amp;idl=1">
              <img src="/img/icon_elenco.gif">
            </a></td>
          </tr>
          <tr valign="top">
            <td>&nbsp;</td>
            <td><a href="/pls/steno/steno2024.stenograma_scris?ids=96&amp;idm=1&amp;idl=1">
              <img src="/img/icon_go2.gif">
            </a></td>
            <td colspan="2">Declaraţii politice şi intervenţii ale deputaţilor:</td>
            <td><a href="steno2024.stenograma_scris?ids=96&amp;idm=1&amp;idv=4699&amp;idl=1">
              <img src="/img/icon_elenco.gif">
            </a></td>
          </tr>
          <tr valign="top">
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td><a href="/pls/steno/steno2024.stenograma_scris?ids=96&amp;idm=1,029&amp;idl=1">
              <img src="/img/icon_go1.gif">
            </a></td>
            <td bgcolor="#fffef2">
              <table class="innertable"><tbody><tr>
                <td>
                  <a href="/pls/parlam/structura2015.mp?idm=1&amp;cam=2&amp;leg=2024">
                    Mirela Elena Adomnicăi
                  </a>
                  - declaraţie politică privind
                  "Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice";
                </td>
                <td nowrap="">în scris</td>
              </tr></tbody></table>
            </td>
            <td><a href="steno2024.stenograma_scris?ids=96&amp;idm=1,029&amp;sir=&amp;idv=4699&amp;idl=1">
              <img src="/img/icon_elenco.gif">
            </a></td>
          </tr>
        </tbody>
        """
        records = parse_listing_records(
            listing_html,
            "https://www.cdep.ro/pls/steno/steno2015.lista?idv=4699",
            "political_declarations",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "political_declaration:cdep:96:1_029")
        self.assertEqual(
            records[0].source_url,
            "https://www.cdep.ro/pls/steno/steno2024.stenograma_scris?ids=96&idm=1,029&idl=1",
        )
        self.assertEqual(
            records[0].title,
            "Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice",
        )

    def test_hydrate_political_declaration_uses_listing_title_and_detail_body(self):
        listing_record = ListingRecord(
            record_id="political_declaration:cdep:96:1_029",
            source_url="https://www.cdep.ro/pls/steno/steno2024.stenograma_scris?ids=96&idm=1,029&idl=1",
            identifier=None,
            title="Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice",
            details_text='Mirela Elena Adomnicăi - declaraţie politică privind "Responsabilitatea instituţiilor statului"; în scris',
            columns=[],
        )

        class FakeFetcher:
            def fetch(self, url: str) -> str:
                self.url = url
                return """
                <html><body>
                  <td width="100%">
                    <p align="justify"><b><a href="/pls/parlam/structura2015.mp?idm=1">Doamna Mirela Elena Adomnicăi</a></b>:</p>
                    <p align="justify">"Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice"</p>
                    <p align="justify">Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.</p>
                    <p align="justify">România are nevoie de instituţii solide.</p>
                  </td>
                </body></html>
                """

        declarations = hydrate_political_declaration_records(
            member_id="deputat_1",
            listing_records=[listing_record],
            fetcher=FakeFetcher(),
        )

        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0].title, listing_record.title)
        self.assertEqual(
            declarations[0].full_text,
            "Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.\n\n"
            "România are nevoie de instituţii solide.",
        )

    def test_political_declaration_detail_page_parses_current_individual_page(self):
        detail_url = "https://www.cdep.ro/pls/steno/steno2024.stenograma_scris?ids=96&idm=1,029&idl=1"
        detail_html = """
        <html><body>
          <table>
            <tr>
              <td><a href="/pls/steno/steno2024.stenograma_scris?ids=96&amp;idm=1&amp;idl=1">Secţiune declaraţii</a></td>
            </tr>
          </table>
          <td width="100%">
            <p align="justify"><b><a href="/pls/parlam/structura2015.mp?idm=1">Doamna Mirela Elena Adomnicăi</a></b>:</p>
            <p align="justify">"Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice"</p>
            <p align="justify">Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.</p>
          </td>
        </body></html>
        """

        class FailingFetcher:
            def fetch(self, url: str) -> str:
                raise AssertionError(f"Unexpected extra fetch: {url}")

        text_url, title, full_text = parse_political_declaration_detail(
            detail_html,
            detail_url,
            fetcher=FailingFetcher(),
        )

        self.assertEqual(text_url, detail_url)
        self.assertEqual(
            title,
            "Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice",
        )
        self.assertEqual(
            full_text,
            "Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.",
        )

    def test_parse_political_declaration_text_page_title_and_body(self):
        html = """
        <html><body>
          <table>
            <tr><td>
              <p align="justify"></p>
              <p align="justify">Doamna Mirela Elena Adomnicăi:</p>
              <p align="justify">"Necesitatea intervenţiei statului pentru plafonarea preţului carburanţilor"</p>
              <p align="justify">Creşterea accelerată a preţurilor la carburanţi reprezintă una dintre cele mai presante probleme.</p>
              <p align="justify">În primul rând, carburanţii nu sunt un bun de lux.</p>
            </td></tr>
          </table>
        </body></html>
        """
        title, body = parse_political_declaration_text_page(html)

        self.assertEqual(
            title,
            "Necesitatea intervenţiei statului pentru plafonarea preţului carburanţilor",
        )
        self.assertEqual(
            body,
            "Creşterea accelerată a preţurilor la carburanţi reprezintă una dintre cele mai presante probleme.\n\n"
            "În primul rând, carburanţii nu sunt un bun de lux.",
        )

    def test_parse_political_declaration_text_page_with_unclosed_cdep_paragraphs(self):
        html = """
        <html><body>
          <td width="100%">
            <!-- START=5403,, -->
            <p align="justify"><b><a href="/pls/parlam/structura2015.mp?idm=1">Doamna Mirela Elena Adomnicăi</a></b>:
            <p align="justify">"Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice"
            <p align="justify">Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.
            <p align="justify">România are nevoie de instituţii solide.
            <!-- END -->
          </td>
          <p>Palatul Parlamentului, str.Izvor nr.2-4, sect.5, Bucuresti</p>
          <p>Copyright © Camera Deputaţilor</p>
        </body></html>
        """
        title, body = parse_political_declaration_text_page(html)

        self.assertEqual(
            title,
            "Responsabilitatea instituţiilor statului şi protejarea copiilor de conflictele politice",
        )
        self.assertEqual(
            body,
            "Intervenţia mea are ca punct de plecare o situaţie care a generat dezbateri intense.\n\n"
            "România are nevoie de instituţii solide.",
        )

    def test_init_db_renames_legacy_activity_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute(
                    """
                    CREATE TABLE members (
                        member_id TEXT PRIMARY KEY,
                        source_member_id TEXT NOT NULL,
                        chamber TEXT NOT NULL,
                        name TEXT NOT NULL,
                        normalized_name TEXT NOT NULL,
                        party_id TEXT,
                        bills_authored_total INTEGER NOT NULL DEFAULT 0,
                        amendments_added_total INTEGER NOT NULL DEFAULT 0,
                        profile_url TEXT,
                        circumscriptie TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO members (
                        member_id, source_member_id, chamber, name, normalized_name
                    )
                    VALUES ('deputat_1', '1', 'deputat', 'Deputat Test', 'deputat test')
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE laws (
                        law_id TEXT PRIMARY KEY,
                        source_url TEXT NOT NULL UNIQUE,
                        identifier TEXT,
                        title TEXT NOT NULL,
                        details_text TEXT NOT NULL,
                        columns_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    )
                    VALUES (
                        'law:cdep:1', 'https://example.test/law/1', 'PL-x 1/2025',
                        'Legacy law', 'Legacy details', '[]'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE member_laws (
                        member_id TEXT NOT NULL,
                        law_id TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (member_id) REFERENCES members(member_id),
                        FOREIGN KEY (law_id) REFERENCES laws(law_id),
                        PRIMARY KEY (member_id, law_id)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO member_laws (member_id, law_id)
                    VALUES ('deputat_1', 'law:cdep:1')
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE motions (
                        motion_id TEXT PRIMARY KEY,
                        source_url TEXT NOT NULL UNIQUE,
                        title TEXT NOT NULL,
                        details_text TEXT NOT NULL,
                        columns_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO motions (
                        motion_id, source_url, title, details_text, columns_json
                    )
                    VALUES (
                        'motion:cdep:1', 'https://example.test/motion/1',
                        'Legacy motion', 'Legacy motion details', '[]'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE member_motions (
                        member_id TEXT NOT NULL,
                        motion_id TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (member_id) REFERENCES members(member_id),
                        FOREIGN KEY (motion_id) REFERENCES motions(motion_id),
                        PRIMARY KEY (member_id, motion_id)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO member_motions (member_id, motion_id)
                    VALUES ('deputat_1', 'motion:cdep:1')
                    """
                )

            init_db(db_path)

            with sqlite3.connect(db_path) as conn:
                names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                self.assertNotIn("laws", names)
                self.assertNotIn("motions", names)
                self.assertIn("dep_act_laws", names)
                self.assertIn("dep_act_member_laws", names)
                self.assertIn("dep_act_motions", names)
                self.assertIn("dep_act_member_motions", names)
                self.assertEqual(
                    conn.execute(
                        "SELECT title FROM dep_act_laws WHERE law_id = 'law:cdep:1'"
                    ).fetchone()[0],
                    "Legacy law",
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT title FROM dep_act_motions WHERE motion_id = 'motion:cdep:1'"
                    ).fetchone()[0],
                    "Legacy motion",
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM dep_act_member_laws "
                        "WHERE member_id = 'deputat_1' AND law_id = 'law:cdep:1'"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM dep_act_member_motions "
                        "WHERE member_id = 'deputat_1' AND motion_id = 'motion:cdep:1'"
                    ).fetchone()[0],
                    1,
                )
                law_fk_targets = {
                    row[2]
                    for row in conn.execute(
                        "PRAGMA foreign_key_list(dep_act_member_laws)"
                    )
                }
                motion_fk_targets = {
                    row[2]
                    for row in conn.execute(
                        "PRAGMA foreign_key_list(dep_act_member_motions)"
                    )
                }
                self.assertEqual(law_fk_targets, {"members", "dep_act_laws"})
                self.assertEqual(motion_fk_targets, {"members", "dep_act_motions"})

    def test_store_records_deduplicates_laws_and_associations(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                records = [
                    ListingRecord(
                        record_id="law:cdep:22110",
                        source_url="https://example.test/law?idp=22110",
                        identifier="PL-x 44/2025",
                        adopted_law_identifier="Lege 233/2025",
                        title="PL-x 44/2025",
                        details_text="PL-x 44/2025 privind testarea",
                        columns=[
                            "1",
                            "PL-x 44/2025",
                            "privind testarea",
                            "Lege 233/2025",
                        ],
                    )
                ]

                first = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="laws",
                    records=records,
                )
                second = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="laws",
                    records=records,
                )
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity={},
                    results={"laws": first},
                )

                self.assertEqual(first.stored, 1)
                self.assertEqual(second.stored, 0)
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM dep_act_laws").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM dep_act_member_laws").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT is_initiator FROM dep_act_member_laws"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT adopted_law_identifier FROM dep_act_laws"
                    ).fetchone()[0],
                    "Lege 233/2025",
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT legislative_proposals_stored FROM dep_act_member_activity_crawl"
                    ).fetchone()[0],
                    1,
                )
                law_records = load_law_records_for_initiator_hydration(
                    conn,
                    member_ids=["deputat_1"],
                )
                self.assertEqual(len(law_records), 1)
                self.assertEqual(law_records[0].record_id, "law:cdep:22110")
                self.assertEqual(
                    law_records[0].source_url,
                    "https://example.test/law?idp=22110",
                )
                conn.execute(
                    """
                    UPDATE dep_act_member_laws
                    SET is_initiator = 1
                    WHERE member_id = 'deputat_1' AND law_id = 'law:cdep:22110'
                    """
                )
                self.assertEqual(
                    load_law_records_for_initiator_hydration(
                        conn,
                        member_ids=["deputat_1"],
                    ),
                    [],
                )
                self.assertEqual(
                    len(
                        load_law_records_for_initiator_hydration(
                            conn,
                            member_ids=["deputat_1"],
                            only_without_initiators=False,
                        )
                    ),
                    1,
                )

    def test_load_law_records_for_initiator_hydration_honors_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                for idx in range(3):
                    law_id = f"law:test:limit:{idx}"
                    conn.execute(
                        """
                        INSERT INTO dep_act_laws (
                            law_id, source_url, identifier, title, details_text, columns_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            law_id,
                            f"https://example.test/law/{idx}",
                            f"PL-x {idx}/2026",
                            f"Title {idx}",
                            "Details",
                            "[]",
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO dep_act_member_laws (member_id, law_id)
                        VALUES ('deputat_1', ?)
                        """,
                        (law_id,),
                    )
                conn.commit()

                records = load_law_records_for_initiator_hydration(
                    conn,
                    member_ids=["deputat_1"],
                    limit=2,
                )

                self.assertEqual(len(records), 2)
                self.assertEqual(
                    [record.record_id for record in records],
                    ["law:test:limit:0", "law:test:limit:1"],
                )

    def test_load_law_records_for_initiator_hydration_limit_skips_cached_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                for idx in range(4):
                    law_id = f"law:test:limit-cache:{idx}"
                    identifier = f"PL-x {idx}/2026"
                    conn.execute(
                        """
                        INSERT INTO dep_act_laws (
                            law_id, source_url, identifier, title, details_text, columns_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            law_id,
                            f"https://example.test/law/cache/{idx}",
                            identifier,
                            f"Title {idx}",
                            "Details",
                            "[]",
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO dep_act_member_laws (member_id, law_id)
                        VALUES ('deputat_1', ?)
                        """,
                        (law_id,),
                    )
                _write_bytes_atomic(
                    _law_initiator_pdf_cache_path(
                        cache_dir,
                        law_id="law:test:limit-cache:0",
                        identifier="PL-x 0/2026",
                    ),
                    b"%PDF-1.4 cached 0",
                )
                _write_bytes_atomic(
                    _law_initiator_pdf_cache_path(
                        cache_dir,
                        law_id="law:test:limit-cache:2",
                        identifier="PL-x 2/2026",
                    ),
                    b"%PDF-1.4 cached 2",
                )
                conn.commit()

                records = load_law_records_for_initiator_hydration(
                    conn,
                    member_ids=["deputat_1"],
                    limit=2,
                    law_initiator_pdf_dir=cache_dir,
                    prefer_without_cached_pdf_when_limited=True,
                )

                self.assertEqual(len(records), 2)
                self.assertEqual(
                    [record.record_id for record in records],
                    ["law:test:limit-cache:1", "law:test:limit-cache:3"],
                )

    def test_count_law_initiator_pdf_cache_gaps_counts_missing_cached_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                for idx in range(3):
                    law_id = f"law:test:cache-gap:{idx}"
                    identifier = f"PL-x {idx}/2026"
                    conn.execute(
                        """
                        INSERT INTO dep_act_laws (
                            law_id, source_url, identifier, title, details_text, columns_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            law_id,
                            f"https://example.test/law/gap/{idx}",
                            identifier,
                            f"Title {idx}",
                            "Details",
                            "[]",
                        ),
                    )
                _write_bytes_atomic(
                    _law_initiator_pdf_cache_path(
                        cache_dir,
                        law_id="law:test:cache-gap:1",
                        identifier="PL-x 1/2026",
                    ),
                    b"%PDF-1.4 cached 1",
                )
                conn.commit()

                missing, total = count_law_initiator_pdf_cache_gaps(
                    conn,
                    law_initiator_pdf_dir=cache_dir,
                )

                self.assertEqual((missing, total), (2, 3))

    def test_run_law_initiator_hydration_phase_logs_remaining_cache_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:test:phase:0",
                        "https://example.test/law/phase/0",
                        "PL-x 0/2026",
                        "Title 0",
                        "Details",
                        "[]",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO dep_act_member_laws (member_id, law_id, is_initiator)
                    VALUES ('deputat_1', 'law:test:phase:0', 1)
                    """
                )
                conn.commit()

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    run_law_initiator_hydration_phase(
                        conn,
                        selected=[
                            {
                                "member_id": "deputat_1",
                                "source_member_id": "1",
                                "name": "Deputat Test",
                            }
                        ],
                        fetcher=MagicMock(),
                        update_existing=False,
                        ocr_language="ron",
                        ocr_pages=10,
                        debug=False,
                        law_limit=1,
                        law_initiator_pdf_dir=cache_dir,
                    )

                output = buffer.getvalue()
                self.assertIn(
                    "Law initiator OCR hydration complete: no stored laws to process.",
                    output,
                )
                self.assertIn(
                    "Law initiator OCR hydration cache status: 1/1 stored law(s) still do not have a local initiator PDF",
                    output,
                )

    def test_store_political_declarations(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                declarations = [
                    PoliticalDeclarationRecord(
                        declaration_id="political_declaration:cdep:77",
                        member_id="deputat_1",
                        source_url="https://example.test/detail?id=77",
                        text_url="https://example.test/text?id=77",
                        title='declaraţie politică referitoare la "Necesitatea"',
                        full_text="Text integral al declaraţiei.",
                        details_text="2025-02-12 declaraţie politică",
                        columns=["2025-02-12", "declaraţie politică"],
                    )
                ]

                first = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="political_declarations",
                    records=declarations,
                )
                second = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="political_declarations",
                    records=declarations,
                )
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity={},
                    results={"political_declarations": first},
                )

                self.assertEqual(first.stored, 1)
                self.assertEqual(second.stored, 0)
                self.assertEqual(
                    conn.execute("SELECT member_id, title, full_text FROM dep_act_political_declarations").fetchone(),
                    (
                        "deputat_1",
                        'declaraţie politică referitoare la "Necesitatea"',
                        "Text integral al declaraţiei.",
                    ),
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT political_declarations_stored FROM dep_act_member_activity_crawl"
                    ).fetchone()[0],
                    1,
                )

    def test_store_political_declarations_update_prunes_stale_member_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                stale = [
                    PoliticalDeclarationRecord(
                        declaration_id="political_declarations:url:bad",
                        member_id="deputat_1",
                        source_url="https://example.test/profile",
                        text_url="https://example.test/profile",
                        title="Bad old row",
                        full_text="Copyright © Camera Deputaţilor",
                        details_text="bad",
                        columns=[],
                    )
                ]
                fresh = [
                    PoliticalDeclarationRecord(
                        declaration_id="political_declaration:cdep:96:1_029",
                        member_id="deputat_1",
                        source_url="https://example.test/stenograma_scris?ids=96&idm=1,029&idl=1",
                        text_url="https://example.test/stenograma_scris?ids=96&idm=1,029&idl=1",
                        title="Responsabilitatea instituţiilor statului",
                        full_text="Intervenţia mea are ca punct de plecare.",
                        details_text="fresh",
                        columns=[],
                    )
                ]

                store_records(
                    conn,
                    member_id="deputat_1",
                    kind="political_declarations",
                    records=stale,
                )
                store_records(
                    conn,
                    member_id="deputat_1",
                    kind="political_declarations",
                    records=fresh,
                    update_existing=True,
                )

                self.assertEqual(
                    conn.execute(
                        "SELECT political_declaration_id, title FROM dep_act_political_declarations"
                    ).fetchall(),
                    [
                        (
                            "political_declaration:cdep:96:1_029",
                            "Responsabilitatea instituţiilor statului",
                        )
                    ],
                )

    def test_store_questions_in_single_member_owned_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                records = [
                    ListingRecord(
                        record_id="question:cdep:819",
                        source_url="https://example.test/question?id=819",
                        identifier="nr.819B/25-11-2025",
                        title="Stadiul analizării propunerilor",
                        details_text="Stadiul analizării propunerilor",
                        columns=["1", "Interpelarea nr.819B/25-11-2025"],
                        recipient="Ministerul Mediului, Apelor şi Pădurilor în atenţia: doamnei Diana-Anda Buzoianu - Ministru",
                    )
                ]

                first = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="questions",
                    records=records,
                )
                second = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="questions",
                    records=records,
                )

                self.assertEqual(first.stored, 1)
                self.assertEqual(second.stored, 0)
                self.assertEqual(
                    conn.execute(
                        "SELECT member_id, identifier, recipient, text FROM dep_act_questions_interpellations"
                    ).fetchone(),
                    (
                        "deputat_1",
                        "nr.819B/25-11-2025",
                        "Ministerul Mediului, Apelor şi Pădurilor în atenţia: doamnei Diana-Anda Buzoianu - Ministru",
                        "Stadiul analizării propunerilor",
                    ),
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='member_questions_interpellations'"
                    ).fetchone(),
                    None,
                )

    def test_crawl_member_backfills_existing_question_recipients_without_profile_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_questions_interpellations (
                        question_id,
                        member_id,
                        source_url,
                        identifier,
                        text,
                        columns_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "question:cdep:819",
                        "deputat_1",
                        "https://example.test/question?id=819",
                        "nr.819B/25-11-2025",
                        "Stadiul analizării propunerilor",
                        "[]",
                    ),
                )

                class FakeFetcher:
                    def fetch(self, url: str) -> str:
                        if url == "https://example.test/profile":
                            return """
                            <table>
                              <tr>
                                <td>Declaraţii politice depuse în scris:</td>
                                <td><a href="https://example.test/declarations">0</a></td>
                              </tr>
                            </table>
                            """
                        if url == "https://example.test/declarations":
                            return "<table></table>"
                        if url == "https://example.test/question?id=819":
                            return """
                            <table>
                              <tr>
                                <td>Destinatar:</td>
                                <td><b>Ministerul Mediului</b><br>în atenţia:
                                doamnei <b>Diana-Anda Buzoianu</b> - Ministru</td>
                              </tr>
                            </table>
                            """
                        raise AssertionError(f"Unexpected fetch: {url}")

                _, results, error = crawl_member(
                    conn,
                    member={
                        "member_id": "deputat_1",
                        "name": "Deputat Test",
                        "profile_url": "https://example.test/profile",
                    },
                    fetcher=FakeFetcher(),
                    update_existing=True,
                )

                self.assertIsNone(error)
                self.assertEqual(results["questions"].stored, 1)
                self.assertEqual(
                    conn.execute(
                        "SELECT recipient FROM dep_act_questions_interpellations WHERE question_id = ?",
                        ("question:cdep:819",),
                    ).fetchone()[0],
                    "Ministerul Mediului în atenţia: doamnei Diana-Anda Buzoianu - Ministru",
                )

    def test_crawl_member_uses_standard_motion_page_when_profile_omits_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn, member_id="deputat_2")

                class FakeFetcher:
                    def fetch(self, url: str) -> str:
                        if url == "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024":
                            return """
                            <table>
                              <tr>
                                <td>Declaraţii politice depuse în scris:</td>
                                <td><a href="https://example.test/declarations">0</a></td>
                              </tr>
                            </table>
                            """
                        if "pag=11" in url:
                            return """
                            <table>
                              <tr>
                                <td>1</td>
                                <td>
                                  <a href="/pls/proiecte/upl_pck2015.motiune?id=77">
                                    Moţiune simplă privind testarea parserului
                                  </a>
                                </td>
                              </tr>
                            </table>
                            """
                        return "<table></table>"

                activity, results, error = crawl_member(
                    conn,
                    member={
                        "member_id": "deputat_2",
                        "name": "Deputat Test",
                        "profile_url": "https://www.cdep.ro/pls/parlam/structura2015.mp?idm=3&cam=2&leg=2024",
                    },
                    fetcher=FakeFetcher(),
                    dry_run=True,
                )

                self.assertIsNone(error)
                self.assertEqual(activity["motions"].count, 1)
                self.assertIn("pag=11", activity["motions"].url)
                self.assertEqual(results["motions"].seen, 1)

    def test_store_records_updates_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                original = [
                    ListingRecord(
                        record_id="law:cdep:22110",
                        source_url="https://example.test/law?idp=22110",
                        identifier="PL-x 44/2025",
                        title="Original title",
                        details_text="Original details",
                        columns=["original"],
                    )
                ]
                refreshed = [
                    ListingRecord(
                        record_id="law:cdep:22110",
                        source_url="https://example.test/law?idp=22110",
                        identifier="PL-x 44/2025",
                        title="Refreshed title",
                        details_text="Refreshed details",
                        columns=["refreshed"],
                    )
                ]

                store_records(
                    conn,
                    member_id="deputat_1",
                    kind="laws",
                    records=original,
                )
                insert_only = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="laws",
                    records=refreshed,
                )
                self.assertEqual(insert_only.stored, 0)
                self.assertEqual(
                    conn.execute("SELECT title FROM dep_act_laws").fetchone()[0],
                    "Original title",
                )

                refreshed_result = store_records(
                    conn,
                    member_id="deputat_1",
                    kind="laws",
                    records=refreshed,
                    update_existing=True,
                )
                self.assertEqual(refreshed_result.stored, 1)
                self.assertEqual(
                    conn.execute("SELECT title FROM dep_act_laws").fetchone()[0],
                    "Refreshed title",
                )

    def test_member_activity_crawl_updates_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                first_activity = {
                    "legislative_proposals": type(
                        "Activity",
                        (),
                        {
                            "url": "https://example.test/old",
                            "count": 1,
                            "promulgated_count": 0,
                        },
                    )()
                }
                second_activity = {
                    "legislative_proposals": type(
                        "Activity",
                        (),
                        {
                            "url": "https://example.test/new",
                            "count": 2,
                            "promulgated_count": 1,
                        },
                    )()
                }
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity=first_activity,
                    results={},
                )
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity=second_activity,
                    results={},
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT legislative_proposals_url FROM dep_act_member_activity_crawl"
                    ).fetchone()[0],
                    "https://example.test/old",
                )
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity=second_activity,
                    results={},
                    update_existing=True,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT legislative_proposals_url FROM dep_act_member_activity_crawl"
                    ).fetchone()[0],
                    "https://example.test/new",
                )

    def test_member_activity_error_preserves_existing_activity_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                update_member_activity_crawl(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    activity={
                        "political_declarations": type(
                            "Activity",
                            (),
                            {
                                "url": "https://example.test/declarations",
                                "count": 4,
                                "promulgated_count": None,
                            },
                        )()
                    },
                    results={
                        "political_declarations": type(
                            "Result",
                            (),
                            {"stored": 4},
                        )()
                    },
                )
                update_member_activity_error(
                    conn,
                    member_id="deputat_1",
                    profile_url="https://example.test/profile",
                    error="network failed",
                )

                self.assertEqual(
                    conn.execute(
                        """
                        SELECT political_declarations_url,
                               political_declarations_count,
                               political_declarations_stored,
                               last_error
                        FROM dep_act_member_activity_crawl
                        """
                    ).fetchone(),
                    (
                        "https://example.test/declarations",
                        4,
                        4,
                        "network failed",
                    ),
                )

    # ---------------------------------------------------------------------
    # Fix 1: no anchor -> empty snippet -> no false-positive body matches.
    # ---------------------------------------------------------------------
    def test_extract_initiators_returns_empty_when_no_anchor_found(self):
        """
        Body text from Expunerea de motive must never be fed to the matcher.
        Before the fix, 4000 chars of body were returned and ~10 deputies
        whose name tokens happened to appear in the prose got falsely
        attributed as initiators (bug report: law:b678_2025 matched 11).
        """
        text = (
            "Prezenta propunere legislativă reglementează modul în care "
            "autoritățile publice cooperează cu societatea civilă. "
            "Domnul Ion Popescu a participat la dezbaterea publică. "
            "Doamna Maria Ionescu a semnat avizul consultativ."
        )
        section, source = extract_initiators_section_with_source(text)
        self.assertEqual(section, "")
        self.assertEqual(source, "none")
        self.assertEqual(
            match_initiator_member_ids(
                section,
                [
                    ("d1", "Ion Popescu", "ion popescu"),
                    ("d2", "Maria Ionescu", "maria ionescu"),
                ],
            ),
            [],
        )

    # ---------------------------------------------------------------------
    # Fix 4 + provenance: 'Inițiator,\nDeputat NAME' signature anchor.
    # Mirrors the b678_2025 shape: a single-initiator signature at the
    # very bottom of the Expunerea de motive PDF, no initiator list.
    # ---------------------------------------------------------------------
    def test_extract_initiators_signature_deputat_anchor(self):
        text = (
            "Prezentul proiect de lege reglementează ... (body copy) ... "
            "București, 12 octombrie 2025\n"
            "Inițiator,\n"
            "Deputat Silviu-Octavian Gurlui"
        )
        section, source = extract_initiators_section_with_source(text)
        self.assertEqual(source, "em_signature_initiator_deputat")
        self.assertIn("Silviu-Octavian Gurlui", section)
        matched = match_initiator_member_ids(
            section,
            [
                ("d_gurlui", "Silviu-Octavian Gurlui", "silviu octavian gurlui"),
                ("d_other", "Ion Popescu", "ion popescu"),
            ],
        )
        self.assertEqual(matched, ["d_gurlui"])

    # ---------------------------------------------------------------------
    # Fix 4: 'Deputat NAME\nCircumscripția electorală' signature anchor.
    # ---------------------------------------------------------------------
    def test_extract_initiators_signature_circumscriptie_anchor(self):
        text = (
            "... cuprinsul expunerii de motive continuă aici ...\n"
            "Deputat Raisa Enachi\n"
            "Circumscripția electorală nr. 39 Vaslui"
        )
        section, source = extract_initiators_section_with_source(text)
        self.assertEqual(source, "em_signature_deputat_circumscriptie")
        self.assertIn("Raisa Enachi", section)
        matched = match_initiator_member_ids(
            section,
            [("d_raisa", "Raisa Enachi", "raisa enachi")],
        )
        self.assertEqual(matched, ["d_raisa"])

    # ---------------------------------------------------------------------
    # Fix 5: truncate prose tail at body-copy / table-header stop words.
    # ---------------------------------------------------------------------
    def test_extract_initiators_prose_truncates_at_stop_words(self):
        """
        OCR may glue the signature line to a next-page table header, e.g.
        'Initiator: Cezar-Mihail Drăgoescu Lista iniSatornor la propunerea
        legislativa'. The snippet must stop at 'Lista' so downstream shape
        checks see only the signature fragment.
        """
        text = (
            "Initiator: Cezar-Mihail Drăgoescu Lista iniSatornor la "
            "propunerea legislativa pentru modificarea art. 1"
        )
        section, source = extract_initiators_section_with_source(text)
        self.assertIn("Cezar-Mihail Drăgoescu", section)
        self.assertNotIn("propunerea", section.lower())
        self.assertEqual(source, "em_initiatori_colon")

    # ---------------------------------------------------------------------
    # Fix 6: single-signature fuzzy fallback - OCR mangles given name.
    # ---------------------------------------------------------------------
    def test_single_signature_fuzzy_match_ocr_garbled_given_name(self):
        """
        Real case (law:b217_2026): OCR produced 'Cezar-rmail' instead of
        'Cezar-Mihail'. Given the member roster has exactly one deputy
        with a fuzzy match of this shape (Cezar-Mihail Drăgoescu),
        we accept it as the initiator.
        """
        window = "Initiator: Cezar-rmail"
        member_rows = [
            ("d_dragoescu", "Cezar-Mihail Drăgoescu", "cezar mihail dragoescu"),
            ("d_other_a", "Ion Popescu", "ion popescu"),
            ("d_other_b", "Maria Ionescu", "maria ionescu"),
        ]
        self.assertEqual(
            single_signature_fuzzy_match(window, member_rows),
            ["d_dragoescu"],
        )

    def test_single_signature_fuzzy_match_rejects_ambiguous(self):
        """If two members fuzzy-match equally, reject rather than guess."""
        window = "Initiator: Ion Po"
        member_rows = [
            ("d_1", "Ion Popescu", "ion popescu"),
            ("d_2", "Ion Popovici", "ion popovici"),
        ]
        self.assertEqual(single_signature_fuzzy_match(window, member_rows), [])

    def test_single_signature_fuzzy_match_rejects_empty_window(self):
        self.assertEqual(
            single_signature_fuzzy_match(
                "",
                [("d1", "Ion Popescu", "ion popescu")],
            ),
            [],
        )

    # ---------------------------------------------------------------------
    # Senat: prefer "adresa de înaintare ... pentru dezbatere" (AD.PDF).
    # The AD PDF is a short cover letter from the submitting chamber's
    # Birou permanent that lists **only** the formal initiator(s); unlike
    # the EM PDF it never embeds a susținători table, which is why the
    # matcher previously picked up co-signers as initiators.
    # ---------------------------------------------------------------------
    def test_senat_prefers_adresa_inaintare_over_em_and_forma(self):
        """
        When both AD and EM are present on the fișă, AD must come first
        (it is the clean single-purpose cover letter). EM is kept in the
        list as a secondary candidate because some AD letters don't name
        the initiator; the per-PDF extraction loop stops on the first
        PDF that yields at least one matched member. Forma inițiatorului
        is only used when neither AD nor EM is exposed.
        """
        html = """
        <table>
          <tr>
            <td>
              <a href='PDF\\2026\\26b134AD.pdf?nocache=true'>
                 - adresa de înaintare a iniţiativei legislative pentru dezbatere
              </a><br/>
              <a href='PDF\\2026\\26b134FG.pdf?nocache=true'>forma inițiatorului</a><br/>
              <a href='PDF\\2026\\26b134EM.pdf?nocache=true'>expunerea de motive</a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b134&an_cls=2026"
        ad = parse_senat_adresa_inaintare_pdf_url(html, base)
        self.assertEqual(
            ad,
            "https://senat.ro/legis/PDF/2026/26b134AD.pdf?nocache=true",
        )
        urls = list_senat_law_initiator_pdf_urls(html, base)
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], ad)
        self.assertIn("em.pdf", urls[1].lower())
        self.assertTrue(all("fg.pdf" not in u.lower() for u in urls))

    def test_senat_adresa_inaintare_absolute_href_with_forward_slashes(self):
        html = """
        <table>
          <tr>
            <td>
              <a href="/legis/PDF/2026/26b134AD.PDF?nocache=true" target="_blank">
                <img alt='adresa de înaintare a iniţiativei legislative pentru dezbatere'/>
                adresa de înaintare a iniţiativei legislative pentru dezbatere
              </a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b134&an_cls=2026"
        ad = parse_senat_adresa_inaintare_pdf_url(html, base)
        self.assertEqual(
            ad,
            "https://senat.ro/legis/PDF/2026/26b134AD.PDF?nocache=true",
        )

    def test_senat_adresa_inaintare_ignores_generic_adresa_row(self):
        """
        The law fișă has a summary row `<tr><td>Adresa:</td><td> - </td></tr>`
        (textual metadata, no PDF) and a separate `adresă de desesizare`
        (DE.PDF) link. Neither should be mistaken for the cover letter.
        """
        html = """
        <table>
          <tr><td>Adresa:</td><td> - </td></tr>
          <tr>
            <td>
              <a href='PDF\\2026\\26b134DE.pdf?nocache=true'>
                 - adresa de desesizare a Camerei care nu are competenţă de primă Cameră sesizată
              </a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b134&an_cls=2026"
        self.assertIsNone(parse_senat_adresa_inaintare_pdf_url(html, base))

    def test_senat_falls_back_to_em_when_adresa_missing(self):
        html = """
        <table>
          <tr>
            <td>
              <a href="/legis/PDF/2026/26b131EM.PDF">expunerea de motive</a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b131&an_cls=2026"
        self.assertIsNone(parse_senat_adresa_inaintare_pdf_url(html, base))
        urls = list_senat_law_initiator_pdf_urls(html, base)
        self.assertEqual(len(urls), 1)
        self.assertIn("EM.PDF", urls[0])

    # ---------------------------------------------------------------------
    # Fix 7: senat fallback to 'Forma inițiatorului' PDF.
    # ---------------------------------------------------------------------
    def test_senat_falls_back_to_forma_initiatorului_when_em_missing(self):
        """
        Senat stage-B initiatives often expose only 'forma inițiatorului'
        (FG.PDF / FI.PDF) and no 'expunerea de motive' (EM.PDF). Tail-OCR
        of that PDF still recovers the initiator signature at the bottom.
        """
        html = """
        <table>
          <tr>
            <td>
              <a href="/legis/PDF/2025/25b626FG.PDF?nocache=true">
                forma inițiatorului
              </a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b626&an_cls=2025"
        forma = parse_senat_forma_initiatorului_pdf_url(html, base)
        self.assertEqual(
            forma,
            "https://senat.ro/legis/PDF/2025/25b626FG.PDF?nocache=true",
        )
        self.assertEqual(
            list_senat_law_initiator_pdf_urls(html, base),
            [forma],
        )

    def test_senat_does_not_fallback_when_em_present(self):
        """When both EM.PDF and Forma inițiatorului are present, prefer EM only."""
        html = """
        <table>
          <tr>
            <td>
              <a href="/legis/PDF/2026/26b131FG.PDF">forma inițiatorului</a>
              <a href="/legis/PDF/2026/26b131EM.PDF">expunerea de motive</a>
            </td>
          </tr>
        </table>
        """
        base = "https://senat.ro/legis/lista.aspx?nr_cls=b131&an_cls=2026"
        urls = list_senat_law_initiator_pdf_urls(html, base)
        self.assertEqual(len(urls), 1)
        self.assertIn("EM.PDF", urls[0])

    # ---------------------------------------------------------------------
    # Fix 8: CDEP URL normalization + transient error classification.
    # ---------------------------------------------------------------------
    def test_normalize_law_source_url_injects_cam2_for_cdep(self):
        self.assertEqual(
            _normalize_law_source_url(
                "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?idp=22204"
            ),
            "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?cam=2&idp=22204",
        )

    def test_normalize_law_source_url_preserves_existing_cam(self):
        original = (
            "https://www.cdep.ro/pls/proiecte/upl_pck2015.proiect?cam=1&idp=22204"
        )
        self.assertEqual(_normalize_law_source_url(original), original)

    def test_normalize_law_source_url_ignores_non_cdep_and_senat(self):
        senat = "https://senat.ro/legis/lista.aspx?nr_cls=b131&an_cls=2026"
        self.assertEqual(_normalize_law_source_url(senat), senat)
        unrelated = "https://example.test/profile"
        self.assertEqual(_normalize_law_source_url(unrelated), unrelated)

    def test_match_initiator_requires_surname_on_matching_line(self):
        """
        When two members share the exact compound given name (`Remus-
        Gabriel`), only the one whose surname actually appears on the
        initiator line should match. Regression for `law:b138_2026`:
        snippet line `Lapusan Remus-Gabriel` must match `Lăpuşan Remus-
        Gabriel` (surname `lapusan` present) but NOT `Mihalcea Remus-
        Gabriel` (surname absent) even though 2/3 sub-tokens are shared.
        """
        text = (
            "INITIATORI\n"
            "Govor Mircea-Vasile\n"
            "Lapusan Remus-Gabriel\n"
            "Samoila Ion\n"
        )
        rows = [
            ("d_govor", "Govor Mircea-Vasile", "govor mircea-vasile"),
            ("d_lapusan", "Lăpuşan Remus-Gabriel", "lapusan remus-gabriel"),
            ("d_samoila", "Samoilă Ion", "samoila ion"),
            ("d_mihalcea", "Mihalcea Remus-Gabriel", "mihalcea remus-gabriel"),
        ]
        matched = match_initiator_member_ids(text, rows)
        self.assertIn("d_govor", matched)
        self.assertIn("d_lapusan", matched)
        self.assertIn("d_samoila", matched)
        self.assertNotIn("d_mihalcea", matched)

    def test_match_initiator_recovers_half_of_hyphenated_given_name(self):
        """
        Regression for `law:b143_2026` and `law:b152_2026`: OCR extracts
        only one half of a compound given name (`Remus Negoi` for member
        `NEGOI Eugen-Remus`; `Lapusan Remus` for `Lăpuşan Remus-Gabriel`).
        Splitting the member's hyphenated name into sub-tokens lets the
        matcher accept 2-of-3 subtoken matches while still rejecting the
        shared-given-name false positives.
        """
        text = "Initiatori:\nRemus Negoi\nLapusan Remus\n"
        rows = [
            ("s_negoi", "NEGOI Eugen-Remus", "negoi eugen-remus"),
            ("d_lapusan", "Lăpuşan Remus-Gabriel", "lapusan remus-gabriel"),
            ("d_other", "Stoian Remus", "stoian remus"),
        ]
        matched = match_initiator_member_ids(text, rows)
        self.assertIn("s_negoi", matched)
        self.assertIn("d_lapusan", matched)
        self.assertNotIn("d_other", matched)

    def test_match_initiator_rejects_cross_line_subtoken_combination(self):
        """
        In a multi-initiator list, sub-tokens of a candidate member name
        must NOT combine across different lines (= different people) to
        form a spurious match. Regression for `law:b131_2026` where the
        snippet lists five distinct initiators; without per-line
        matching, members like `Puşcaşu Lucian-Florin` (sub-tokens
        puscasu/lucian/florin) would falsely match because `lucian`
        appears in line `Doru Lucian Musat` and `florin` appears in
        line `Florin-Cornel Popovici` — both in the blob but never on
        the same line.
        """
        text = (
            "Initiatori:\n"
            "Florin-Cornel Popovici\n"
            "Doru Lucian Musat\n"
            "Petrisor-Gabriel Peiu\n"
            "Veronica Grosu\n"
            "Andra-Claudia Constantinescu\n"
        )
        rows = [
            ("d1", "Popovici Florin-Cornel", "popovici florin-cornel"),
            ("d2", "Muşat Doru-Lucian", "musat doru-lucian"),
            ("d3", "Grosu Veronica", "grosu veronica"),
            ("d4", "Constantinescu Andra-Claudia", "constantinescu andra-claudia"),
            ("s1", "PEIU Petrișor-Gabriel", "peiu petrisor-gabriel"),
            ("fp1", "Puşcaşu Lucian-Florin", "puscasu lucian-florin"),
            ("fp2", "Bogdănel Lucian-Gabriel", "bogdanel lucian-gabriel"),
        ]
        matched = match_initiator_member_ids(text, rows)
        self.assertEqual(
            sorted(matched),
            sorted(["d1", "d2", "d3", "d4", "s1"]),
        )
        self.assertNotIn("fp1", matched)
        self.assertNotIn("fp2", matched)

    def test_match_initiator_rejects_shared_given_name_only(self):
        """
        Regression for law:b134_2026: snippet contains `deputat Rodica Plopeanu`
        so the correct member (`Plopeanu Rodica`) must match, but members who
        share only the given name `Rodica` (Nassar Rodica, Cușnir Rodica,
        Constantinescu Rodica) must NOT be promoted. The old fuzzy-line rule
        accepted `hits >= len(parts) - 1` which falsely promoted every `X
        Rodica` 2-token name once "Rodica" was present in the blob.
        """
        initiators_text = (
            "Initiator:\n"
            "[Inițiatori identificați în textul dinaintea tabelului susținători "
            "(posibil OCR manuscris):]\n"
            "deputat Rodica Plopeanu\n"
            "\\chut^' urouIA `giv\n"
            "ft) fry\n"
        )
        member_rows = [
            ("d_plopeanu", "Plopeanu Rodica", "plopeanu rodica"),
            ("d_nassar", "Nassar Rodica", "nassar rodica"),
            ("s_cusnir", "CUȘNIR Rodica", "cusnir rodica"),
            ("s_constantinescu", "CONSTANTINESCU Rodica", "constantinescu rodica"),
        ]
        self.assertEqual(
            match_initiator_member_ids(initiators_text, member_rows),
            ["d_plopeanu"],
        )

    def test_hydrate_law_initiators_ad_without_names_falls_back_to_em(self):
        """When AD.PDF contains no usable initiator snippet, keep the flow
        purely local and continue to the next candidate PDF (EM.PDF)."""
        law_page_html = (
            "<html><body><table>"
            "<tr><td>adresa de înaintare a iniţiativei legislative pentru dezbatere</td>"
            "<td><a href=\"/legis/PDF/2026/26b135AD.PDF\">AD</a></td></tr>"
            "<tr><td>Expunerea de motive</td>"
            "<td><a href=\"/legis/PDF/2026/26b135EM.PDF\">EM</a></td></tr>"
            "</table></body></html>"
        )
        fetcher = MagicMock()
        fetcher.fetch.return_value = law_page_html
        fetcher.fetch_bytes.return_value = b"%PDF-1.4 dummy"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                ensure_activity_schema(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title,
                        details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:b200_2026",
                        "https://senat.ro/legis/lista.aspx?nr_cls=b200&an_cls=2026",
                        "L 200/2026",
                        "T",
                        "D",
                        "[]",
                    ),
                )
                conn.commit()

            with patch(
                "crawl_deputy_activity.extract_pdf_text_local",
                return_value="Către,\nBiroul Permanent al Senatului",
            ):
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    hydrate_law_initiators_for_records(
                        conn,
                        records=[
                            ListingRecord(
                                record_id="law:b200_2026",
                                source_url="https://senat.ro/legis/lista.aspx?nr_cls=b200&an_cls=2026",
                                identifier="L 200/2026",
                                adopted_law_identifier=None,
                                title="T",
                                details_text="D",
                                columns=[],
                            )
                        ],
                        fetcher=fetcher,
                        member_rows=[],
                        force=True,
                        skip_if_initiator_linked=True,
                        senator_member_rows=[],
                        law_initiator_pdf_dir=cache_dir,
                    )
                    conn.commit()

            # Both AD and EM must have been tried since AD produced no
            # snippet and hydration now relies only on local OCR/parsing.
            fetched_pdf_urls = [
                call.args[0] for call in fetcher.fetch_bytes.call_args_list
            ]
            self.assertEqual(len(fetched_pdf_urls), 2)
            self.assertIn("AD", fetched_pdf_urls[0])
            self.assertIn("EM", fetched_pdf_urls[1])

    def test_hydrate_law_initiators_uses_senat_search_when_source_url_is_search_page(self):
        search_page_html = """
        <html><body>
          <form method="post" action="./lista.aspx?nr_cls=b135&amp;an_cls=2026" id="aspnetForm">
            <input type="hidden" name="__VIEWSTATE" value="vs123" />
            <input name="ctl00$B_Center$Lista$txtNri" />
            <select name="ctl00$B_Center$Lista$ddAni"><option selected value="2026">2026</option></select>
            <h4>Rezultate Cautare</h4>
            <table><tr><td>Nu au fost găsite înregistrări.</td></tr></table>
          </form>
        </body></html>
        """
        search_results_html = """
        <html><body><table>
          <tr><td>adresa de înaintare a iniţiativei legislative pentru dezbatere</td>
              <td><a href="/legis/PDF/2026/26L257AD.PDF?nocache=true">AD</a></td></tr>
          <tr><td>Expunerea de motive</td>
              <td><a href="/legis/PDF/2026/26L257EM.PDF?nocache=true">EM</a></td></tr>
        </table></body></html>
        """

        fetcher = MagicMock()
        fetcher.fetch.return_value = search_page_html
        fetcher.fetch_bytes.return_value = b"%PDF-1.4 dummy"
        fetcher.retries = 1
        fetcher.sleep_seconds = 0
        fetcher.timeout = 30
        fetcher.user_agent = "votez-test-agent"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = init_db(tmp_path / "state.sqlite")
            cache_dir = tmp_path / "outputs" / "pdfs" / "law_initiators"
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                insert_test_member(conn)
                conn.execute(
                    """
                    INSERT INTO dep_act_laws (
                        law_id, source_url, identifier, title, details_text, columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "law:b135_2026",
                        "https://senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
                        "B135/2026",
                        "T",
                        "D",
                        "[]",
                    ),
                )
                conn.commit()

            with patch(
                "crawl_deputy_activity._fetch_senat_search_results_html",
                return_value=search_results_html,
            ) as mocked_search, patch(
                "crawl_deputy_activity.extract_pdf_text_local",
                return_value="Inițiatori:\nDeputat Test\nExpunerea de motive",
            ):
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    hydrate_law_initiators_for_records(
                        conn,
                        records=[
                            ListingRecord(
                                record_id="law:b135_2026",
                                source_url="https://senat.ro/legis/lista.aspx?nr_cls=b135&an_cls=2026",
                                identifier="B135/2026",
                                adopted_law_identifier=None,
                                title="T",
                                details_text="D",
                                columns=[],
                            )
                        ],
                        fetcher=fetcher,
                        member_rows=[("deputat_1", "Deputat Test", "deputat test")],
                        force=True,
                        skip_if_initiator_linked=False,
                        senator_member_rows=[],
                        law_initiator_pdf_dir=cache_dir,
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT motive_pdf_url FROM dep_act_laws WHERE law_id = ?",
                        ("law:b135_2026",),
                    ).fetchone()
                    links = conn.execute(
                        "SELECT member_id, is_initiator FROM dep_act_member_laws WHERE law_id = ?",
                        ("law:b135_2026",),
                    ).fetchall()

            mocked_search.assert_called_once()
            call_args = mocked_search.call_args.kwargs
            self.assertEqual(call_args["search_number"], "b135")
            self.assertEqual(call_args["search_year"], "2026")
            self.assertIn("26L257AD", row[0] or "")
            self.assertEqual(links, [("deputat_1", 1)])
            fetcher.fetch.assert_called_once()
            fetcher.fetch_bytes.assert_called_once_with(
                "https://senat.ro/legis/PDF/2026/26L257AD.PDF?nocache=true"
            )

    def test_is_transient_network_exception_covers_refused_and_timeout(self):
        """Connection-refused / timeout must not poison initiators_parse_error."""
        self.assertTrue(
            _is_transient_network_exception(
                urllib.error.URLError("Connection refused")
            )
        )
        self.assertTrue(_is_transient_network_exception(TimeoutError("slow")))
        self.assertTrue(
            _is_transient_network_exception(
                urllib.error.URLError("timed out")
            )
        )
        self.assertFalse(
            _is_transient_network_exception(ValueError("bad parse"))
        )

    # ------------------------------------------------------------------
    # Hydration circuit-breaker + throttle tests. These exist specifically
    # to guard against the "cdep.ro blocked us and we hammered through
    # 200+ laws" scenario observed in April 2026.
    # ------------------------------------------------------------------

    def _seed_hydration_laws(self, conn: sqlite3.Connection, n: int) -> list[ListingRecord]:
        insert_test_member(conn)
        for i in range(n):
            law_id = f"law:hydrate:{i}"
            source_url = f"https://cdep.test/law/{i}"
            conn.execute(
                """
                INSERT INTO dep_act_laws (
                    law_id, source_url, identifier, title, details_text, columns_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (law_id, source_url, f"PL {i}/2026", f"Title {i}", "Details", "[]"),
            )
        conn.commit()
        return [
            ListingRecord(
                record_id=f"law:hydrate:{i}",
                source_url=f"https://cdep.test/law/{i}",
                identifier=None,
                adopted_law_identifier=None,
                title=f"Title {i}",
                details_text="Details",
                columns=[],
            )
            for i in range(n)
        ]

    def test_hydrate_circuit_breaker_aborts_after_consecutive_failures(self):
        """After N consecutive transient failures, hydration stops early
        so we don't hammer a blocked server. Remaining laws are left for
        the next run (not marked as parse errors)."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                records = self._seed_hydration_laws(conn, 10)
            fetcher = MagicMock()
            fetcher.fetch.side_effect = urllib.error.URLError("Connection refused")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                hydrate_law_initiators_for_records(
                    conn,
                    records=records,
                    fetcher=fetcher,
                    member_rows=[("deputat_1", "Test", "test")],
                    force=True,
                    skip_if_initiator_linked=False,
                    consecutive_failure_abort_after=3,
                    consecutive_failure_pause_after=0,
                    per_fetch_sleep_seconds=0,
                    sleep_fn=lambda _s: None,
                )
                unhydrated = conn.execute(
                    """
                    SELECT COUNT(*) FROM dep_act_laws
                    WHERE initiators_text IS NULL
                      AND initiators_parse_error IS NULL
                    """
                ).fetchone()[0]
            # 3 consecutive failures should halt the loop, leaving the
            # remaining 7 laws untouched (not persisted as parse errors).
            self.assertEqual(fetcher.fetch.call_count, 3)
            self.assertEqual(unhydrated, 10)

    def test_hydrate_circuit_breaker_pauses_without_aborting(self):
        """When the pause threshold is hit but abort is disabled, the
        loop sleeps for `pause_seconds` and continues."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                records = self._seed_hydration_laws(conn, 6)
            fetcher = MagicMock()
            fetcher.fetch.side_effect = urllib.error.URLError("Connection refused")
            sleep_calls: list[float] = []
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                hydrate_law_initiators_for_records(
                    conn,
                    records=records,
                    fetcher=fetcher,
                    member_rows=[("deputat_1", "Test", "test")],
                    force=True,
                    skip_if_initiator_linked=False,
                    consecutive_failure_abort_after=0,
                    consecutive_failure_pause_after=2,
                    consecutive_failure_pause_seconds=30.0,
                    per_fetch_sleep_seconds=0,
                    sleep_fn=sleep_calls.append,
                )
            # All 6 laws processed; circuit breaker paused at failures 2, 4, 6.
            self.assertEqual(fetcher.fetch.call_count, 6)
            self.assertEqual(
                [s for s in sleep_calls if s == 30.0],
                [30.0, 30.0, 30.0],
            )

    def test_hydrate_consecutive_failure_counter_resets_on_success(self):
        """A successful fetch (even one that later errors during parse)
        resets the circuit-breaker counter, so sporadic failures across
        a long run do not falsely trip the abort."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                records = self._seed_hydration_laws(conn, 5)
            transient = urllib.error.URLError("Connection refused")
            # Pattern: fail, fail, success-but-no-pdf, fail, fail.
            # With abort_after=3 this MUST NOT abort because the success
            # in the middle resets the counter.
            fetcher = MagicMock()
            fetcher.fetch.side_effect = [
                transient,
                transient,
                "<html>no pdf here</html>",
                transient,
                transient,
            ]
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                hydrate_law_initiators_for_records(
                    conn,
                    records=records,
                    fetcher=fetcher,
                    member_rows=[("deputat_1", "Test", "test")],
                    force=True,
                    skip_if_initiator_linked=False,
                    consecutive_failure_abort_after=3,
                    consecutive_failure_pause_after=0,
                    per_fetch_sleep_seconds=0,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(fetcher.fetch.call_count, 5)

    def test_hydrate_per_fetch_sleep_is_applied_between_laws(self):
        """The hydration loop throttles between law fetches — the exact
        sequence of "0.1s per fetch, no pause, no abort" must produce
        one sleep per processed-but-not-last law."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_db(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                records = self._seed_hydration_laws(conn, 4)
            fetcher = MagicMock()
            fetcher.fetch.return_value = "<html>no pdf here</html>"
            sleep_calls: list[float] = []
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                hydrate_law_initiators_for_records(
                    conn,
                    records=records,
                    fetcher=fetcher,
                    member_rows=[("deputat_1", "Test", "test")],
                    force=True,
                    skip_if_initiator_linked=False,
                    consecutive_failure_abort_after=0,
                    consecutive_failure_pause_after=0,
                    per_fetch_sleep_seconds=0.1,
                    sleep_fn=sleep_calls.append,
                )
            # 4 laws -> sleep between iterations 1-2, 2-3, 3-4. Last
            # iteration does not sleep (index < len(records) is False).
            self.assertEqual(fetcher.fetch.call_count, 4)
            self.assertEqual(sleep_calls, [0.1, 0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
