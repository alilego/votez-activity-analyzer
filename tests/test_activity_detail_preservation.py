import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import crawl_deputy_activity as crawler


def record(url):
    return crawler.ListingRecord(record_id="record", source_url=url, identifier=None,
                                 title="Title", details_text="Listing text", columns=[])


class DetailPreservationTests(unittest.TestCase):
    def test_membership_request_is_not_a_primary_initiator_signature(self):
        text='Deputat Ion Popescu Circumscripția electorală nr. 1, vă rog să aprobați solicitarea de retragere a semnăturii de co-inițiator.'
        snippet,_=crawler.extract_initiators_section_with_source(text)
        self.assertEqual(snippet,'')

    def test_long_ocr_document_includes_head_and_tail_within_page_budget(self):
        import pypdfium2
        document=mock.MagicMock();document.__len__.return_value=16
        page=mock.Mock();indices=[]
        def get_page(index):
            indices.append(index);return page
        document.__getitem__.side_effect=get_page
        with mock.patch.object(pypdfium2,'PdfDocument',return_value=document), \
                mock.patch.object(crawler,'_run_tesseract',return_value='text'):
            crawler._extract_pdf_text_ocr(b'%PDF',max_pages=10,from_tail=True,include_head=True)
        self.assertEqual(indices,list(range(5))+list(range(11,16)))

    def test_initiator_matching_rejects_short_name_and_substring_collisions(self):
        roster=[('dan','Tanasă Dan','tanasa dan'),('stefan','Tanasă Ştefan','tanasa stefan'),
            ('ion','Ion Nicolae-Mirel','ion nicolae-mirel'),('simion','Simion George-Nicolae','simion george-nicolae'),
            ('nicolae','Nicolae Dan-Mihai','nicolae dan-mihai'),('mihaiu','Mihaiu Radu-Nicolae','mihaiu radu-nicolae')]
        for text,expected in [('INIȚIATORI: TANASA STEFAN',['stefan']),
                ('INIȚIATOR: George-Nicolae Simion',['simion']),
                ('INIȚIATOR: MIHAIU Radu-Nicolae',['mihaiu'])]:
            self.assertEqual(crawler.match_initiator_member_ids(text,roster),expected)

    def test_initiator_block_stops_before_next_proposal_table(self):
        text='Inițiatori. Deputat Marian CRUSOVEANU Propunere legislativă pentru modificarea legii Nr crt Nume Calin Petru'
        snippet,_=crawler.extract_initiators_section_with_source(text)
        self.assertNotIn('Calin Petru',snippet)
        roster=[('deputy','Cruşoveanu Marian','crusoveanu marian'),('senator','MARIAN Călin-Petru','marian calin-petru')]
        self.assertEqual(crawler.match_initiator_member_ids(snippet,roster),['deputy'])

    def test_single_signature_one_letter_error_requires_full_unique_name(self):
        window = 'Inițiator: Deputat Maria GERNIT PARTIDUL OAMENILOR TINERI'
        roster = [('cernit','Cernit Maria','cernit maria'),('other','Horga Maria-Gabriela','horga maria-gabriela'),
                  ('marian','Marian Călin-Petru','marian calin-petru')]
        self.assertEqual(crawler.single_signature_fuzzy_match(window,roster),['cernit'])
        self.assertEqual(crawler.single_signature_fuzzy_match(window,roster+[('ambiguous','Dernit Maria','dernit maria')]),[])
        self.assertEqual(crawler.single_signature_fuzzy_match('Inițiator: Deputat Maria',roster),[])

    def test_written_title_mentioning_chamber_is_not_navigation(self):
        page = '''<table><tr><td></td><td><a href="steno2024.stenograma_scris?ids=123&idm=1.016">Text</a></td>
        <td>Florin Roman - declaraţie politică: "Camera Deputaţilor a adoptat propunerea"; în scris</td></tr></table>'''
        records = crawler.parse_listing_records(page, 'https://www.cdep.ro/ords/pls/steno/steno2024.lista', 'political_declarations')
        self.assertEqual([r.record_id for r in records], ['political_declaration:cdep:123:1_016'])

    def test_numbered_question_mentioning_chamber_is_not_navigation(self):
        page = '''<table><tr><td>19.</td><td>Interpelarea nr.817B/25.11.2025</td>
        <td><a href="/ords/pls/parlam/interpelari2015.detalii?idi=81507&idl=1">Clarificări la interpelarea înregistrată la Camera Deputaților</a></td></tr></table>'''
        records = crawler.parse_listing_records(page, 'https://www.cdep.ro/profile', 'questions')
        self.assertEqual([r.record_id for r in records], ['question:cdep:81507'])

    def test_question_recipient_survives_registry_member_id_shift(self):
        with sqlite3.connect(':memory:') as conn:
            conn.execute('CREATE TABLE dep_act_questions_interpellations (question_id TEXT,source_url TEXT,recipient TEXT,member_id TEXT)')
            conn.execute('INSERT INTO dep_act_questions_interpellations VALUES (?,?,?,?)',
                ('record','https://www.cdep.ro/question/1','Ministerul Finanțelor','old_member_id'))
            cache = crawler._existing_question_recipients(conn)
        fetcher = mock.Mock()
        records = crawler.hydrate_question_records(listing_records=[record('https://www.cdep.ro/new-url')],
            fetcher=fetcher, existing_recipients=cache)
        self.assertEqual(records[0].recipient, 'Ministerul Finanțelor')
        fetcher.fetch.assert_not_called()

    def test_published_listing_total_detects_missing_records(self):
        base = 'https://www.cdep.ro/ords/pls/steno/steno2024.lista?idv=1&leg=2024'
        item = '<table><tr><td><a href="steno2024.stenograma_scris?ids=152&idm=1.029">Declaraţie politică despre Praid</a></td></tr></table>'
        count = '<table><tr><td>înregistrări găsite:</td><td>2</td></tr></table>'
        fetcher = mock.Mock(sleep_seconds=0)
        fetcher.fetch.return_value = count + item
        with self.assertRaisesRegex(ValueError, '1 of 2 published results'):
            crawler.load_political_declaration_listing(count + item, base, fetcher)
        fetcher.fetch.assert_called_once_with(base)
        self.assertEqual(len(crawler.load_political_declaration_listing(count.replace('>2<','>1<') + item, base, mock.Mock(sleep_seconds=0))), 1)
        fetcher.reset_mock()
        fetcher.fetch.return_value = count + item + item.replace('ids=152','ids=143')
        self.assertEqual(len(crawler.load_political_declaration_listing(count + item, base, fetcher)), 2)
        fetcher.fetch.assert_called_once_with(base)

    def test_numbered_law_at_chamber_is_not_filtered_as_navigation(self):
        page = '''<table><tr><td>60.</td><td>BP824/03.11.2026 B491/2026</td>
        <td><a href="/ords/pls/proiecte/upl_pck2015.proiect?cam=2&idp=23633">Propunere legislativă pentru învăţământ</a></td>
        <td>la Camera Deputatilor</td></tr></table>'''
        records = crawler.parse_listing_records(page, "https://www.cdep.ro/profile", "laws")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "law:cdep:23633")

    def test_written_pagination_keeps_series_and_deduplicates_records(self):
        base = "https://www.cdep.ro/ords/pls/steno/steno2024.lista?idv=5396&leg=2024&idl=1"
        item = '<table><tr><td><a href="steno2024.stenograma_scris?ids=152&idm=1.029">Declaraţie politică despre Praid</a></td></tr></table>'
        next_link = '<a href="steno2015.lista?idv=5396&leg=2024&pags=21">2</a>'
        unrelated = '<a href="steno2015.lista?idv=OTHER&leg=2024&pags=41">3</a>'
        back_link = '<a href="steno2015.lista?idv=5396&leg=2024&pags=1">1</a>'
        fetcher = mock.Mock(sleep_seconds=0)
        fetcher.fetch.return_value = item + item.replace("ids=152", "ids=143") + next_link + back_link
        records = crawler.load_political_declaration_listing(item + next_link + unrelated, base, fetcher)
        self.assertEqual(len(records), 2)
        fetcher.fetch.assert_called_once_with(
            "https://www.cdep.ro/ords/pls/steno/steno2024.lista?idv=5396&leg=2024&pags=21")

    def test_written_listing_accepts_current_dot_separator_and_legacy_comma(self):
        for separator in (".", ","):
            url = f"https://www.cdep.ro/ords/pls/steno/steno2024.stenograma_scris?ids=152&idm=1{separator}029&idl=1"
            page = f'<table><tr><td><a href="{url}">Declaraţie politică despre Praid</a></td></tr></table>'
            records = crawler.parse_listing_records(page, url, "political_declarations")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].record_id, "political_declaration:cdep:152:1_029")
        self.assertFalse(crawler._is_political_declaration_detail_url(
            "https://www.cdep.ro/steno2024.stenograma_scris?ids=152&idm=1&idl=1"))

    def test_existing_recipient_is_preserved_without_fetching(self):
        url = "https://www.cdep.ro/question/1"
        fetcher = mock.Mock()
        result = crawler.hydrate_question_records(
            listing_records=[record(url)], fetcher=fetcher,
            existing_recipients={url: "Ministerul Educaţiei"})
        self.assertEqual(result[0].recipient, "Ministerul Educaţiei")
        fetcher.fetch.assert_not_called()

    def test_stable_question_id_preserves_details_when_url_changes(self):
        fetcher = mock.Mock()
        result = crawler.hydrate_question_records(
            listing_records=[record("https://www.cdep.ro/ords/pls/new-url")], fetcher=fetcher,
            existing_recipients={"record": "Ministerul Educaţiei"})
        self.assertEqual(result[0].recipient, "Ministerul Educaţiei")
        fetcher.fetch.assert_not_called()

    def cache(self, folder, speeches):
        (folder / "stenograma_2025-06-04_2.json").write_text(json.dumps({
            "source_url": "https://www.cdep.ro/ords/pls/steno/steno2024.stenograma_scris?ids=56&idl=1",
            "speeches": speeches,
        }))
        return crawler._load_cached_written_declarations(folder)

    def test_exact_session_and_author_reuses_all_text_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache(Path(tmp), [{"speaker": "Domnul Ion Popescu", "text": "Prima parte.", "text2": "A doua parte."}])
            fetcher = mock.Mock()
            result = crawler.hydrate_political_declaration_records(
                member_id="deputat_1", member_name="Popescu Ion",
                listing_records=[record("https://www.cdep.ro/ords/pls/steno/steno2024.stenograma_scris?ids=56&idm=1.001&idl=1")],
                fetcher=fetcher, cached_stenograms=cache)
            self.assertEqual(result[0].full_text, "Prima parte. A doua parte.")
            fetcher.fetch.assert_not_called()

    def test_ambiguous_author_falls_back_to_official_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache(Path(tmp), [{"speaker": "Domnul Ion Popescu", "text": "First"},
                                           {"speaker": "Domnul Popescu Ion", "text": "Different"}])
            fetcher = mock.Mock()
            fetcher.fetch.return_value = "page"
            with mock.patch.object(crawler, "parse_political_declaration_detail", return_value=("url", "title", "Fetched text")):
                result = crawler.hydrate_political_declaration_records(
                    member_id="deputat_1", member_name="Ion Popescu",
                    listing_records=[record("https://www.cdep.ro/ords/pls/steno/steno2024.stenograma_scris?ids=56&idm=1.001&idl=1")],
                    fetcher=fetcher, cached_stenograms=cache)
            fetcher.fetch.assert_called_once()
            self.assertEqual(result[0].full_text, "Fetched text")

    def test_distinct_transcript_series_cannot_share_cached_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache(Path(tmp), [{"speaker": "Domnul Ion Popescu", "text": "Cached"}])
            fetcher = mock.Mock()
            with mock.patch.object(crawler, "parse_political_declaration_detail", return_value=("url", "title", "Fetched")):
                result = crawler.hydrate_political_declaration_records(
                    member_id="deputat_1", member_name="Ion Popescu",
                    listing_records=[record("https://www.cdep.ro/ords/pls/steno/steno2015.stenograma_scris?ids=56&idm=1.001&idl=1")],
                    fetcher=fetcher, cached_stenograms=cache)
            fetcher.fetch.assert_called_once()
            self.assertEqual(result[0].full_text, "Fetched")


if __name__ == "__main__":
    unittest.main()
