import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from init_db import init_db
from member_identity import IdentityIndex, member_id, source_member_id, guard_member_identity
from motion_signatures import parse_signatures, store_signatures
from repair_member_identities import repair
from export_activity import export_activity_snapshots
from validate_motion_exports import validate_motion_exports


def member(official_id, name, party):
    url = f'https://www.cdep.ro/ords/pls/parlam/structura2015.mp?leg=2024&cam=2&idm={official_id}'
    row = dict(name=name,profile_url=url,chamber='deputat',party=party,is_current=1)
    return dict(row, member_id=member_id(row,'deputat'))


def motion_page(signatories, total=None):
    groups = {}
    for person, party in signatories:
        groups.setdefault(party,[]).append(person)
    lists = ''.join('<b>Deputaţi '+party+'</b><ol>'+''.join(
        '<li><a href="'+p['profile_url']+'">'+p['name']+'</a></li>' for p in people)+'</ol>' for party,people in groups.items())
    return f'<html><body>Nr./Data înregistrării: 1 / 28.04.2026 Moţiunea a fost semnată de către {total if total is not None else len(signatories)} deputaţi {lists} Voturi pentru moţiune: 281</body></html>'


class IdentityRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_db(self.root/'state.sqlite')
        self.conn = sqlite3.connect(self.root/'state.sqlite')
        self.conn.row_factory = sqlite3.Row
        self.jianu = member(158,'Jianu Iosif-Florin','PNL')
        self.fifor = member(105,'Fifor Mihai-Viorel','PSD')
        self.fechet = member(103,'Fechet Mircea','PNL')
        self.canonical = {p['member_id']:p for p in (self.jianu,self.fifor,self.fechet)}
        for legacy,p in [('deputat_103',self.fechet),('deputat_158',self.jianu),('deputat_105',self.fifor)]:
            self.conn.execute('''INSERT INTO members(member_id,source_member_id,chamber,name,normalized_name,party_id,profile_url)
                VALUES(?,?,'deputat',?,?,?,?)''',(legacy,legacy.split('_')[-1],p['name'],p['name'].lower(),p['party'],p['profile_url']))
        self.conn.execute("INSERT INTO dep_act_motions(motion_id,source_url,title,details_text,columns_json) VALUES('motion:test','https://www.cdep.ro/motion','Example','','[]')")
        self.conn.execute("INSERT INTO dep_act_member_motions(member_id,member_normalized_name,motion_id) VALUES('deputat_103','fifor mihai-viorel','motion:test')")
        self.conn.execute("INSERT INTO dep_act_member_motions(member_id,member_normalized_name,motion_id) VALUES('deputat_105','fifor mihai-viorel','motion:test')")
        self.conn.execute("INSERT INTO dep_act_member_motions(member_id,member_normalized_name,motion_id) VALUES('deputat_158','jianu iosif-florin','motion:test')")
        self.conn.commit()
        self.pages = [('motion:test','https://www.cdep.ro/motion',motion_page([(self.fifor,'PSD'),(self.jianu,'PSD')]))]

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def run_repair(self):
        self.conn.execute('BEGIN IMMEDIATE')
        result = repair(self.conn,self.canonical,[],self.pages)
        self.conn.commit()
        return result

    def test_reordering_insertion_removal_never_changes_key(self):
        people = [self.fifor,self.fechet,self.jianu]
        expected = {p['name']:member_id(p,'deputat') for p in people}
        for order in (people,people[::-1],people[1:]):
            for position,p in enumerate(order,1):
                self.assertEqual(member_id(dict(p,id=str(position)),'deputat'),expected[p['name']])
        self.assertNotEqual(source_member_id(self.fifor,'deputat'),source_member_id(dict(self.fifor,profile_url=self.fifor['profile_url'].replace('leg=2024','leg=2020')),'deputat'))
        with self.assertRaises(ValueError): source_member_id(dict(id='103'),'deputat')

    def test_stale_identity_is_repaired_and_duplicates_preserved_in_audit(self):
        result = self.run_repair()
        rows = self.conn.execute('SELECT member_id FROM dep_act_member_motions').fetchall()
        self.assertEqual({r[0] for r in rows},{self.fifor['member_id'],self.jianu['member_id']})
        self.assertEqual(result['tables']['dep_act_member_motions']['duplicates_merged'],1)
        self.assertEqual(result['tables']['dep_act_member_motions']['stale_identity_repaired'],1)
        self.assertTrue(self.conn.execute("SELECT 1 FROM member_identity_audit WHERE original_json LIKE '%deputat_103%' AND resolved_member_id=?",(self.fifor['member_id'],)).fetchone())
        self.assertFalse(self.conn.execute('PRAGMA foreign_key_check').fetchall())

    def test_party_change_and_member_party_exports_agree(self):
        self.run_repair()
        export_activity_snapshots(self.conn,output_dir=self.root/'out/activity')
        pnl = json.loads((self.root/'out/activity/parties/activity_pnl.json').read_text())
        psd = json.loads((self.root/'out/activity/parties/activity_psd.json').read_text())
        self.assertEqual(pnl['motions_majority_supported'],[])
        self.assertEqual(psd['motions_majority_supported'][0]['party_supporters_count'],2)
        jianu = json.loads(next((self.root/'out/activity/members').glob('*'+self.jianu['member_id']+'*')).read_text())
        self.assertEqual(jianu['motions'][0]['signing_party_id'],'PSD')
        self.assertEqual(jianu['motions'][0]['co_supporting_parties'][0]['members_count'],1)
        self.assertEqual(jianu['motions'][0]['support_basis'],'official_signatures')
        self.assertEqual(validate_motion_exports(self.root/'out'),[])
        psd['motions_majority_supported'][0]['party_supporters_count'] = 14
        (self.root/'out/activity/parties/activity_psd.json').write_text(json.dumps(psd))
        self.assertTrue(validate_motion_exports(self.root/'out'))

    def seed_speech(self):
        self.conn.execute("INSERT INTO runs(run_id,started_at,status) VALUES('r',CURRENT_TIMESTAMP,'completed')")
        self.conn.execute("""INSERT INTO interventions_raw(intervention_id,run_id,session_id,stenogram_path,speech_index,
            raw_speaker,normalized_speaker,member_id,text,text_hash)
            VALUES('speech','r','s','input/example.json',0,'Domnul Mihai-Viorel Fifor','mihai-viorel fifor','deputat_103','Protected speech','hash')""")
        self.conn.execute("""INSERT INTO intervention_analysis(intervention_id,run_id,relevance_label,topics_json,
            evidence_chunk_ids_json,analysis_version,reasoning) VALUES('speech','r','constructive','[]','[]','llm:gpt-5-mini','Protected classification')""")
        self.conn.commit()

    def test_speech_repair_preserves_existing_classification_and_text(self):
        self.seed_speech()
        before = dict(self.conn.execute('SELECT * FROM intervention_analysis').fetchone())
        self.run_repair()
        self.assertEqual(dict(self.conn.execute('SELECT * FROM intervention_analysis').fetchone()),before)
        row = self.conn.execute('SELECT member_id,text FROM interventions_raw').fetchone()
        self.assertEqual(tuple(row),(self.fifor['member_id'],'Protected speech'))

    def test_protected_content_change_aborts_repair(self):
        self.seed_speech()
        self.conn.execute("CREATE TRIGGER corrupt_content AFTER UPDATE OF member_id ON interventions_raw BEGIN UPDATE interventions_raw SET text='corrupt' WHERE intervention_id=NEW.intervention_id; END")
        self.conn.commit()
        before = '\n'.join(self.conn.iterdump())
        self.conn.execute('BEGIN IMMEDIATE')
        with self.assertRaisesRegex(ValueError,'protected content'):
            repair(self.conn,self.canonical,[],self.pages)
        self.conn.rollback()
        self.assertEqual(before,'\n'.join(self.conn.iterdump()))

    def test_ambiguous_identity_is_archived_and_excluded(self):
        p1,p2 = member(301,'Popescu Ion','PSD'),member(302,'Popescu Ion','PNL')
        self.canonical.update({p['member_id']:p for p in (p1,p2)})
        self.conn.execute("INSERT INTO dep_act_member_motions VALUES('deputat_103','motion:unknown',CURRENT_TIMESTAMP,'popescu ion')")
        self.conn.commit()
        result = self.run_repair()
        self.assertEqual(result['tables']['dep_act_member_motions']['unresolved'],1)
        self.assertTrue(result['unresolved'])
        self.assertTrue(self.conn.execute("SELECT 1 FROM member_identity_audit WHERE reason='unresolved_identity'").fetchone())
        self.assertFalse(self.conn.execute("SELECT 1 FROM dep_act_member_motions WHERE motion_id='motion:unknown'").fetchone())

    def test_repair_is_idempotent(self):
        self.run_repair()
        before = '\n'.join(self.conn.iterdump())
        second = self.run_repair()
        self.assertEqual(before,'\n'.join(self.conn.iterdump()))
        self.assertFalse(second['tables']['dep_act_member_motions'].get('remapped',0))

    def test_failure_rolls_back_entire_repair(self):
        before = '\n'.join(self.conn.iterdump())
        self.conn.execute('BEGIN IMMEDIATE')
        with self.assertRaises(ValueError):
            repair(self.conn,self.canonical,[],[('motion:test','https://www.cdep.ro/motion','<html>Broken page</html>')])
        self.conn.rollback()
        self.assertEqual(before,'\n'.join(self.conn.iterdump()))

    def test_guard_prevents_incremental_positional_upsert(self):
        with self.assertRaisesRegex(ValueError,'legacy'):
            guard_member_identity(self.conn,list(self.canonical.values()))
        self.run_repair()
        guard_member_identity(self.conn,list(self.canonical.values()))

    def test_identity_resolution_preserves_given_name_order(self):
        index = IdentityIndex()
        index.add('Stoica Alin-Bogdan','one')
        index.add('Stoica Bogdan-Alin','two')
        self.assertEqual(index.resolve('Alin-Bogdan Stoica'),'one')
        self.assertEqual(index.resolve('Bogdan-Alin Stoica'),'two')
        index.add('Stoica Alin-Bogdan','three')
        self.assertIsNone(index.resolve('Alin-Bogdan Stoica'))

    def test_partial_source_rejected_and_vote_totals_do_not_add_signatures(self):
        page = motion_page([(self.jianu,'PSD')])
        self.assertEqual(len(parse_signatures(page,'https://www.cdep.ro')['signatures']),1)
        with self.assertRaises(ValueError): parse_signatures(motion_page([(self.jianu,'PSD')],total=2),'https://www.cdep.ro')

    def test_official_motion_1584_fixture_has_no_pnl_signatories(self):
        path = Path(__file__).parent/'fixtures/motion_1584_signatories.html'
        parsed = parse_signatures(path.read_text(),'https://www.cdep.ro')
        self.assertEqual(parsed['counts'],{'deputat':175,'senator':79})
        deputies = [s for s in parsed['signatures'] if s['chamber']=='deputat']
        self.assertFalse([s for s in deputies if s['signing_party_id']=='PNL'])
        jianu = next(s for s in deputies if s['profile_key']=='cdep_2024_2_158')
        self.assertEqual(jianu['signing_party_id'],'PSD')
        self.assertEqual(parsed['registered_on'],'2026-04-28')

    def test_unverified_profile_links_never_imply_party_support(self):
        export_activity_snapshots(self.conn,output_dir=self.root/'out/activity')
        for file in (self.root/'out/activity/parties').glob('*.json'):
            self.assertEqual(json.loads(file.read_text())['motions_majority_supported'],[])


if __name__ == '__main__': unittest.main()
