#!/usr/bin/env python3
"""Backed-up, transactional repair of legacy positional member identities.

Default is a dry run on a SQLite backup in a temporary file. No LLM work.
Supply immutable historical registry snapshots and cached official motion HTML;
--fetch-sources fills the cache from the DB's official URLs. Every displaced or
unresolved record is retained verbatim in member_identity_audit, including rows
merged on canonical relationship keys. Ambiguous names are never guessed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from urllib.parse import urljoin
from lxml import html

from member_identity import IdentityIndex, member_id, source_member_id, normalized_name
from motion_signatures import ensure_schema, store_signatures

ACTIVITY_TABLES = ('dep_act_member_activity_crawl', 'dep_act_member_laws',
    'dep_act_member_decision_projects', 'dep_act_member_motions',
    'dep_act_questions_interpellations', 'dep_act_political_declarations')


def payload(row):
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True)


def archive(conn, table, row, reason, new_id=None):
    original = payload(row)
    key = hashlib.sha256((table+'\n'+original+'\n'+reason).encode()).hexdigest()
    conn.execute('INSERT OR IGNORE INTO member_identity_audit VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)',
                 (key, table, original, reason, new_id))


def profile_name(page):
    doc = html.fromstring(page)
    # Official profile heading has given names before the uppercase surname.
    headings = doc.xpath('//h1|//h2')
    return [' '.join(h.text_content().split()) for h in headings]


def protected_fingerprints(conn):
    """Content and completed analysis must survive identity-only repair."""
    result = {}
    excluded = {'interventions_raw': {'member_id'},
                'dep_act_questions_interpellations': {'member_id'},
                'dep_act_political_declarations': {'member_id'}}
    tables = ('interventions_raw','intervention_analysis','processed_stenograms',
              'session_topics','session_chunks','dep_act_laws','dep_act_laws_votes',
              'dep_act_questions_interpellations','dep_act_political_declarations')
    for table in tables:
        columns = [r['name'] for r in conn.execute('PRAGMA table_info('+table+')') if r['name'] not in excluded.get(table,set())]
        pk = [r['name'] for r in conn.execute('PRAGMA table_info('+table+')') if r['pk']]
        digest = hashlib.sha256()
        count = 0
        for row in conn.execute('SELECT '+','.join(columns)+' FROM '+table+' ORDER BY '+','.join(pk)):
            digest.update(json.dumps(tuple(row),ensure_ascii=False).encode())
            count += 1
        result[table] = dict(rows=count,sha256=digest.hexdigest())
    return result


def canonical_registry(registries, historical, sources):
    members, aliases = {}, []
    for chamber, rows in registries:
        for row in rows:
            mid = member_id(row, chamber)
            if mid in members:
                raise ValueError('Duplicate official membership in registry: '+mid)
            members[mid] = dict(row, member_id=mid, chamber=chamber, is_current=1)
    roster_file = sources / 'official_roster.html'
    if not roster_file.exists():
        raise ValueError('Need an official roster snapshot to validate scoped profile identifiers')
    roster = {}
    for link in html.fromstring(roster_file.read_text()).xpath('//a[contains(@href,"structura2015.mp")]'):
        row = dict(name=' '.join(link.text_content().split()), profile_url=urljoin('https://www.cdep.ro',link.get('href')))
        mid = member_id(row,'deputat')
        if mid in roster and normalized_name(roster[mid]['name']) != normalized_name(row['name']):
            raise ValueError('Official roster reuses a profile for different names: '+mid)
        roster[mid] = row
    for mid, row in members.items():
        if row['chamber'] == 'deputat' and (mid not in roster or normalized_name(row['name']) != normalized_name(roster[mid]['name'])):
            raise ValueError('Current registry disagrees with official roster: '+row['name'])
    # The official page also links former deputies separately. Retain these
    # people for historical activity, without assigning them a current party.
    for mid, row in roster.items():
        if mid not in members:
            members[mid] = dict(row, member_id=mid, chamber='deputat', party=None, is_current=0)
    # A historical roster is evidence of an old spelling/profile association.
    # A changed/retired name additionally needs a current official profile
    # snapshot. Do not recover a historical person just from their old row ID.
    for row in historical:
        mid = member_id(row, 'deputat')
        existing = members.get(mid)
        if existing and normalized_name(existing['name']) == normalized_name(row['name']):
            continue
        official_id = source_member_id(row, 'deputat').split('_')[-1]
        file = sources / f'profile_{official_id}.html'
        names = profile_name(file.read_text()) if file.exists() else []
        check = IdentityIndex()
        expected = existing['name'] if existing else row['name']
        check.add(expected, mid)
        if not any(check.resolve(n) == mid for n in names):
            raise ValueError('Historical identity needs a verified official profile: '+row['name'])
        historical_file = sources / f'historical_profile_{official_id}.html'
        if existing and normalized_name(existing['name']) != normalized_name(row['name']):
            # Legal name changes require archived official name evidence too.
            # Without it, leave the old spelling unresolved; don't infer an alias.
            if not historical_file.exists():
                continue
            check = IdentityIndex()
            check.add(row['name'],mid)
            old_page = historical_file.read_text()
            links = html.fromstring(old_page).xpath('//a[contains(@href,"structura2015.mp")]/@href')
            if not any(check.resolve(n)==mid for n in profile_name(old_page)) or not any(
                member_id({'profile_url':urljoin('https://www.cdep.ro',link)},'deputat')==mid for link in links):
                raise ValueError('Archived official name/profile evidence disagrees: '+row['name'])
            file = historical_file
        aliases.append((mid,normalized_name(row['name']),str(file)))
        if not existing:
            members[mid] = dict(row, member_id=mid, chamber='deputat', party=None, is_current=0)
    return members, aliases


def repair(conn, canonical, aliases, pages):
    """Caller owns BEGIN/COMMIT. Suitable for dry-run and repeatable migration."""
    conn.row_factory = sqlite3.Row
    protected_before = protected_fingerprints(conn)
    report = {'tables': {}, 'motions': {}, 'unresolved': [], 'schema_version': 1}
    linked = []
    for table, in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if any(c['name'] == 'member_id' for c in conn.execute(f'PRAGMA table_info("{table}")')):
            linked.append(table)
    allowed = set(ACTIVITY_TABLES) | {'members','interventions_raw','dep_act_motion_signatures','member_identity_aliases'}
    unknown = set(linked) - allowed
    if unknown:
        raise ValueError('Unaudited member-linked tables: '+repr(sorted(unknown)))
    report['audited_member_tables'] = linked
    report['name_keyed_tables'] = ['dep_act_laws_votes']
    report['historical_artifacts'] = 'run_outputs and run_* files are immutable run snapshots, not current identity indexes'
    conn.execute('''CREATE TABLE IF NOT EXISTS member_identity_audit(
        audit_id TEXT PRIMARY KEY, table_name TEXT NOT NULL, original_json TEXT NOT NULL,
        reason TEXT NOT NULL, resolved_member_id TEXT, recorded_at TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS member_identity_aliases(
        member_id TEXT NOT NULL, normalized_name TEXT NOT NULL, evidence_path TEXT NOT NULL,
        PRIMARY KEY(member_id,normalized_name))''')
    if 'is_current' not in [r['name'] for r in conn.execute('PRAGMA table_info(members)')]:
        conn.execute('ALTER TABLE members ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1')
    old_members = {r['member_id']:dict(r) for r in conn.execute('SELECT * FROM members')}
    cols = [r['name'] for r in conn.execute('PRAGMA table_info(members)')]
    # Insert canonical members before moving children, with FK checking deferred
    # by the caller. Old namespace source IDs cannot collide with scoped IDs.
    for mid, member in canonical.items():
        if mid in old_members:
            continue
        old = next((r for r in old_members.values()
                    if member_id(r,r['chamber']) == mid), {})
        record = {k:old[k] for k in cols if k in old}
        record.update(member_id=mid, source_member_id=source_member_id(member,member['chamber']),
                      chamber=member['chamber'], name=member['name'], normalized_name=normalized_name(member['name']),
                      party_id=member.get('party'), profile_url=member['profile_url'],
                      circumscriptie=member.get('circumscriptie'), is_current=member['is_current'])
        keys = list(record)
        conn.execute('INSERT INTO members('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')', [record[k] for k in keys])
    for row in aliases:
        conn.execute('INSERT OR IGNORE INTO member_identity_aliases VALUES(?,?,?)', row)
    index = IdentityIndex()
    for mid, member in canonical.items():
        index.add(member['name'],mid)
    for mid, name, _ in aliases:
        index.add(name,mid)
    for table in ACTIVITY_TABLES:
        rows = conn.execute('SELECT * FROM '+table).fetchall()
        pk = [r['name'] for r in sorted(conn.execute('PRAGMA table_info('+table+')'),key=lambda x:x['pk']) if r['pk']]
        stats = Counter(before=len(rows))
        planned = {}
        for row in rows:
            name = row['member_normalized_name']
            mid = index.resolve(name)
            # Crawl checkpoints have authoritative profile links too. Refuse
            # contradictory name/profile evidence rather than trusting rank.
            if table == 'dep_act_member_activity_crawl' and mid:
                if member_id(dict(row),'deputat') != mid:
                    mid = None
            if not mid:
                archive(conn,table,row,'unresolved_identity')
                report['unresolved'].append(dict(table=table, member_id=row['member_id'], recorded_name=name, reason='missing_or_ambiguous_identity'))
                stats['unresolved'] += 1
                continue
            new = dict(row, member_id=mid)
            if row['member_id'] != mid:
                archive(conn,table,row,'identity_remapped',mid)
                stats['remapped'] += 1
                if row['member_id'] in old_members and name != old_members[row['member_id']]['normalized_name']:
                    stats['stale_identity_repaired'] += 1
            key = tuple(new[k] for k in pk)
            if key in planned:
                archive(conn,table,row,'duplicate_canonical_relationship',mid)
                previous = planned[key]
                archive(conn,table,previous,'canonical_relationship_merged',mid)
                if 'is_initiator' in new:
                    previous['is_initiator'] = max(previous['is_initiator'],new['is_initiator'])
                if 'created_at' in new:
                    previous['created_at'] = min(previous['created_at'],new['created_at'])
                stats['duplicates_merged'] += 1
            else:
                planned[key] = new
        if stats['remapped'] or stats['unresolved'] or stats['duplicates_merged']:
            conn.execute('DELETE FROM '+table)
            if planned:
                keys = list(next(iter(planned.values())))
                conn.executemany('INSERT INTO '+table+'('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')',
                                 [[r[k] for k in keys] for r in planned.values()])
        stats['after'] = len(planned)
        report['tables'][table] = dict(stats)
    stats = Counter()
    unresolved_speakers = Counter()
    for row in conn.execute('SELECT * FROM interventions_raw').fetchall():
        # Speech source identifies its speaker, not the overwritten member row.
        mid = index.resolve(row['normalized_speaker'])
        if mid != row['member_id']:
            if row['member_id']:
                archive(conn,'interventions_raw',row,'speech_identity_remapped' if mid else 'unresolved_speech_identity',mid)
            conn.execute('UPDATE interventions_raw SET member_id=? WHERE intervention_id=?',(mid,row['intervention_id']))
            stats['remapped' if mid else 'unresolved_previously_attributed'] += 1
            if mid and row['member_id'] in old_members and member_id(old_members[row['member_id']],old_members[row['member_id']]['chamber']) != mid:
                stats['stale_identity_repaired'] += 1
        if not mid:
            unresolved_speakers[row['normalized_speaker']] += 1
    report['tables']['interventions_raw'] = dict(stats)
    report['unresolved_speakers'] = dict(unresolved_speakers)
    for old_id, row in old_members.items():
        if old_id not in canonical:
            archive(conn,'members',row,'obsolete_positional_identity')
            conn.execute('DELETE FROM members WHERE member_id=?',(old_id,))
    ensure_schema(conn)
    for motion_id, url, page in pages:
        report['motions'][motion_id] = store_signatures(conn,motion_id,url,page)
        # Authoritative deputy signatures reconcile the legacy association table
        # too. Preserve displaced links, rather than merely hiding them in JSON.
        official = {r[0] for r in conn.execute('SELECT member_id FROM dep_act_motion_signatures WHERE motion_id=? AND member_id IS NOT NULL',(motion_id,))}
        old_links = {r[0] for r in conn.execute('SELECT member_id FROM dep_act_member_motions WHERE motion_id=?',(motion_id,))}
        report['motions'][motion_id]['relationships_added'] = len(official-old_links)
        report['motions'][motion_id]['relationships_removed'] = len(old_links-official)
        for row in conn.execute('SELECT * FROM dep_act_member_motions WHERE motion_id=?',(motion_id,)).fetchall():
            if row['member_id'] not in official:
                archive(conn,'dep_act_member_motions',row,'not_verified_by_official_signatory_list')
                conn.execute('DELETE FROM dep_act_member_motions WHERE motion_id=? AND member_id=?',(motion_id,row['member_id']))
        for mid in official:
            conn.execute('INSERT OR IGNORE INTO dep_act_member_motions(member_id,member_normalized_name,motion_id) VALUES(?,?,?)',
                         (mid,normalized_name(canonical[mid]['name']),motion_id))
    report['tables']['dep_act_member_motions']['after'] = conn.execute('SELECT count(*) FROM dep_act_member_motions').fetchone()[0]
    report['members_before'] = len(old_members)
    report['members_after'] = conn.execute('SELECT count(*) FROM members').fetchone()[0]
    report['audit_rows'] = conn.execute('SELECT count(*) FROM member_identity_audit').fetchone()[0]
    protected_after = protected_fingerprints(conn)
    if protected_before != protected_after:
        raise ValueError('Repair changed protected content or analysis; rolling back')
    report['protected_content'] = protected_after
    violations = conn.execute('PRAGMA foreign_key_check').fetchall()
    if violations:
        raise ValueError('Foreign key violations: '+repr([tuple(v) for v in violations[:20]]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=Path('state/state.sqlite'))
    parser.add_argument('--sources',type=Path,default=Path('state/identity_repair/sources'))
    parser.add_argument('--historical-registry',type=Path,action='append',default=[])
    parser.add_argument('--fetch-sources',action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--dry-run',action='store_true',help='Report on a temporary copy (the default)')
    parser.add_argument('--report',type=Path,default=Path('state/identity_repair/report.json'))
    args = parser.parse_args()
    args.sources.mkdir(parents=True,exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_path = args.sources.parent / f'before_identity_repair_{timestamp}.sqlite'
    with sqlite3.connect(args.db) as original, sqlite3.connect(backup_path) as backup:
        original.backup(backup)
    print('Backup:',backup_path,flush=True)
    registries = [(ch,json.loads(Path('input/'+file).read_text())['members']) for ch,file in
                  [('deputat','toti_deputatii.json'),('senator','toti_senatorii.json')]]
    historical = [r for file in args.historical_registry for r in json.loads(file.read_text())['members']]
    if args.fetch_sources:
        from crawl_deputy_activity import Fetcher
        fetcher = Fetcher()
        from urllib.parse import parse_qs, urlparse
        legs = {parse_qs(urlparse(r['profile_url']).query)['leg'][0] for ch,rows in registries if ch=='deputat' for r in rows}
        if len(legs) != 1:
            raise ValueError('Use a separately verified roster snapshot for multiple legislatures')
        roster_file = args.sources/'official_roster.html'
        if not roster_file.exists():
            roster_file.write_text(fetcher.fetch('https://www.cdep.ro/ords/pls/parlam/structura2015.de?leg='+next(iter(legs))+'&idl=1'))
        current = {member_id(r,ch):r for ch,rows in registries for r in rows}
        for row in historical:
            mid = member_id(row,'deputat')
            if mid not in current or normalized_name(current[mid]['name']) != normalized_name(row['name']):
                file = args.sources/('profile_'+source_member_id(row,'deputat').split('_')[-1]+'.html')
                if not file.exists():
                    file.write_text(fetcher.fetch(row['profile_url']))
                archived = args.sources/('historical_profile_'+source_member_id(row,'deputat').split('_')[-1]+'.html')
                cached = Path('../votez-scraper/downloaded_pages/profiles')/(row['name']+'.html')
                if not archived.exists() and cached.exists():
                    archived.write_bytes(cached.read_bytes())
    canonical, aliases = canonical_registry(registries,historical,args.sources)
    with sqlite3.connect(args.db) as conn:
        motions = conn.execute('SELECT motion_id,source_url FROM dep_act_motions').fetchall()
    pages = []
    for mid,url in motions:
        file = args.sources / (mid.replace(':','_')+'.html')
        if not file.exists() and args.fetch_sources:
            file.write_text(fetcher.fetch(url))
        if not file.exists():
            raise ValueError('Missing official motion snapshot: '+str(file))
        pages.append((mid,url,file.read_text()))
    with tempfile.TemporaryDirectory() as temp:
        target = args.db if args.apply else Path(temp)/'dry-run.sqlite'
        if not args.apply:
            with sqlite3.connect(backup_path) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
        conn = sqlite3.connect(target)
        try:
            # Legacy PKs must be removed after children migrate; check every FK
            # explicitly before commit. This connection owns the entire repair.
            conn.execute('PRAGMA foreign_keys=OFF')
            conn.execute('BEGIN IMMEDIATE')
            report = repair(conn,canonical,aliases,pages)
            report.update(applied=args.apply,backup=str(backup_path),sources=[dict(motion_id=m,sha256=hashlib.sha256(p.encode()).hexdigest()) for m,_,p in pages])
            evidence = [args.sources/'official_roster.html',Path('input/toti_deputatii.json'),Path('input/toti_senatorii.json')] + args.historical_registry
            evidence += list(args.sources.glob('*profile_*.html'))
            report['identity_evidence'] = [dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in evidence]
            if args.apply:
                conn.commit()
            else:
                conn.rollback()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('unresolved','unresolved_speakers','sources')},ensure_ascii=False,indent=2))
    print('Full report:',args.report)


if __name__ == '__main__':
    main()
