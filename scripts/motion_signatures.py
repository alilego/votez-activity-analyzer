"""Evidence-backed motion signatures. Vote totals are deliberately not parsed.

The activity crawler covers deputies. Senate signatories remain in the source
snapshot and coverage report, not inferred as missing deputy relationships.
"""
import hashlib
import json
import re
from urllib.parse import urljoin, urlparse, parse_qs
from lxml import html
from member_identity import member_id, normalized_name


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS dep_act_motion_sources (
        motion_id TEXT PRIMARY KEY, source_url TEXT NOT NULL,
        source_sha256 TEXT NOT NULL, registered_on TEXT,
        deputy_signatories INTEGER NOT NULL, senate_signatories INTEGER NOT NULL,
        report_json TEXT NOT NULL, fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS dep_act_motion_signatures (
        motion_id TEXT NOT NULL, profile_key TEXT NOT NULL, member_id TEXT,
        source_name TEXT NOT NULL, signing_party_id TEXT NOT NULL,
        chamber TEXT NOT NULL, profile_url TEXT NOT NULL, unresolved_reason TEXT,
        PRIMARY KEY(motion_id,profile_key),
        FOREIGN KEY(member_id) REFERENCES members(member_id))""")


def parse_signatures(page, url):
    doc = html.fromstring(page)
    text = " ".join(doc.itertext())
    date = re.search(r"Data\s+înregistrării:\s*\d+\s*/\s*(\d{2})[.-](\d{2})[.-](\d{4})", text)
    signed_on = f"{date[3]}-{date[2]}-{date[1]}" if date else None
    signatures = {}
    for heading in doc.xpath('//b|//strong'):
        label = " ".join(heading.text_content().split())
        match = re.fullmatch(r"(Deputaţi|Deputați|Senatori)\s+(.+)", label)
        if not match:
            continue
        chamber = "deputat" if match[1].startswith("Deputa") else "senator"
        group = match[2]
        # Use the official group spelling, with the same unaffiliated label as
        # the registry. No current-affiliation fallback, including unknown groups.
        party = "Neafiliaţi" if normalized_name(group) == "neafiliati" else group
        lists = heading.xpath('following-sibling::*[1][self::ol or self::ul]')
        if not lists:
            # The introduction may also bold "Deputaţi din ...". It isn't a
            # party heading; only structured signatory lists are evidence.
            continue
        for link in lists[0].xpath('.//li/a'):
            profile = urljoin(url, link.get('href', ''))
            q = parse_qs(urlparse(profile).query)
            if not all(q.get(k) for k in ('leg','cam','idm')):
                raise ValueError("Signatory has no scoped official profile")
            if q['cam'][0] != ('2' if chamber == 'deputat' else '1'):
                raise ValueError("Signatory chamber disagrees with profile")
            key = "cdep_" + "_".join(str(int(q[k][0])) for k in ('leg','cam','idm'))
            item = dict(profile_key=key, chamber=chamber, source_name=' '.join(link.text_content().split()),
                        signing_party_id=party, profile_url=profile)
            if key in signatures:
                raise ValueError(f"Duplicate/ambiguous official signatory {key}")
            signatures[key] = item
    if not signatures:
        raise ValueError("No official signatory groups found; refusing inferred signatures")
    counts = {ch: sum(s['chamber'] == ch for s in signatures.values()) for ch in ('deputat','senator')}
    # Reject partial pages: the publisher states the number of signatories.
    total = re.search(r"semnată\s+de\s+către\s+(\d+)\s+(?:parlamentari|deputaţi|deputați)", text)
    if total and int(total[1]) != len(signatures):
        raise ValueError(f"Official signatory count {total[1]} != parsed {len(signatures)}")
    if not signed_on:
        raise ValueError("Missing motion registration date")
    return dict(registered_on=signed_on, signatures=list(signatures.values()), counts=counts)


def store_signatures(conn, motion_id, url, page):
    parsed = parse_signatures(page, url)
    ensure_schema(conn)
    members = {r[0]: (r[1],r[2]) for r in conn.execute('SELECT member_id,normalized_name,profile_url FROM members')}
    unresolved = []
    rows = []
    for sig in parsed['signatures']:
        mid = 'deputat_' + sig['profile_key'] if sig['chamber'] == 'deputat' else None
        reason = 'outside_deputy_crawler_coverage' if mid is None else None
        if mid and mid not in members:
            reason, mid = 'official_signatory_missing_from_registry', None
        if mid and normalized_name(sig['source_name']) != members[mid][0]:
            # Name changes need independently verified aliases, not silent trust
            # in an identifier. Repair can supply such aliases explicitly.
            aliases = conn.execute("SELECT 1 FROM member_identity_aliases WHERE member_id=? AND normalized_name=?", (mid,normalized_name(sig['source_name']))).fetchone() if conn.execute("SELECT 1 FROM sqlite_master WHERE name='member_identity_aliases'").fetchone() else None
            if not aliases:
                reason, mid = 'official_profile_name_disagreement', None
        if reason and sig['chamber'] == 'deputat':
            unresolved.append(dict(sig, reason=reason))
        rows.append((motion_id,sig['profile_key'],mid,sig['source_name'],sig['signing_party_id'],sig['chamber'],sig['profile_url'],reason))
    digest = hashlib.sha256(page.encode('utf-8') if isinstance(page,str) else page).hexdigest()
    report = dict(coverage='deputies_only', counts=parsed['counts'], unresolved=unresolved)
    conn.execute('DELETE FROM dep_act_motion_signatures WHERE motion_id=?',(motion_id,))
    conn.executemany('INSERT INTO dep_act_motion_signatures VALUES (?,?,?,?,?,?,?,?)',rows)
    conn.execute('''INSERT INTO dep_act_motion_sources(motion_id,source_url,source_sha256,registered_on,deputy_signatories,senate_signatories,report_json)
        VALUES(?,?,?,?,?,?,?) ON CONFLICT(motion_id) DO UPDATE SET source_url=excluded.source_url,
        source_sha256=excluded.source_sha256,registered_on=excluded.registered_on,deputy_signatories=excluded.deputy_signatories,
        senate_signatories=excluded.senate_signatories,report_json=excluded.report_json''',
        (motion_id,url,digest,parsed['registered_on'],parsed['counts']['deputat'],parsed['counts']['senator'],json.dumps(report,ensure_ascii=False)))
    return report
