"""Official membership keys, never alphabetical roster positions.

CDep idm is scoped to (leg, cam), not a lifetime person identifier. Senate
ParlamentarID is the official profile UUID; retain that existing namespace.
Names are evidence for repairing legacy links, never primary keys.
"""
import re
import unicodedata
from urllib.parse import urlparse, parse_qs


def source_member_id(row, chamber):
    p = urlparse(str(row.get("profile_url", "")))
    q = {k.lower(): v for k, v in parse_qs(p.query).items()}
    host = (p.hostname or "").lower().removeprefix("www.")
    if chamber == "deputat":
        if host != "cdep.ro" or not p.path.lower().endswith("structura2015.mp"):
            raise ValueError("Deputy identity requires an official CDep profile")
        vals = [q.get(k, [""])[0] for k in ("leg", "cam", "idm")]
        if not all(re.fullmatch(r"[0-9]+", v) for v in vals) or vals[1] != "2":
            raise ValueError("Deputy identity requires legislature, chamber=2 and idm")
        return "cdep_" + "_".join(str(int(v)) for v in vals)
    value = q.get("parlamentarid", [""])[0].lower()
    if host != "senat.ro" or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value):
        raise ValueError("Senator identity requires an official Senate profile UUID")
    return value


def member_id(row, chamber):
    return chamber + "_" + source_member_id(row, chamber)


def normalized_name(value):
    value = re.sub(r"\s+-\s+.*$", "", value or "")
    value = re.sub(r"^\s*(domnul|doamna)\s+", "", value, flags=re.I)
    value = re.sub(r"\([^)]*\)", "", value)
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", value).casefold()
                            if not unicodedata.combining(c)).split())


class IdentityIndex:
    """Exact spelling first; rotated speech names only when unambiguous.

    Sorting tokens would conflate Stoica Alin-Bogdan and Stoica Bogdan-Alin.
    Preserve the order of given names and generate only surname rotations.
    """
    def __init__(self):
        self.names = {}
        self.tokens = {}

    def add(self, name, mid):
        name = ' '.join(re.findall(r'[a-z0-9]+', normalized_name(name)))
        parts = name.split()
        for i in range(len(parts)):
            alias = ' '.join(parts[i:] + parts[:i])
            self.names.setdefault(alias, set()).add(mid)
        self.tokens.setdefault(' '.join(sorted(parts)), set()).add(mid)

    def resolve(self, name):
        name = ' '.join(re.findall(r'[a-z0-9]+', normalized_name(name)))
        candidates = self.names.get(name, set())
        if not candidates:
            candidates = self.tokens.get(' '.join(sorted(name.split())), set())
        return next(iter(candidates)) if len(candidates) == 1 else None


def guard_member_identity(conn, members):
    """Refuse legacy databases and profile/key disagreements before upserts."""
    for old_id, chamber, url in conn.execute("SELECT member_id,chamber,profile_url FROM members"):
        if old_id != member_id({"profile_url": url}, chamber):
            raise ValueError("Unsafe legacy member identity: run scripts/repair_member_identities.py first")
    for row in members:
        if row["member_id"] != member_id(row, row["chamber"]):
            raise ValueError("Member key does not match official profile")
