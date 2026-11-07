"""Cross-check member and party signature claims before frontend deployment."""
from collections import Counter
import json
from pathlib import Path


def validate_motion_exports(output_dir: Path) -> list[str]:
    errors, sources, member_claims = [], {}, set()
    for file in (output_dir/'activity/members').glob('activity_*.json'):
        payload = json.loads(file.read_text())
        mid = payload['member_id']
        for motion in payload.get('motions',[]):
            motion_id = motion['motion_id']
            signatures = motion.get('signatories',[])
            groups = {s['member_id']:s['signing_party_id'] for s in signatures}
            if len(groups) != len(signatures) or mid not in groups:
                errors.append(f'{file.name}: missing/duplicate official signature for {motion_id}')
                continue
            if motion.get('support_basis') != 'official_signatures' or motion.get('signing_party_id') != groups[mid]:
                errors.append(f'{file.name}: unsupported signing-party attribution for {motion_id}')
            if motion_id in sources and sources[motion_id] != groups:
                errors.append(f'{motion_id}: inconsistent official signatory lists across member exports')
            sources[motion_id] = groups
            counts = Counter(groups.values())
            counts[groups[mid]] -= 1
            actual = {g['party_id']:g['members_count'] for g in motion.get('co_supporting_parties',[])}
            if actual != dict(counts):
                errors.append(f'{file.name}: co-signatory group counts disagree for {motion_id}')
            member_claims.add((motion_id,mid))
    for motion_id, groups in sources.items():
        if any((motion_id,mid) not in member_claims for mid in groups):
            errors.append(f'{motion_id}: a verified signatory has no member export')
    for file in (output_dir/'activity/parties').glob('activity_*.json'):
        payload = json.loads(file.read_text())
        party = payload['party_id']
        claimed = {m['motion_id']:m for m in payload.get('motions_majority_supported',[])}
        for motion_id, groups in sources.items():
            expected = {mid for mid,group in groups.items() if group == party}
            threshold = payload.get('majority_threshold',0)
            if threshold and len(expected) >= threshold and motion_id not in claimed:
                errors.append(f'{file.name}: missing qualifying signed motion {motion_id}')
        for motion_id,motion in claimed.items():
            groups = sources.get(motion_id,{})
            expected = {mid for mid,group in groups.items() if group == party}
            actual = {s['member_id'] for s in motion.get('party_supporters',[])}
            if not expected or actual != expected or motion.get('party_supporters_count') != len(expected):
                errors.append(f'{file.name}: party signatories disagree with member exports for {motion_id}')
            counts = {g['party_id']:g['members_count'] for g in motion.get('all_supporting_parties',[])}
            if counts != dict(Counter(groups.values())):
                errors.append(f'{file.name}: all signatory groups disagree for {motion_id}')
    return errors
