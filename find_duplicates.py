import hashlib
import json
import re
import shutil
import time
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path


SOURCE_DIR = Path(r"C:\Data\Downloads\outputforlocal\dataset\cleaned_scripts")
DEST_DIR = SOURCE_DIR / "script_deduped_output"

MAX_HAMMING = 6
TOKEN_RATIO_LIMIT = 1.35
TITLE_SIM_THRESHOLD = 0.60

EXCLUDE_DIR_NAMES = {
    "script_deduped_output",
    "duplicates",
    "duplicates_exact",
    "duplicates_near",
    "unique",
    "_manifests",
}

NAME_STOPWORDS = {
    "film", "tv", "pilot", "script", "screenplay", "teleplay", "draft",
    "shooting", "final", "revision", "revised", "rev", "fka", "transcript",
    "first", "second", "third", "fourth", "fifth", "sixth",
    "the", "a", "an", "and", "of", "for", "by", "to", "from"
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_text(text):
    return re.findall(r"[a-z0-9']+", text)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simhash(tokens):
    v = [0] * 64
    counts = Counter(tokens)

    for token, weight in counts.items():
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "big")

        for i in range(64):
            if (x >> i) & 1:
                v[i] += weight
            else:
                v[i] -= weight

    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)

    return out


def hamming(a, b):
    return (a ^ b).bit_count()


def read_file(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"cannot read {path}")


def should_skip(path):
    try:
        rel_parts = path.relative_to(SOURCE_DIR).parts
    except ValueError:
        return False

    return any(part.lower() in EXCLUDE_DIR_NAMES for part in rel_parts[:-1])


def shorten(name, max_len=120):
    if len(name) <= max_len:
        return name

    stem = Path(name).stem
    suffix = Path(name).suffix

    stem_hash = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(suffix) - len(stem_hash) - len("__trunc_")
    keep = max(20, keep)

    return stem[:keep] + "__trunc_" + stem_hash + suffix


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    base = dst
    i = 1
    while dst.exists():
        dst = base.with_stem(base.stem + f"__{i}")
        i += 1

    shutil.copy2(src, dst)
    return dst


def extract_title_candidate(file_name):
    stem = Path(file_name).stem
    stem = unicodedata.normalize("NFKC", stem)

    if "_" in stem:
        left = re.split(r"_+", stem, maxsplit=1)[0]
        if left.strip():
            stem = left

    stem = re.split(
        r"(?i)\b(film|tv|pilot|script|screenplay|teleplay|draft|shooting|final|revision|revised|rev|fka|transcript)\b",
        stem,
        maxsplit=1
    )[0]

    return stem.strip()


def title_tokens(file_name):
    title = extract_title_candidate(file_name).lower()
    title = title.replace("&", " and ")
    title = title.replace("+", " plus ")
    title = unicodedata.normalize("NFKC", title)

    tokens = re.findall(r"[a-z0-9]+", title)
    cleaned = []

    for tok in tokens:
        if tok in NAME_STOPWORDS:
            continue
        if re.fullmatch(r"(19|20)\d{2}", tok):
            continue
        cleaned.append(tok)

    if cleaned:
        return set(cleaned)

    fallback = re.findall(r"[a-z0-9]+", Path(file_name).stem.lower())
    fallback = [t for t in fallback if t not in NAME_STOPWORDS]
    return set(fallback)


def title_similarity(name_a, name_b):
    a = title_tokens(name_a)
    b = title_tokens(name_b)

    if not a or not b:
        return 0.0

    inter = len(a & b)
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0

    return inter / denom


def connected_components(nodes, edge_fn):
    nodes = list(nodes)
    seen = set()
    comps = []

    for start in nodes:
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            for other in nodes:
                if other in seen:
                    continue
                if edge_fn(cur, other):
                    seen.add(other)
                    stack.append(other)

        comps.append(sorted(comp))

    return comps


def content_near(d1, d2):
    if d1["hash"] == d2["hash"]:
        return True

    dist = hamming(d1["sim"], d2["sim"])
    ratio = max(d1["tokens"], d2["tokens"]) / max(1, min(d1["tokens"], d2["tokens"]))

    return dist <= MAX_HAMMING and ratio <= TOKEN_RATIO_LIMIT


def name_similar(d1, d2):
    return title_similarity(d1["name"], d2["name"]) >= TITLE_SIM_THRESHOLD


def main():
    log("starting")

    files = [p for p in SOURCE_DIR.rglob("*.txt") if not should_skip(p)]
    files = sorted(files)

    log(f"source: {SOURCE_DIR}")
    log(f"destination: {DEST_DIR}")
    log(f"found {len(files)} txt files after exclusions")

    unique_dir = DEST_DIR / "unique"
    exact_dir = DEST_DIR / "duplicates_exact"
    near_dir = DEST_DIR / "duplicates_near"
    manifests_dir = DEST_DIR / "_manifests"

    for d in [unique_dir, exact_dir, near_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data = []
    failures = []

    for i, path in enumerate(files, start=1):
        log(f"[{i}/{len(files)}] reading {path.name}")

        try:
            raw = read_file(path)
            norm = normalize_text(raw)
            tokens = tokenize_text(norm)

            record = {
                "path": path,
                "name": path.name,
                "hash": sha256(norm),
                "sim": simhash(tokens),
                "char_count": len(norm),
                "line_count": norm.count("\n") + 1 if norm else 0,
                "tokens": len(tokens),
                "title_tokens": sorted(title_tokens(path.name)),
            }
            data.append(record)

            log(
                f"    ok | chars={record['char_count']} "
                f"tokens={record['tokens']} "
                f"hash={record['hash'][:10]} "
                f"title={record['title_tokens']}"
            )

        except Exception as e:
            failures.append({"file": str(path), "error": str(e)})
            log(f"    failed | {e}")

    if not data:
        log("no readable files found")
        (manifests_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        return

    log("building exact-duplicate buckets")

    exact_map = defaultdict(list)
    for idx, d in enumerate(data):
        exact_map[d["hash"]].append(idx)

    exact_buckets = list(exact_map.values())
    log(f"exact hash buckets: {len(exact_buckets)}")

    rep_indices = [bucket[0] for bucket in exact_buckets]

    log("building content-based candidate groups from bucket representatives")

    candidate_components = connected_components(
        rep_indices,
        lambda i, j: content_near(data[i], data[j])
    )

    log(f"content candidate groups: {len(candidate_components)}")

    candidate_groups = []
    for comp in candidate_components:
        members = []
        for rep_idx in comp:
            members.extend(exact_map[data[rep_idx]["hash"]])
        candidate_groups.append(sorted(members))

    final_groups = []

    log("refining candidate groups with regex-based filename title matching")

    for group_idx, members in enumerate(candidate_groups, start=1):
        hashes = {data[i]["hash"] for i in members}

        if len(members) == 1:
            final_groups.append({"type": "unique", "members": members})
            continue

        if len(hashes) == 1:
            final_groups.append({"type": "exact", "members": members})
            continue

        log(f"    candidate group {group_idx}: {len(members)} files")

        name_components = connected_components(
            members,
            lambda i, j: data[i]["hash"] == data[j]["hash"] or name_similar(data[i], data[j])
        )

        if len(name_components) > 1:
            log(f"        split into {len(name_components)} filename-based subgroups")
            for sub_idx, comp in enumerate(name_components, start=1):
                names = [data[i]["name"] for i in comp]
                log(f"        subgroup {sub_idx}: {names}")

        for comp in name_components:
            comp_hashes = {data[i]["hash"] for i in comp}

            if len(comp) == 1:
                final_groups.append({"type": "unique", "members": comp})
            elif len(comp_hashes) == 1:
                final_groups.append({"type": "exact", "members": comp})
            else:
                final_groups.append({"type": "near", "members": comp})

    unique_count = sum(1 for g in final_groups if g["type"] == "unique")
    exact_count = sum(1 for g in final_groups if g["type"] == "exact")
    near_count = sum(1 for g in final_groups if g["type"] == "near")

    log(
        f"final groups | unique={unique_count} "
        f"exact={exact_count} near={near_count}"
    )

    manifest = []

    exact_num = 0
    near_num = 0

    log("copying files")

    for group in final_groups:
        members = group["members"]
        gtype = group["type"]

        if gtype == "unique":
            idx = members[0]
            src = data[idx]["path"]

            out_name = shorten(src.name)
            dst = safe_copy(src, unique_dir / out_name)

            log(f"    unique | {src.name} -> {dst}")

            manifest.append({
                "file": str(src),
                "type": "unique",
                "group_folder": "",
                "copied_to": [str(dst)],
                "representative_in_unique": True,
                "title_tokens": data[idx]["title_tokens"],
            })
            continue

        if gtype == "exact":
            exact_num += 1
            group_folder = exact_dir / f"group_{exact_num:04d}"
        else:
            near_num += 1
            group_folder = near_dir / f"group_{near_num:04d}"

        group_folder.mkdir(parents=True, exist_ok=True)

        canonical_idx = max(members, key=lambda i: data[i]["char_count"])
        canonical_src = data[canonical_idx]["path"]

        rep_name = shorten(f"REP__{canonical_src.name}")
        rep_dst = safe_copy(canonical_src, unique_dir / rep_name)

        log(
            f"    representative -> unique | {canonical_src.name} -> {rep_dst}"
        )

        for idx in members:
            src = data[idx]["path"]

            if idx == canonical_idx:
                out_name = shorten("CANONICAL__" + src.name)
            else:
                out_name = shorten(src.name)

            dst = safe_copy(src, group_folder / out_name)

            log(f"    {gtype} | {src.name} -> {dst}")

            manifest.append({
                "file": str(src),
                "type": gtype,
                "group_folder": str(group_folder),
                "copied_to": [str(dst)] if idx != canonical_idx else [str(dst), str(rep_dst)],
                "representative_in_unique": idx == canonical_idx,
                "canonical_source": str(canonical_src),
                "title_tokens": data[idx]["title_tokens"],
            })

    summary = {
        "source": str(SOURCE_DIR),
        "destination": str(DEST_DIR),
        "files_processed": len(data),
        "files_failed": len(failures),
        "groups_unique": unique_count,
        "groups_exact": exact_count,
        "groups_near": near_count,
        "max_hamming": MAX_HAMMING,
        "token_ratio_limit": TOKEN_RATIO_LIMIT,
        "title_sim_threshold": TITLE_SIM_THRESHOLD,
    }

    (manifests_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (manifests_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (manifests_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    log("done")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
