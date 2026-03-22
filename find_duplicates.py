import hashlib
import json
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE_DIR = Path(r"C:\Data\Downloads\outputforlocal\dataset\cleaned_scripts")
DEST_DIR = SOURCE_DIR / "script_deduped_output"

MAX_HAMMING = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simhash(tokens):
    v = [0] * 64
    counts = Counter(tokens)

    for token, w in counts.items():
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        x = int.from_bytes(h, "big")

        for i in range(64):
            if (x >> i) & 1:
                v[i] += w
            else:
                v[i] -= w

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
        except:
            pass
    raise RuntimeError(f"cannot read {path}")


def shorten(name, max_len=120):
    if len(name) <= max_len:
        return name
    stem = Path(name).stem[:80]
    suffix = Path(name).suffix
    return stem + "__trunc" + suffix


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    # avoid overwrite
    base = dst
    i = 1
    while dst.exists():
        dst = base.with_stem(base.stem + f"__{i}")
        i += 1

    shutil.copy2(src, dst)
    return dst


def main():
    log("starting")

    files = list(SOURCE_DIR.rglob("*.txt"))
    log(f"found {len(files)} files")

    unique_dir = DEST_DIR / "unique"
    exact_dir = DEST_DIR / "duplicates_exact"
    near_dir = DEST_DIR / "duplicates_near"

    for d in [unique_dir, exact_dir, near_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data = []

    for i, path in enumerate(files):
        log(f"[{i+1}/{len(files)}] reading {path.name}")

        try:
            raw = read_file(path)
            norm = normalize(raw)
            tokens = tokenize(norm)

            data.append({
                "path": path,
                "name": path.name,
                "hash": sha256(norm),
                "sim": simhash(tokens),
                "len": len(norm),
                "tokens": len(tokens)
            })

            log(f"    ok | chars={len(norm)} hash={data[-1]['hash'][:10]}")

        except Exception as e:
            log(f"    failed | {e}")

    log("grouping duplicates")

    groups = []
    used = set()

    for i in range(len(data)):
        if i in used:
            continue

        group = [i]
        used.add(i)

        for j in range(i + 1, len(data)):
            if j in used:
                continue

            d1 = data[i]
            d2 = data[j]

            if d1["hash"] == d2["hash"]:
                group.append(j)
                used.add(j)
                continue

            dist = hamming(d1["sim"], d2["sim"])
            ratio = max(d1["tokens"], d2["tokens"]) / max(1, min(d1["tokens"], d2["tokens"]))

            if dist <= MAX_HAMMING and ratio < 1.35:
                group.append(j)
                used.add(j)

        groups.append(group)

    log(f"total groups: {len(groups)}")

    manifest = []

    log("copying")

    for gid, group in enumerate(groups):
        if len(group) == 1:
            i = group[0]
            src = data[i]["path"]

            name = shorten(src.name)
            dst = unique_dir / name

            dst = safe_copy(src, dst)

            log(f"    unique | {src.name}")

            manifest.append({
                "file": str(src),
                "type": "unique",
                "dest": str(dst)
            })
            continue

        hashes = set(data[i]["hash"] for i in group)

        if len(hashes) == 1:
            base = exact_dir / f"group_{gid}"
            gtype = "exact"
        else:
            base = near_dir / f"group_{gid}"
            gtype = "near"

        base.mkdir(parents=True, exist_ok=True)

        canonical = max(group, key=lambda i: data[i]["len"])

        for i in group:
            src = data[i]["path"]

            name = src.name
            if i == canonical:
                name = "CANONICAL__" + name

            name = shorten(name)
            dst = base / name

            dst = safe_copy(src, dst)

            log(f"    {gtype} | {src.name}")

            manifest.append({
                "file": str(src),
                "type": gtype,
                "group": gid,
                "dest": str(dst)
            })

    (DEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log("done")


if __name__ == "__main__":
    main()