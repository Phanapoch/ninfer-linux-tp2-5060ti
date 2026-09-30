#!/usr/bin/env python3
"""Insert "-split" into a .ninfer v2 manifest's identity weights_id, in place.

The Ostfralla Qwen3.8-27B NVFP4 export declares weights_id "nvfp4" in its manifest,
but its tensor layout is the split-storage schema that the Windows-TP2 fork
registers as "nvfp4-split". Loading it requires that identity field to read
"nvfp4-split" (see the fork README, "Tested Qwen3.8-27B NVFP4 NInfer artifacts").

The edit is surgical: the byte sequence of the weights_id string value is rewritten
inside the existing manifest padding region and the json_bytes prefix field is
updated, so the payload start offset - and every object offset/bytes in the
directory - is unchanged. The ~18 GB weight payload is never read or copied.

Usage:
  patch_artifact_identity.py <artifact.ninfer> <old-weights-id> <new-weights-id>
  patch_artifact_identity.py <artifact.ninfer> --restore

Examples:
  patch_artifact_identity.py out/qwen3_8_27b_nvfp4.ninfer nvfp4 nvfp4-split
  patch_artifact_identity.py out/qwen3_8_27b_nvfp4.ninfer --restore
"""
import json
import os
import struct
import sys

PREFIX = 16
ALIGN = 4096


def align_up(value, align=ALIGN):
    return (value + align - 1) // align * align


def read_prefix(path):
    with open(path, "rb") as fh:
        prefix = fh.read(PREFIX)
    magic = prefix[:8]
    (json_bytes,) = struct.unpack("<Q", prefix[8:16])
    return magic, json_bytes


def read_manifest(path, json_bytes):
    with open(path, "rb") as fh:
        fh.seek(PREFIX)
        return fh.read(json_bytes)


def find_weights_id_span(raw):
    """Return (start, end) byte span of the weights_id JSON string value."""
    key = b'"weights_id"'
    if raw.count(key) != 1:
        raise SystemExit(f"expected exactly one {key!r} in the manifest, "
                         f"found {raw.count(key)}")
    i = raw.index(key) + len(key)
    while i < len(raw) and raw[i : i + 1].isspace():
        i += 1
    if raw[i : i + 1] != b":":
        raise SystemExit("malformed manifest: no ':' after weights_id")
    i += 1
    while i < len(raw) and raw[i : i + 1].isspace():
        i += 1
    if raw[i : i + 1] != b'"':
        raise SystemExit("malformed manifest: weights_id value is not a string")
    j = raw.index(b'"', i + 1) + 1
    return i, j


def main():
    argv = sys.argv[1:]
    restore = "--restore" in argv
    args = [a for a in argv if not a.startswith("--")]

    if not args:
        print(__doc__)
        return 2
    path = args[0]
    backup = path + ".manifest-backup"

    if restore:
        if not os.path.exists(backup):
            raise SystemExit(f"no backup at {backup}")
        payload_start = os.path.getsize(backup)
        with open(backup, "rb") as fh:
            header = fh.read()
        with open(path, "r+b") as fh:
            fh.write(header)
            fh.flush()
            os.fsync(fh.fileno())
        magic, json_bytes = read_prefix(path)
        raw = read_manifest(path, json_bytes)
        print(f"restored manifest from {backup} ({payload_start} B header region)")
        return 0

    if len(args) < 3:
        print(__doc__)
        return 2
    old_id, new_id = args[1], args[2]

    size = os.path.getsize(path)
    magic, json_bytes = read_prefix(path)
    if magic != b"NINFER\x00\x02":
        raise SystemExit(f"not a v2 artifact (magic {magic!r})")
    raw = read_manifest(path, json_bytes)

    start, end = find_weights_id_span(raw)
    current = raw[start:end]
    if current != ('"%s"' % old_id).encode():
        raise SystemExit(f"weights_id is {current!r}, expected \"{old_id}\"")

    new_literal = ('"%s"' % new_id).encode()
    new_raw = raw[:start] + new_literal + raw[end:]
    delta = len(new_raw) - len(raw)

    payload_start = align_up(PREFIX + json_bytes)
    new_payload_start = align_up(PREFIX + len(new_raw))
    if new_payload_start != payload_start:
        raise SystemExit(f"payload start would move {payload_start} -> "
                         f"{new_payload_start}; refusing")
    max_json = payload_start - PREFIX
    if len(new_raw) > max_json:
        raise SystemExit(f"new manifest ({len(new_raw)} B) exceeds padding budget "
                         f"({max_json} B)")

    # Cross-check the computed payload start against the manifest's own object table.
    # Object offsets are payload-relative, so the last object must end exactly at EOF.
    manifest = json.loads(raw)
    end_of_objects = max(o["offset"] + o["bytes"] for o in manifest["objects"])
    if payload_start + end_of_objects != size:
        raise SystemExit(f"computed payload start {payload_start} inconsistent with the "
                         f"object table: {payload_start} + {end_of_objects} != {size}; "
                         f"refusing")

    if not os.path.exists(backup):
        with open(path, "rb") as fh:
            header = fh.read(payload_start)
        with open(backup, "wb") as fh:
            fh.write(header)

    with open(path, "r+b") as fh:
        fh.write(magic)
        fh.write(struct.pack("<Q", len(new_raw)))
        fh.write(new_raw)
        fh.write(b"\x00" * (payload_start - PREFIX - len(new_raw)))
        fh.flush()
        os.fsync(fh.fileno())

    # Verify: re-read what the engine's reader would see.
    magic2, jb2 = read_prefix(path)
    raw2 = read_manifest(path, jb2)
    s2, e2 = find_weights_id_span(raw2)
    ok_magic = magic2 == magic
    ok_prefix = jb2 == len(new_raw)
    ok_id = raw2[s2:e2] == new_literal
    # Everything outside the manifest must be untouched.
    with open(path, "rb") as fh:
        fh.seek(payload_start)
        payload_head = fh.read(64)
    with open(path, "rb") as fh:
        fh.seek(size - 64)
        payload_tail = fh.read(64)
    assert ok_magic and ok_prefix and ok_id, "verification failed"
    assert os.path.getsize(path) == size, "file size changed"

    print(f'weights_id: "{old_id}" -> {raw2[s2:e2].decode()}')
    print(f"manifest: {json_bytes} -> {jb2} bytes (+{delta})")
    print(f"payload_start: {payload_start} unchanged; file size {size} unchanged")
    print(f"payload head: {payload_head[:16].hex()} | tail: {payload_tail[-16:].hex()}")
    print(f"header backup: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
