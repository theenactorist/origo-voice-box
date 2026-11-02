#!/usr/bin/env python3
"""Rebuild the local macOS sidecar with new backend code and its existing dependencies.

Run with Python 3.12 and PyInstaller 6.22.0 installed in an isolated build venv:

    python scripts/repack-origo-backend.py --output /path/to/new/voicebox-server

The trusted source app supplies its original bootloader, Python, native libraries,
third-party modules, package data, and entitlements. Only backend modules and the
server entry point are compiled from this checkout. No source-app files, user data,
model caches, or running processes are modified. The result is a signed, standalone
onefile executable; it has no dependency on this build venv or a _MEI directory.

This is an incremental local packaging path, not dependency analysis: any newly
introduced third-party dependency must already exist in the source archive. Use the
normal build_binary.py workflow when dependencies themselves need to change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import PyInstaller
from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter
from PyInstaller.utils import osx


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_exact(stream, length: int) -> bytes:
    result = stream.read(length)
    if len(result) != length:
        raise ValueError(f"Truncated archive: wanted {length} bytes, got {len(result)}")
    return result


def copy_region(source, destination, start: int, length: int) -> str:
    source.seek(start)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        data = read_exact(source, min(1024 * 1024, remaining))
        digest.update(data)
        if destination is not None:
            destination.write(data)
        remaining -= len(data)
    return digest.hexdigest()


def archive_layout(path: Path):
    """Keep the original ordered TOC, including any duplicate runtime options."""
    reader = CArchiveReader(str(path))
    with path.open("rb") as stream:
        stream.seek(reader._end_offset - CArchiveReader._COOKIE_LENGTH)
        cookie = struct.unpack(CArchiveReader._COOKIE_FORMAT, read_exact(stream, 88))
        _, _, toc_offset, toc_length, version, library = cookie
        stream.seek(reader._start_offset + toc_offset)
        toc_data = read_exact(stream, toc_length)
    entries = []
    cursor = 0
    while cursor < len(toc_data):
        size, offset, packed, unpacked, compressed, kind = struct.unpack_from("!IIIIBc", toc_data, cursor)
        if size < 19 or cursor + size > len(toc_data):
            raise ValueError("Invalid CArchive table entry")
        name = toc_data[cursor + 18 : cursor + size].rstrip(b"\0").decode("utf-8")
        if offset + packed > toc_offset:
            raise ValueError(f"CArchive payload extends past the data area: {name}")
        entries.append((offset, packed, unpacked, compressed, kind.decode("ascii"), name))
        cursor += size
    return reader, entries, version, library


def backend_sources(root: Path):
    sources = {}
    excluded = {"tests", "pyi_hooks", "venv", ".venv", "build", "dist", "__pycache__"}
    for path in sorted((root / "backend").rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.name == "build_binary.py" or path.name.startswith("pyi_rth_"):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Backend source escapes the checkout: {path}")
        package = path.name == "__init__.py"
        parts = relative.parts[:-1] if package else relative.with_suffix("").parts
        name = ".".join(parts)
        sources[name] = (1 if package else 0, relative.as_posix(), path.read_bytes())
    for required in ("backend", "backend.main", "backend.config", "backend.server", "backend.backends"):
        if required not in sources:
            raise ValueError(f"Required backend source is missing: {required}")
    return sources


def is_backend(name: str) -> bool:
    return name == "backend" or name.startswith("backend.")


def rebuild_pyz(source_path, original, pyz_name, sources, output):
    original_pyz = original.open_embedded_archive(pyz_name)
    old_backend = {name for name in original_pyz.toc if is_backend(name)}
    with source_path.open("rb") as source:
        source.seek(original_pyz._start_offset)
        header = read_exact(source, 12)
        if header[:4] != b"PYZ\0" or header[4:8] != importlib.util.MAGIC_NUMBER:
            raise ValueError("Build interpreter bytecode does not match the source PYZ")
        toc = []
        preserved = {}
        expected = {}
        with output.open("wb") as target:
            target.write(b"\0" * 17)
            for name in sorted((set(original_pyz.toc) - old_backend) | set(sources)):
                offset = target.tell()
                if name in sources:
                    kind, filename, content = sources[name]
                    code = compile(content, filename, "exec", dont_inherit=True, optimize=0)
                    raw = marshal.dumps(code)
                    data = zlib.compress(raw, 6)
                    target.write(data)
                    expected[name] = hashlib.sha256(raw).hexdigest()
                    size = len(data)
                else:
                    kind, old_offset, size = original_pyz.toc[name]
                    preserved[name] = copy_region(source, target, original_pyz._start_offset + old_offset, size)
                toc.append((name, (kind, offset, size)))
            toc_offset = target.tell()
            target.write(marshal.dumps(toc))
            target.seek(0)
            target.write(b"PYZ\0" + importlib.util.MAGIC_NUMBER + struct.pack("!i", toc_offset))
    return preserved, expected, old_backend


def rebuild_package(source_path, reader, entries, version, library, pyz_name, new_pyz, sources, output):
    preserved = {}
    replaced = {}
    toc = []
    with source_path.open("rb") as source, output.open("wb") as target:
        for old_offset, size, unpacked, compressed, kind, name in entries:
            offset = target.tell()
            if name == pyz_name:
                if compressed:
                    raise ValueError("The embedded PYZ must be uncompressed")
                with new_pyz.open("rb") as pyz_stream:
                    shutil.copyfileobj(pyz_stream, target)
                size = unpacked = new_pyz.stat().st_size
                replaced[name] = sha256_file(new_pyz)
            elif name == "server" and kind == "s":
                content = sources["backend.server"][2]
                raw = marshal.dumps(compile(content, "server.py", "exec", dont_inherit=True, optimize=0))
                data = zlib.compress(raw, 9) if compressed else raw
                target.write(data)
                size, unpacked = len(data), len(raw)
                replaced[name] = hashlib.sha256(raw).hexdigest()
            elif name.startswith("backend/") and name.endswith(".py"):
                module = name[:-3].replace("/", ".")
                if module.endswith(".__init__"):
                    module = module[: -len(".__init__")]
                if module not in sources:
                    raise ValueError(f"Physical backend source has no checkout counterpart: {name}")
                raw = sources[module][2]
                data = zlib.compress(raw, 9) if compressed else raw
                target.write(data)
                size, unpacked = len(data), len(raw)
                replaced[name] = hashlib.sha256(raw).hexdigest()
            else:
                preserved[(name, kind)] = copy_region(source, target, reader._start_offset + old_offset, size)
            toc.append((offset, size, unpacked, compressed, kind, name))
        toc_offset = target.tell()
        table = CArchiveWriter._serialize_toc(toc)
        target.write(table)
        target.write(struct.pack("!8sIIII64s", CArchiveReader._COOKIE_MAGIC_PATTERN,
                                 toc_offset + len(table) + 88, toc_offset, len(table), version, library))
    return preserved, replaced


def verify_result(path, original_entries, pyz_name, sources, preserved_pyz, expected_pyz, preserved_pkg, replaced_pkg):
    reader, entries, _, _ = archive_layout(path)
    if [(e[4], e[5]) for e in entries] != [(e[4], e[5]) for e in original_entries]:
        raise ValueError("CArchive entries changed unexpectedly")
    for before, after in zip(original_entries, entries, strict=True):
        if (after[5], after[4]) in preserved_pkg and before[1:] != after[1:]:
            raise ValueError(f"Preserved archive metadata changed: {after[5]}")
    with path.open("rb") as stream:
        for offset, size, _, _, kind, name in entries:
            if (name, kind) in preserved_pkg:
                actual = copy_region(stream, None, reader._start_offset + offset, size)
                if actual != preserved_pkg[(name, kind)]:
                    raise ValueError(f"Preserved archive payload changed: {name}")
    for name, expected in replaced_pkg.items():
        if hashlib.sha256(reader.extract(name)).hexdigest() != expected:
            raise ValueError(f"Replacement archive payload failed verification: {name}")
    pyz = reader.open_embedded_archive(pyz_name)
    if set(pyz.toc) != set(preserved_pyz) | set(sources):
        raise ValueError("PYZ module inventory changed unexpectedly")
    with path.open("rb") as stream:
        for name, (kind, offset, size) in pyz.toc.items():
            if name in preserved_pyz:
                actual = copy_region(stream, None, pyz._start_offset + offset, size)
                if actual != preserved_pyz[name]:
                    raise ValueError(f"Preserved PYZ module changed: {name}")
            else:
                stream.seek(pyz._start_offset + offset)
                raw = zlib.decompress(read_exact(stream, size))
                if kind != sources[name][0] or hashlib.sha256(raw).hexdigest() != expected_pyz[name]:
                    raise ValueError(f"Compiled backend module failed verification: {name}")
    subprocess.run(["/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(path)], check=True)
    return len(entries), len(pyz.toc)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-app", type=Path, default=Path("/Applications/Origo Voice Box.app"))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 12):
        parser.error("This local packaging path requires macOS and Python 3.12")
    source_app, root, output = args.source_app.resolve(), args.source_root.resolve(), args.output.resolve()
    source = source_app / "Contents/MacOS/voicebox-server"
    report = (args.report or output.with_suffix(".audit.json")).resolve()
    if any(p.is_relative_to(source_app) or p.is_relative_to(Path("/Applications")) for p in (output, report)):
        parser.error("Output and report must be outside installed application bundles")
    if output == report or output.exists() or report.exists():
        parser.error("Output and report must be distinct new paths; existing files are never overwritten")
    with (source_app / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if info.get("CFBundleIdentifier") != "sh.voicebox.app":
        parser.error("The source app must have bundle identifier sh.voicebox.app")
    if osx.get_binary_architectures(str(source)) != (False, ["arm64"]):
        parser.error("This helper currently supports the thin arm64 sidecar only")
    subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(source)], check=True)
    original_hash = sha256_file(source)
    reader, entries, version, library = archive_layout(source)
    if version != 312:
        parser.error(f"Source archive contains Python {version}, expected 312")
    pyz_names = [e[5] for e in entries if e[4] == "z"]
    if len(pyz_names) != 1 or not any(e[4:6] == ("s", "server") for e in entries):
        parser.error("Expected exactly one PYZ and one server entry point")
    pyz_name = pyz_names[0]
    sources = backend_sources(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    # Retain staging artifacts for inspection; the caller can remove them after acceptance.
    work = Path(tempfile.mkdtemp(prefix="origo-backend-package-", dir=output.parent))
    new_pyz, new_pkg, candidate = work / "PYZ.pyz", work / "backend.pkg", work / "voicebox-server"
    preserved_pyz, expected_pyz, old_backend = rebuild_pyz(source, reader, pyz_name, sources, new_pyz)
    preserved_pkg, replaced_pkg = rebuild_package(source, reader, entries, version, library,
                                                 pyz_name, new_pyz, sources, new_pkg)
    entitlement_result = subprocess.run(["/usr/bin/codesign", "--display", "--entitlements", "-", "--xml", str(source)],
                                       capture_output=True, check=True)
    entitlements = plistlib.loads(entitlement_result.stdout) if entitlement_result.stdout.strip() else {}
    entitlement_path = work / "entitlements.plist"
    entitlement_path.write_bytes(plistlib.dumps(entitlements))
    shutil.copyfile(source, candidate)
    candidate.chmod(0o755)
    osx.remove_signature_from_binary(str(candidate))
    with candidate.open("r+b") as stream:
        stream.truncate(reader._start_offset)
    osx.fix_exe_for_code_signing(str(candidate))
    osx.update_exe_identifier(str(candidate), str(new_pkg))
    with candidate.open("ab") as target, new_pkg.open("rb") as package:
        shutil.copyfileobj(package, target)
    osx.fix_exe_for_code_signing(str(candidate))
    osx.sign_binary(str(candidate), identity="-", entitlements_file=str(entitlement_path))
    archive_count, module_count = verify_result(candidate, entries, pyz_name, sources, preserved_pyz,
                                              expected_pyz, preserved_pkg, replaced_pkg)
    signed_entitlements = subprocess.run(
        ["/usr/bin/codesign", "--display", "--entitlements", "-", "--xml", str(candidate)],
        capture_output=True, check=True,
    )
    if plistlib.loads(signed_entitlements.stdout) != entitlements:
        raise ValueError("Signed candidate entitlements do not match the source app")
    if sha256_file(source) != original_hash:
        raise ValueError("Source sidecar changed during packaging; candidate is not accepted")
    if backend_sources(root) != sources:
        raise ValueError("Backend source changed during packaging; rerun after edits are complete")
    old_pyz = reader.open_embedded_archive(pyz_name)
    changed_backend = []
    with source.open("rb") as stream:
        for name in sorted(set(sources) & old_backend):
            _, offset, size = old_pyz.toc[name]
            stream.seek(old_pyz._start_offset + offset)
            raw = zlib.decompress(read_exact(stream, size))
            if hashlib.sha256(raw).hexdigest() != expected_pyz[name]:
                changed_backend.append(name)
    output_hash = sha256_file(candidate)
    audit = {
        "source": str(source), "source_sha256": original_hash, "output": str(output), "output_sha256": output_hash,
        "python": sys.version, "pyinstaller": PyInstaller.__version__, "architecture": "arm64",
        "source_root": str(root), "staging_directory": str(work), "self_contained": True,
        "preservation_verified": True, "codesign_verified": True, "source_snapshot_verified": True,
        "runtime_smoke_test": "not run",
        "archive_entries": archive_count, "pyz_modules": module_count,
        "preserved_archive_payloads": len(preserved_pkg), "preserved_third_party_modules": len(preserved_pyz),
        "replaced_archive_entries": sorted(replaced_pkg), "compiled_backend_modules": len(sources),
        "changed_backend_modules": changed_backend,
        "added_backend_modules": sorted(set(sources) - old_backend),
        "removed_backend_modules": sorted(old_backend - set(sources)), "entitlements": entitlements,
        "backend_source_sha256": {name: hashlib.sha256(value[2]).hexdigest() for name, value in sources.items()},
    }
    # Hard link publishes the complete candidate atomically and refuses a concurrent overwrite.
    os.link(candidate, output)
    with report.open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "report": str(report), "sha256": output_hash,
                      "backend_modules": len(sources), "preserved_modules": len(preserved_pyz)}))


if __name__ == "__main__":
    main()
