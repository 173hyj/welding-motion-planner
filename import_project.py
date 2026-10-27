"""Import the original ZIP/PSZ without rewriting source engineering data."""
import argparse
import codecs
import hashlib
import json
import pathlib
import re
import zipfile
import xml.etree.ElementTree as ET


def decoded_name(name, flags):
    if not flags & 0x800:
        try:
            return name.encode("cp437").decode("gbk")
        except UnicodeError:
            pass
    return name


def safe_name(name):
    name = name.replace("\\", "/")
    if name.startswith("/") or ":" in name or ".." in pathlib.PurePosixPath(name).parts:
        raise ValueError("Unsafe archive path: " + name)
    return name


def xml_root(data):
    header = data[:200].decode("ascii", errors="ignore")
    found = re.search(r'encoding=[\"\']([^\"\']+)', header)
    encoding = found.group(1) if found else "utf-8-sig"
    codecs.lookup(encoding)
    return ET.fromstring(data.decode(encoding))


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    result = []
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = safe_name(decoded_name(entry.filename, entry.flag_bits))
            path = destination / name
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with source.open(entry) as incoming, path.open("wb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    checksum.update(chunk)
                    outgoing.write(chunk)
            result.append({"path": name, "size": entry.file_size, "sha256": checksum.hexdigest()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"archive": str(args.archive.resolve()), "outer": extract(args.archive, args.output / "source/outer")}
    archives = list((args.output / "source/outer").rglob("*.psz"))
    if len(archives) != 1:
        raise ValueError("Expected exactly one embedded PSZ")
    manifest["psz"] = extract(archives[0], args.output / "source/psz")
    with (args.output / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    print(json.dumps({"outer_files": len(manifest["outer"]), "psz_files": len(manifest["psz"])}, indent=2))


if __name__ == "__main__":
    main()
