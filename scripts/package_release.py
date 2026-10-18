"""Make a source-only GitHub upload ZIP with per-file SHA256; no runtime assets."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from verify_release import ROOT, files, main as verify

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Write the archive outside the source tree")
    verify()
    selected = sorted(files())
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in selected:
            info = zipfile.ZipInfo(
                "kaloscope-style-adapter/" + str(f.relative_to(ROOT)), date_time=(2026, 10, 17, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, f.read_bytes())
        info = zipfile.ZipInfo("kaloscope-style-adapter/FILE_SHA256.json", date_time=(2026, 10, 17, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        z.writestr(info, json.dumps(hashes, indent=2) + "\n")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"{output.name}: {output.stat().st_size} bytes, SHA256 {checksum}")
