"""Check the standalone source tree, evidence hashes and generated documentation."""

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
ROOT_EXCLUDED = {"build", "dist", "models", "cache", "runs", "data", "vendor"}


def files():
    for p in ROOT.rglob("*"):
        if (
            p.is_file()
            and p.name != "FILE_SHA256.json"
            and p.relative_to(ROOT).parts[0] not in ROOT_EXCLUDED
            and not any(
                x in EXCLUDED or x.endswith(".egg-info") or x.startswith((".venv", ".env"))
                for x in p.relative_to(ROOT).parts
            )
        ):
            yield p


def main():
    seen = list(files())
    required = {
        "src/kaloscope_dinov3/models/" + name
        for name in ("__init__.py", "vision_transformer.py", "convnext.py")
    }
    if not required <= {str(p.relative_to(ROOT)) for p in seen}:
        raise AssertionError("Release is missing DINOv3 model implementation files")
    for p in seen:
        if p.suffix == ".py":
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                if module and (module.startswith("research.") or module.startswith("integrations.")):
                    raise AssertionError(f"Original workspace dependency: {p}: {module}")
        if p.suffix == ".json":
            json.loads(p.read_text())
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                json.loads(line)
        if p.suffix in (".pt", ".pth", ".bin", ".safetensors", ".png", ".jpg"):
            raise AssertionError(f"Unexpected model/data artifact: {p}")
        if p.stat().st_size > 10 * 1024 * 1024:
            raise AssertionError(f"Oversized artifact: {p}")
        if p.suffix in (".py", ".json", ".toml") and (
            "src" in p.relative_to(ROOT).parts or "configs" in p.relative_to(ROOT).parts
        ):
            if re.search(r"/(?:mnt|home|root)/", p.read_text()):
                raise AssertionError(f"Hardcoded machine path: {p}")
        if p.suffix == ".md":
            # Ignore fenced examples and external URLs; local documentation links must resolve.
            text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
            for href in re.findall(r"\]\(([^)]+)\)", text):
                href = href.split("#")[0]
                if not href or "://" in href or href.startswith("mailto:"):
                    continue
                if not (p.parent / href).exists():
                    raise AssertionError(f"Broken local link: {p.relative_to(ROOT)} -> {href}")
    for item in json.loads((ROOT / "evidence/provenance.json").read_text()):
        if "packaged_sha256" in item:
            actual = hashlib.sha256((ROOT / item["artifact"]).read_bytes()).hexdigest()
            if actual != item["packaged_sha256"]:
                raise AssertionError(f"Evidence modified: {item['artifact']}")
    spec = importlib.util.spec_from_file_location("render_evidence", ROOT / "scripts/render_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text, _ = module.render()
    if text != (ROOT / "docs/Z3_TABLES.md").read_text():
        raise AssertionError("Regenerate Z3_TABLES.md from evidence")
    print(
        json.dumps(
            {"status": "passed", "files": len(seen), "bytes": sum(p.stat().st_size for p in seen)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
