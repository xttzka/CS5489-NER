"""Prepare the six corpora and all shared evaluation inputs in dependency order."""

import argparse
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen


PROJECT = Path(__file__).resolve().parents[1]
CONLL_ARCHIVE = PROJECT / "data" / "raw" / "conll2003" / "conll2003.zip"
CONLL_URL = "https://data.deepai.org/conll2003.zip"
CONLL_SHA256 = "96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3"


def pos_python(override=None):
    if override is not None:
        return override
    if find_spec("spacy") and find_spec("en_core_web_sm"):
        return Path(sys.executable)
    for path in (PROJECT / ".venv-pos" / "Scripts" / "python.exe", PROJECT / ".venv-pos" / "bin" / "python"):
        if path.is_file():
            return path
    raise ValueError("install requirements-pos.txt in .venv-pos or specify --pos-python")


def run_module(module, *options, python=sys.executable):
    subprocess.run([str(python), "-B", "-m", f"data.{module}", *options], cwd=PROJECT, check=True)


def check_conll_archive(download):
    if not CONLL_ARCHIVE.is_file() and download:
        CONLL_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=CONLL_ARCHIVE.parent) as work:
            staged = Path(work) / CONLL_ARCHIVE.name
            with urlopen(CONLL_URL, timeout=60) as response:
                content = response.read()
            if sha256(content).hexdigest() != CONLL_SHA256:
                raise ValueError(f"{CONLL_URL}: unexpected ZIP checksum")
            staged.write_bytes(content)
            staged.replace(CONLL_ARCHIVE)
    if not CONLL_ARCHIVE.is_file():
        raise ValueError(f"place the source ZIP at {CONLL_ARCHIVE} or use --all --download")
    if sha256(CONLL_ARCHIVE.read_bytes()).hexdigest() != CONLL_SHA256:
        raise ValueError(f"{CONLL_ARCHIVE}: unexpected ZIP checksum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--all", action="store_true", help="Convert the sources and regenerate derived inputs")
    stage.add_argument("--derived", action="store_true", help="Use existing processed JSONL to regenerate derived inputs")
    parser.add_argument("--download", action="store_true", help="With --all, download missing source files")
    parser.add_argument("--pos-python", type=Path, help="Interpreter with the pinned spaCy model")
    args = parser.parse_args()
    if args.download and not args.all:
        parser.error("--download requires --all")

    try:
        pos = pos_python(args.pos_python)
        if args.all:
            check_conll_archive(args.download)
            if args.download:
                run_module("prepare_external_ner", "--download")
            run_module("prepare_conll2003")
            run_module("prepare_external_ner")
        run_module("make_eval_subsets")
        run_module("make_entity_index")
        run_module("add_pos_tags", python=pos)
        run_module("make_entity_replacement")
        run_module("make_perturbation_sets")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
