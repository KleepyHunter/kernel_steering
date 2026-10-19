import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_and_extract_caa(repo_ref: str, output_dir: Path) -> None:
    url = f"https://codeload.github.com/nrimsky/CAA/zip/{repo_ref}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "caa.zip"
        urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        extracted_root = next(Path(tmp_dir).glob("CAA-*"), None)
        if extracted_root is None:
            raise RuntimeError("Could not find extracted CAA root directory.")

        source_datasets = extracted_root / "datasets"
        if not source_datasets.exists():
            raise RuntimeError("Extracted archive does not contain datasets directory.")

        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["generate", "test", "raw"]:
            src = source_datasets / split
            if not src.exists():
                continue
            dst = output_dir / split
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Download CAA datasets into local dataset/caa directory.")
    parser.add_argument("--ref", type=str, default="refs/heads/main", help="Git ref for codeload URL, e.g. refs/heads/main or <commit_sha>")
    parser.add_argument("--output", type=str, default="dataset/caa", help="Output directory for extracted datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    download_and_extract_caa(repo_ref=args.ref, output_dir=output_dir)
    print(f"CAA datasets downloaded to {output_dir}")


if __name__ == "__main__":
    main()
