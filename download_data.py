"""Download a CICIDS dataset archive supplied by the user via environment variable.

The official dataset landing page is documented in data/README.md. Set
CICIDS_DOWNLOAD_URL to a direct archive URL that you are authorized to download.
This avoids hard-coding a third-party mirror or committing a >50 MB dataset.
"""
import os, urllib.request, zipfile, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
url=os.getenv("CICIDS_DOWNLOAD_URL")
if not url:
    raise SystemExit(
        "Set CICIDS_DOWNLOAD_URL to a direct CICIDS2017 MachineLearningCVE archive URL. "
        "See data/README.md for the official dataset source."
    )
archive=DATA/"cicids_dataset_download"
DATA.mkdir(exist_ok=True)
print("Downloading dataset archive...")
urllib.request.urlretrieve(url, archive)
print(f"Downloaded {archive.stat().st_size/1024/1024:.1f} MB")
if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(DATA/member.filename).resolve()
            if DATA.resolve() not in target.parents and target != DATA.resolve():
                raise RuntimeError("Unsafe archive path detected.")
        z.extractall(DATA)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            target=(DATA/member.name).resolve()
            if DATA.resolve() not in target.parents and target != DATA.resolve():
                raise RuntimeError("Unsafe archive path detected.")
        t.extractall(DATA, filter="data")
else:
    raise RuntimeError("Downloaded file is not a ZIP or TAR archive.")
archive.unlink(missing_ok=True)
print("Dataset extracted. Move the MachineLearningCVE CSV files directly into data/ if nested.")
