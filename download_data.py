"""Download the FlyWire v783 connectome files (~135 MB total) into ./data."""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

FILES = {
    # Whole-brain connectivity used by Shiu et al. 2024 (Nature)
    "Connectivity_783.parquet":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "Completeness_783.csv":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    # Cell-type labels from Schlegel et al. 2024 (Nature)
    "Supplemental_file1_neuron_annotations.tsv":
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv",
}


def progress(name):
    last = [-1]

    def hook(blocks, block_size, total):
        if total <= 0:
            return
        pct = min(100, blocks * block_size * 100 // total)
        if pct // 10 != last[0]:  # print every 10%
            last[0] = pct // 10
            sys.stdout.write(f"\r  {name}: {pct:3d}% of {total / 1e6:.0f} MB")
            sys.stdout.flush()
    return hook


def main():
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  {name}: already downloaded")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp, progress(name))
        tmp.rename(dest)
        print()
    print("Done. Data is in", DATA)


if __name__ == "__main__":
    main()
