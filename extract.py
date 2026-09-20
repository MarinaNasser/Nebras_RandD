import tarfile
from pathlib import Path

source_dir = Path(r"C:\Users\Omen Max\Datasets\Colonoscopy Datasets\real colon\ZIPPED")
target_dir = Path(r"C:\Users\Omen Max\Datasets\Colonoscopy Datasets\real colon\dataset")

target_dir.mkdir(parents=True, exist_ok=True)

# Match both .tar.gz and .tgz extensions
tar_files = list(source_dir.glob("*.tar.gz")) + list(source_dir.glob("*.tgz"))
print(f"Found {len(tar_files)} archive(s) in source directory.\n")

for archive_path in tar_files:
    # Strip both extensions: '001-001_frames.tar.gz' -> '001-001_frames'
    folder_name = archive_path.name
    if folder_name.endswith(".tar.gz"):
        folder_name = folder_name[:-7]
    elif folder_name.endswith(".tgz"):
        folder_name = folder_name[:-4]

    destination_subfolder = target_dir / folder_name

    # Check if target subfolder already exists and contains files
    needs_extract = False
    if not destination_subfolder.exists():
        needs_extract = True
    elif destination_subfolder.is_dir() and not any(destination_subfolder.iterdir()):
        print(f"Folder exists but is empty: {folder_name}")
        needs_extract = True

    if needs_extract:
        destination_subfolder.mkdir(parents=True, exist_ok=True)
        print(f"Extracting: {archive_path.name} -> {destination_subfolder.name}")
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Python 3.12+ safe extraction filter; falls back cleanly on Python 3.10
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=destination_subfolder, filter="data")
                else:
                    tar.extractall(path=destination_subfolder)
            print(f"Done: {archive_path.name}\n")
        except tarfile.TarError as e:
            print(f"[Error] Corrupted or invalid tar archive {archive_path.name}: {e}\n")
        except Exception as e:
            print(f"[Error] Failed extracting {archive_path.name}: {e}\n")
    else:
        print(f"Skipping (already extracted): {archive_path.name}")

print("Verification and extraction complete.")