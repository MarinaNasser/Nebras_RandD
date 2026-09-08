"""Export the raw Kvasir-VQA split to metadata.csv and one JPEG per img_id."""
import argparse
import json
import os
from pathlib import Path


def download_dataset(d_path):
    d_path = Path(d_path).expanduser().resolve()
    d_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(d_path / ".hf_cache"))
    from datasets import load_dataset
    from PIL import Image
    from tqdm.auto import tqdm

    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", cache_dir=str(d_path / ".hf_cache" / "datasets"))
    df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
    if df.isna().any().any():
        raise ValueError("Dataset contains missing metadata values")
    # Keep original row indices for indexing the raw split, including on older pandas.
    unique = df.drop_duplicates('img_id', keep='first')
    for img_id in unique.img_id:
        if not isinstance(img_id, str) or Path(img_id).name != img_id or any(c in img_id for c in '/\\:'):
            raise ValueError(f"Unsafe image ID: {img_id!r}")
    metadata_tmp = d_path / 'metadata.csv.tmp'
    df.to_csv(metadata_tmp, index=False)
    metadata_tmp.replace(d_path / 'metadata.csv')
    (d_path / 'images').mkdir(exist_ok=True)
    for i, row in tqdm(unique.iterrows(), total=len(unique), desc='Exporting images'):
        destination = d_path / 'images' / f"{row['img_id']}.jpg"
        if destination.exists():
            try:
                with Image.open(destination) as image:
                    image.verify()
                continue
            except (OSError, SyntaxError):
                pass
        temporary = destination.with_suffix('.jpg.tmp')
        ds['raw'][int(i)]['image'].convert('RGB').save(temporary, format='JPEG')
        temporary.replace(destination)
    summary = {'dataset': 'SimulaMet-HOST/Kvasir-VQA', 'split': 'raw',
               'rows': len(df), 'unique_images': len(unique), 'directory': str(d_path)}
    (d_path / 'download_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=r'C:\Users\Omen Max\Datasets\Kvasir-VQA')
    download_dataset(parser.parse_args().output)
