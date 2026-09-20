r"""REAL-Colon lesion characterization (second-stage classifier after segmentation/detection).

Usage (PowerShell -- backtick line continuation, one line also works):
    python realcolon_characterize.py `
        --dataset "D:/path/to/real-colon" `
        --lesion-info lesion_info.csv `
        --video-info video_info.csv `
        --output outputs/realcolon_characterization

Expects, under --dataset, either extracted folders or zip files named:
    {SSS-VVV}_frames        (or {SSS-VVV}_frames.zip)
    {SSS-VVV}_annotation    (or {SSS-VVV}_annotation.zip)
Only videos where BOTH halves are found are used; everything else is reported
and skipped, so partial downloads don't silently corrupt the dataset.
"""
import argparse
import csv
import io
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MIN_VIDEOS_PER_CLASS = 3   # below this a class can't be validated/tested meaningfully
FRAMES_PER_LESION = 4      # sampled evenly across the lesion's visible span
CROP_PADDING = 0.15


# --------------------------------------------------------------------------
# Discovery: figure out which videos we actually have both halves for
# --------------------------------------------------------------------------
class VideoSource:
    """Transparent access to a video's frames+annotation, whether extracted or zipped."""

    def __init__(self, dataset_root: Path, video_name: str):
        self.video_name = video_name
        self.frames_dir = self._first_existing(dataset_root, video_name, 'frames', dir=True)
        self.ann_dir = self._first_existing(dataset_root, video_name, ('annotation', 'annotations'), dir=True)
        self.frames_zip = self._first_existing(dataset_root, video_name, 'frames', dir=False)
        self.ann_zip = self._first_existing(dataset_root, video_name, ('annotation', 'annotations'), dir=False)
        self._fz = zipfile.ZipFile(self.frames_zip) if self.frames_zip else None
        self._az = zipfile.ZipFile(self.ann_zip) if self.ann_zip else None

    @staticmethod
    def _first_existing(dataset_root, video_name, suffixes, dir):
        """Try each candidate suffix (e.g. 'annotation' and 'annotations') and return
        the first path that actually exists as a dir (dir=True) or a .zip file (dir=False)."""
        if isinstance(suffixes, str):
            suffixes = (suffixes,)
        for suf in suffixes:
            if dir:
                p = dataset_root / f'{video_name}_{suf}'
                if p.is_dir():
                    return p
            else:
                p = dataset_root / f'{video_name}_{suf}.zip'
                if p.exists():
                    return p
        return None

    @property
    def available(self):
        frames_ok = self.frames_dir is not None or self._fz is not None
        ann_ok = self.ann_dir is not None or self._az is not None
        return frames_ok and ann_ok

    def list_annotations(self):
        if self.ann_dir is not None:
            return sorted(p.name for p in self.ann_dir.glob('*.xml'))
        return sorted(n for n in self._az.namelist() if n.endswith('.xml'))

    def read_annotation(self, name):
        if self.ann_dir is not None:
            return (self.ann_dir / name).read_bytes()
        return self._az.read(name)

    def read_frame(self, filename):
        if self.frames_dir is not None:
            data = (self.frames_dir / filename).read_bytes()
        else:
            data = self._fz.read(filename)
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def discover_videos(dataset_root: Path, all_video_names):
    available, missing = [], []
    for name in all_video_names:
        src = VideoSource(dataset_root, name)
        (available if src.available else missing).append(name)
    return available, missing


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_lesion_labels(lesion_info_path):
    """unique_object_id -> histology_class"""
    return {row['unique_object_id']: row['histology_class'] for row in load_csv(lesion_info_path)}


# --------------------------------------------------------------------------
# Annotation parsing + crop extraction
# --------------------------------------------------------------------------
def frame_number(xml_filename):
    # SSS-VVV_t.xml -> t
    stem = Path(xml_filename).stem
    return int(stem.split('_')[-1])


def parse_frame_objects(xml_bytes):
    root = ET.fromstring(xml_bytes)
    filename = root.findtext('filename')
    objs = []
    for obj in root.findall('object'):
        if obj.findtext('name') != 'lesion':
            continue
        uid = obj.findtext('unique_id')
        box = obj.find('bndbox')
        bbox = [int(float(box.findtext(k))) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        objs.append(dict(unique_id=uid, bbox=bbox))
    return filename, objs


def collect_lesion_frames(src: VideoSource):
    """lesion_unique_id -> list of (frame_filename, bbox), sorted by frame number"""
    per_lesion = defaultdict(list)
    for xml_name in src.list_annotations():
        try:
            fnum = frame_number(xml_name)
        except ValueError:
            continue
        filename, objs = parse_frame_objects(src.read_annotation(xml_name))
        for o in objs:
            per_lesion[o['unique_id']].append((fnum, filename, o['bbox']))
    for uid in per_lesion:
        per_lesion[uid].sort(key=lambda t: t[0])
    return per_lesion


def sample_indices(n, k):
    if n <= k:
        return list(range(n))
    return sorted(set(int(round(i)) for i in np.linspace(0, n - 1, k)))


def pad_box(bbox, w, h, padding=CROP_PADDING):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = round(bw * padding), round(bh * padding)
    return [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]


def extract_crops(dataset_root, videos, lesion_labels, out_dir, frames_per_lesion=FRAMES_PER_LESION):
    """Returns list of dicts: path, unique_object_id, video, class"""
    records = []
    for video in videos:
        src = VideoSource(dataset_root, video)
        per_lesion = collect_lesion_frames(src)
        for uid, entries in per_lesion.items():
            label = lesion_labels.get(uid)
            if label is None:
                continue  # box present but not in lesion_info (shouldn't normally happen)
            idxs = sample_indices(len(entries), frames_per_lesion)
            for i in idxs:
                fnum, filename, bbox = entries[i]
                rgb = src.read_frame(filename)
                if rgb is None:
                    continue
                h, w = rgb.shape[:2]
                x1, y1, x2, y2 = pad_box(bbox, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = rgb[y1:y2, x1:x2]
                class_dir = out_dir / 'crops' / label.replace(' ', '_')
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_path = class_dir / f'{uid}__f{fnum}.jpg'
                Image.fromarray(crop).save(crop_path, quality=95)
                records.append(dict(path=str(crop_path), unique_object_id=uid, video=video, label=label))
        print(f'{video}: {len(per_lesion)} lesion(s) with boxes found', flush=True)
    return records


# --------------------------------------------------------------------------
# Class sample picture
# --------------------------------------------------------------------------
def save_class_samples(records, out_path, classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_class = defaultdict(list)
    for r in records:
        by_class[r['label']].append(r['path'])

    present = [c for c in classes if by_class.get(c)]
    if not present:
        print('No classes have any extracted crops; skipping sample picture.')
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    for ax, cls in zip(axes, present):
        img = Image.open(by_class[cls][0])
        ax.imshow(img)
        ax.set_title(f'{cls}\n(n={len(by_class[cls])})')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved class sample picture to {out_path}')


# --------------------------------------------------------------------------
# Splits, dataset, model, training (mirrors the ERCPMP JNET pipeline pattern)
# --------------------------------------------------------------------------
def video_level_split(records, classes, seed):
    """Keep videos together and require every observed class in all three splits.

    Class coverage takes priority over the approximate 60/20/20 video ratio.
    Ignore metadata classes without crops; raise if an observed class cannot
    be represented in every split.
    """
    rng = random.Random(seed)
    video_classes = defaultdict(set)
    for r in records:
        if r['label'] in classes:
            video_classes[r['video']].add(r['label'])

    class_video_counts = Counter()
    for vids in video_classes.values():
        for c in vids:
            class_video_counts[c] += 1
    absent = [c for c in classes if class_video_counts[c] == 0]
    if absent:
        print(f'Excluding metadata classes with no extracted crops: {absent}')
    classes = [c for c in classes if class_video_counts[c] > 0]
    if not classes:
        raise ValueError('No extracted crops match the requested classes; cannot create splits.')
    insufficient = {c: class_video_counts[c] for c in classes
                    if class_video_counts[c] < MIN_VIDEOS_PER_CLASS}
    if insufficient:
        raise ValueError('Every class needs at least 3 distinct videos with extracted crops '
                         'for train/val/test coverage without video leakage. '
                         f'Insufficient video counts: {insufficient}')

    videos = sorted(video_classes)
    rng.shuffle(videos)
    n = max(1, round(len(videos) * 0.2))
    split_names = ('train', 'val', 'test')
    targets = dict(train=len(videos) - 2 * n, val=n, test=n)
    members = {s: set() for s in split_names}
    assignment = {}
    failed = set()

    def cover_classes():
        # Backtracking handles videos that carry multiple histology classes.
        state = tuple(frozenset(members[s]) for s in split_names)
        if state in failed:
            return False
        unmet = []
        for s in split_names:
            covered = set().union(*(video_classes[v] for v in members[s]))
            for c in classes:
                if c not in covered:
                    candidates = [v for v in videos
                                  if v not in assignment and c in video_classes[v]]
                    if not candidates:
                        failed.add(state)
                        return False
                    unmet.append((s, c, candidates))
        if not unmet:
            return True
        s, _, candidates = min(unmet, key=lambda item: len(item[2]))
        needed = {c for split, c, _ in unmet if split == s}
        candidates.sort(key=lambda v: -len(video_classes[v] & needed))
        for v in candidates:
            assignment[v] = s
            members[s].add(v)
            if cover_classes():
                return True
            members[s].remove(v)
            del assignment[v]
        failed.add(state)
        return False

    if not cover_classes():
        raise ValueError('Cannot place every class in train/val/test while keeping videos '
                         'together: the overlap of classes across videos prevents coverage.')
    for v in videos:
        if v not in assignment:
            s = max(split_names, key=lambda s: targets[s] - len(members[s]))
            assignment[v] = s
            members[s].add(v)

    out = [dict(r, split=assignment[r['video']]) for r in records
           if r['video'] in assignment and r['label'] in classes]
    return out, list(classes)


def transform(train=False):
    ops = [T.Resize((224, 224))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15), T.ColorJitter(.1, .1, .1, .02)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, rows, classes, train=False):
        self.rows, self.classes, self.tf = rows, classes, transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['path']) as im:
            x = self.tf(im.convert('RGB'))
        return x, self.classes.index(r['label'])


def classifier(n_classes, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.inference_mode()
def evaluate(model, rows, classes, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(Crops(rows, classes), batch_size=16):
        probs.extend(model(x.to(device)).softmax(1).cpu().tolist())
    return summarize_predictions(rows, probs, classes)


def summarize_predictions(rows, probs, classes):
    """Score crops and equal-weight probability averages for each distinct lesion."""
    if len(rows) != len(probs):
        raise ValueError('Each crop must have exactly one prediction.')

    def metrics(items):
        y = [classes.index(item['label']) for item in items]
        pred = [int(np.argmax(item['probabilities'])) for item in items]
        return dict(report=classification_report(
            y, pred, labels=list(range(len(classes))), target_names=classes,
            zero_division=0, output_dict=True),
            confusion_matrix=confusion_matrix(
                y, pred, labels=list(range(len(classes)))).tolist())

    crops = [dict(path=r['path'], unique_object_id=r['unique_object_id'],
                  video=r['video'], label=r['label'], probabilities=p)
             for r, p in zip(rows, probs)]
    grouped = defaultdict(list)
    for crop in crops:
        grouped[crop['unique_object_id']].append(crop)
    lesions = []
    for uid, items in sorted(grouped.items()):
        if len({(item['video'], item['label']) for item in items}) != 1:
            raise ValueError(f'Conflicting video or label for lesion {uid}')
        lesions.append(dict(unique_object_id=uid, video=items[0]['video'],
                            label=items[0]['label'], n_crops=len(items),
                            probabilities=np.mean([item['probabilities'] for item in items], axis=0).tolist()))
    return dict(crop=metrics(crops), lesion=metrics(lesions), predictions=lesions)


def train(records, classes, out_dir, epochs, patience, seed, device):
    rows, classes = video_level_split(records, classes, seed)
    if len(classes) < 2:
        print('Fewer than 2 classes have enough video diversity to train on. Stopping before training.')
        return
    splits = {s: [r for r in rows if r['split'] == s] for s in ['train', 'val', 'test']}
    summary = {s: dict(images=len(rr), lesions=len({r['unique_object_id'] for r in rr}),
                        videos=len({r['video'] for r in rr}),
                        classes=dict(Counter(r['label'] for r in rr))) for s, rr in splits.items()}
    print(json.dumps(summary, indent=2))
    (out_dir / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    (out_dir / 'split_records.json').write_text(json.dumps(rows, indent=2))
    for s, rr in splits.items():
        missing = set(classes) - {r['label'] for r in rr}
        if missing:
            raise ValueError(f'Split "{s}" is missing classes: {missing}')

    device = torch.device(device)
    m = classifier(len(classes)).to(device)
    train_rows = splits['train']
    if not train_rows:
        print('No training examples after the video-level split. Stopping.')
        return
    label_counts = Counter(r['label'] for r in train_rows)
    weights = [1.0 / label_counts[r['label']] for r in train_rows]
    loader = DataLoader(Crops(train_rows, classes, True), batch_size=16,
                         sampler=WeightedRandomSampler(weights, len(weights), replacement=True))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)

    history, best, stale = [], -1, 0
    for epoch in range(epochs):
        m.train()
        losses = []
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(m(x.to(device)), y.to(device))
            loss.backward(); opt.step(); losses.append(loss.item())
        val = evaluate(m, splits['val'], classes, device) if splits['val'] else None
        score = val['lesion']['report']['macro avg']['f1-score']
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)),
                            val_lesion_macro_f1=score,
                            val_crop_macro_f1=val['crop']['report']['macro avg']['f1-score']))
        print(history[-1], flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(dict(state_dict=m.state_dict(), classes=classes, architecture='resnet18'),
                       out_dir / 'best.pt')
        else:
            stale += 1
        if stale >= patience:
            break

    if (out_dir / 'best.pt').exists():
        m.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device, weights_only=True)['state_dict'])
    if splits['test']:
        test = evaluate(m, splits['test'], classes, device)
        (out_dir / 'test_results.json').write_text(json.dumps(test, indent=2))
        print(json.dumps(dict(crop=test['crop']['report'], lesion=test['lesion']['report']), indent=2))
    else:
        print('No test videos available for these classes; skipping held-out evaluation.')


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True, help='root folder containing the frame/annotation dirs or zips')
    ap.add_argument('--lesion-info', type=Path, default='lesion_info.csv')
    ap.add_argument('--video-info', type=Path, default='video_info.csv')
    ap.add_argument('--output', type=Path, default=Path('outputs/realcolon_characterization'))
    ap.add_argument('--frames-per-lesion', type=int, default=FRAMES_PER_LESION)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-training', action='store_true', help='only extract crops + sample picture, no model training')
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    video_rows = load_csv(args.video_info)
    all_video_names = [v['unique_video_name'] for v in video_rows]
    lesion_labels = load_lesion_labels(args.lesion_info)
    all_classes = sorted(set(lesion_labels.values()))

    available, missing = discover_videos(args.dataset, all_video_names)
    print(f'Found both frames+annotation for {len(available)}/{len(all_video_names)} videos.')
    if missing:
        print(f'Missing/incomplete: {missing}')
    (args.output / 'available_videos.json').write_text(json.dumps(dict(available=available, missing=missing), indent=2))

    if not available:
        print('No usable videos found under --dataset. Check the folder naming (expects "{SSS-VVV}_frames" '
              'and "{SSS-VVV}_annotation", extracted or as .zip). Stopping.')
        return

    records = extract_crops(args.dataset, available, lesion_labels, args.output, args.frames_per_lesion)
    (args.output / 'records.json').write_text(json.dumps(records, indent=2))
    print(f'Extracted {len(records)} crops across {len({r["unique_object_id"] for r in records})} lesions '
          f'from {len({r["video"] for r in records})} videos.')
    print('Class counts (crops):', dict(Counter(r['label'] for r in records)))
    print('Class counts (distinct lesions):',
          dict(Counter(l for l in {r["unique_object_id"]: r["label"] for r in records}.values())))

    save_class_samples(records, args.output / 'class_samples.png', all_classes)

    if not args.skip_training:
        train(records, all_classes, args.output, args.epochs, args.patience, args.seed, device)


if __name__ == '__main__':
    main()
