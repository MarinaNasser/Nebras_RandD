"""ERCPMP JNET classification of segmentation-derived RGB crops."""
import argparse
import hashlib
import json
import random
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASSES = ['1', '2A', '2B', '3']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def records(dataset):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(dataset / 'ERCPMP_v5_Morphology_Pathological_Data.xlsx') as z:
        strings = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        rows = ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row', ns)[2:]
        labels = {}
        for row in rows:
            d = {''.join(filter(str.isalpha, c.attrib['r'])): strings[int(c.find('s:v', ns).text)] if c.attrib.get('t') == 's' else c.findtext('s:v', default='', namespaces=ns) for c in row}
            pid, label = d.get('A', '').strip(), d.get('L', '').strip()
            if pid in labels and labels[pid] != label:
                raise ValueError(f'Conflicting patient labels: {pid}')
            labels[pid] = label
    included, excluded = [], []
    for p in sorted((dataset / 'ERCPMP_v5_Images_Vidoes').glob('*.jpg')):
        pid = p.stem.split('_')[0]
        label = labels.get(pid)
        r = dict(path=str(p), patient=pid, label=label)
        (included if label in CLASSES else excluded).append(r)
    return included, excluded


def split_records(rows, seed):
    rng = random.Random(seed)
    assignment = {}
    for label in CLASSES:
        ids = sorted({r['patient'] for r in rows if r['label'] == label})
        if len(ids) < 3:
            raise ValueError(f'Need at least three patients for class {label}')
        rng.shuffle(ids)
        n = max(1, round(len(ids) * 0.2))
        for i, pid in enumerate(ids):
            assignment[pid] = 'test' if i < n else 'val' if i < 2*n else 'train'
    return [dict(r, split=assignment[r['patient']]) for r in rows]


def crop_boxes(mask, padding=0.15, min_area=64):
    if mask.ndim != 2:
        raise ValueError('Mask must be two-dimensional')
    h, w = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    boxes = []
    for x, y, bw, bh, area in stats[1:count]:
        if area < min_area:
            continue
        px, py = round(bw*padding), round(bh*padding)
        boxes.append((int(area), [max(0, int(x)-px), max(0, int(y)-py), min(w, int(x+bw)+px), min(h, int(y+bh)+py)]))
    return [b for _, b in sorted(boxes, reverse=True)]


class Segmenter:
    def __init__(self, checkpoint, device):
        sys.path.insert(0, str(ROOT / 'Polyps Segmentation'))
        from model import build_model
        from finetune_scope_negatives import _convert_segformer_state_dict_layout, _extract_state_dict
        self.device = device
        self.model = build_model('segformer', pretrained=False, segformer_size='b3').to(device).eval()
        state = _extract_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.model.load_state_dict(_convert_segformer_state_dict_layout(state, self.model))

    @torch.inference_mode()
    def __call__(self, rgb):
        x = cv2.resize(rgb, (512, 512)).astype(np.float32)/255
        x = (x-np.array(MEAN, dtype=np.float32))/np.array(STD, dtype=np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            p = self.model(x).sigmoid()[0, 0].float().cpu().numpy()
        return cv2.resize(p, (rgb.shape[1], rgb.shape[0])) > 0.5


def transform(train=False):
    ops = [T.Resize((224, 224))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15), T.ColorJitter(.1, .1, .1, .02)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, rows, train=False):
        self.rows, self.tf = rows, transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r['crop']) as im:
            x = self.tf(im.convert('RGB'))
        return x, CLASSES.index(r['label'])


def classifier(pretrained=False):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
    return m


@torch.inference_mode()
def evaluate(model, rows, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(Crops(rows), batch_size=16):
        probs.extend(model(x.to(device)).softmax(1).cpu().tolist())
    patients = {}
    for r, p in zip(rows, probs):
        patients.setdefault(r['patient'], []).append((CLASSES.index(r['label']), p))
    y = [v[0][0] for v in patients.values()]
    pred = [int(np.mean([p for _, p in v], axis=0).argmax()) for v in patients.values()]
    report = classification_report(y, pred, labels=list(range(4)), target_names=CLASSES, zero_division=0, output_dict=True)
    return dict(patient_report=report, confusion_matrix=confusion_matrix(y, pred, labels=list(range(4))).tolist(), patients=len(y), images=len(rows), predictions=[dict(patient=r['patient'], path=r['path'], label=r['label'], probabilities=p) for r, p in zip(rows, probs)])


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists():
        raise FileExistsError('Output already contains a trained model; choose a new --output')
    rows, excluded = records(args.dataset)
    rows = split_records(rows, args.seed)
    write_json(out / 'excluded.json', excluded)
    seg = Segmenter(args.seg_checkpoint, device)
    signature = hashlib.sha256(args.seg_checkpoint.read_bytes()).hexdigest()
    eligible = []
    for i, r in enumerate(rows):
        rgb = np.array(Image.open(r['path']).convert('RGB'))
        boxes = crop_boxes(seg(rgb))
        r['box'] = boxes[0] if boxes else None
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            crop = out / 'crops' / (Path(r['path']).stem + '.png')
            crop.parent.mkdir(exist_ok=True)
            Image.fromarray(rgb[y1:y2, x1:x2]).save(crop)
            r['crop'] = str(crop.resolve())
            eligible.append(r)
        if (i+1) % 25 == 0:
            print(f'Segmented {i+1}/{len(rows)}', flush=True)
    write_json(out / 'manifest.json', rows)
    del seg
    if device.type == 'cuda': torch.cuda.empty_cache()
    splits = {s: [r for r in eligible if r['split'] == s] for s in ['train', 'val', 'test']}
    summary = {s: dict(images=len(rr), patients=len({r['patient'] for r in rr}), classes=dict(Counter(r['label'] for r in rr))) for s, rr in splits.items()}
    write_json(out / 'data_summary.json', dict(splits=summary, excluded_labels=len(excluded), missed_segmentation=len(rows)-len(eligible)))
    print(summary, flush=True)
    for s, rr in splits.items():
        if {r['label'] for r in rr} != set(CLASSES): raise ValueError(f'Missing class after segmentation: {s}')
    m = classifier(True).to(device)
    train_rows = splits['train']
    images_per_patient = Counter(r['patient'] for r in train_rows)
    patients_per_class = Counter({c: len({r['patient'] for r in train_rows if r['label'] == c}) for c in CLASSES})
    weights = [1/(images_per_patient[r['patient']]*patients_per_class[r['label']]) for r in train_rows]
    loader = DataLoader(Crops(train_rows, True), batch_size=16, sampler=WeightedRandomSampler(weights, len(weights), replacement=True))
    opt = torch.optim.AdamW([{'params': [p for n,p in m.named_parameters() if not n.startswith('fc.')], 'lr': 1e-5}, {'params': m.fc.parameters(), 'lr': 3e-4}], weight_decay=1e-4)
    history, best, stale = [], -1, 0
    for epoch in range(args.epochs):
        m.train()
        # Small dataset: preserve ImageNet batch statistics throughout fine-tuning.
        for module in m.modules():
            if isinstance(module, torch.nn.BatchNorm2d): module.eval()
        for n,p in m.named_parameters(): p.requires_grad = epoch >= 3 or n.startswith('fc.')
        losses = []
        for x,y in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(m(x.to(device)), y.to(device))
            loss.backward(); opt.step(); losses.append(loss.item())
        val = evaluate(m, splits['val'], device)
        score = val['patient_report']['macro avg']['f1-score']
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), val_patient_macro_f1=score))
        write_json(out / 'history.json', history)
        print(history[-1], flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(dict(state_dict=m.state_dict(), classes=CLASSES, architecture='resnet18', epoch=epoch+1, segmentation_sha256=signature, segmentation_checkpoint=str(args.seg_checkpoint.resolve()), image_size=224, seed=args.seed), out / 'best.pt')
        else: stale += 1
        if stale >= args.patience: break
    m.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True)['state_dict'])
    test = evaluate(m, splits['test'], device)
    test['coverage'] = {s: dict(eligible_images=len(splits[s]), total_images=sum(r['split']==s for r in rows)) for s in splits}
    test['majority_baseline_class'] = Counter(r['label'] for r in {r['patient']:r for r in train_rows}.values()).most_common(1)[0][0]
    test['majority_baseline_patient_accuracy'] = float(np.mean([r['label']==test['majority_baseline_class'] for r in {r['patient']:r for r in splits['test']}.values()]))
    write_json(out / 'test_results.json', test)
    print(json.dumps(test['patient_report'], indent=2), flush=True)


class Characterizer:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if ckpt['classes'] != CLASSES: raise ValueError('Unsupported class mapping')
        self.model = classifier().to(self.device).eval()
        self.model.load_state_dict(ckpt['state_dict'])
        self.tf = transform()

    @torch.inference_mode()
    def predict(self, rgb, mask):
        if rgb.shape[:2] != mask.shape: raise ValueError('Image and mask dimensions differ')
        results = []
        for box in crop_boxes(mask):
            x1,y1,x2,y2 = box
            x = self.tf(Image.fromarray(rgb[y1:y2,x1:x2])).unsqueeze(0).to(self.device)
            p = self.model(x).softmax(1)[0].cpu().tolist()
            results.append(dict(box=box, jnet=CLASSES[int(np.argmax(p))], probabilities=dict(zip(CLASSES,p))))
        return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'predict'])
    p.add_argument('--dataset', type=Path, default=Path(r'C:\Users\Omen Max\Datasets\Colonoscopy Datasets\ERCPMP'))
    p.add_argument('--seg-checkpoint', type=Path, default=ROOT / 'Polyps Segmentation/checkpoints/best_model.pt')
    p.add_argument('--output', type=Path, default=HERE / 'outputs/ercpmp_jnet')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--mask', type=Path)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.command == 'train': train(args)
    else:
        if args.image is None: p.error('predict requires --image')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rgb = np.array(Image.open(args.image).convert('RGB'))
        mask = np.array(Image.open(args.mask).convert('L')) > 0 if args.mask else Segmenter(args.seg_checkpoint, device)(rgb)
        result = Characterizer(args.checkpoint or args.output / 'best.pt', device).predict(rgb, mask)
        print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
