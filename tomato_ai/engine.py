from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: str,
    desc: str = 'Eval',
) -> Dict[str, object]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    paths: List[str] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch['image'].to(device)
        sensors = batch['sensor'].to(device)
        labels = batch.get('label')
        if labels is not None:
            labels = labels.to(device)

        logits = model(images, sensors)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        if labels is not None:
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            y_true.extend(labels.detach().cpu().tolist())

        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().numpy())
        paths.extend(batch.get('path', []))

    result: Dict[str, object] = {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'y_pred': y_pred,
        'y_prob': np.asarray(y_prob),
        'paths': paths,
    }
    if y_true:
        result['y_true'] = y_true
        result['accuracy'] = float(accuracy_score(y_true, y_pred))
        result['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return result


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    criterion,
    device: str,
    scaler=None,
    grad_clip: float | None = None,
    desc: str = 'Train',
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    use_amp = scaler is not None and device.startswith('cuda')

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch['image'].to(device)
        sensors = batch['sensor'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images, sensors)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, sensors)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'accuracy': float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if y_true else 0.0,
    }
