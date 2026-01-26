# Old Evaluation System Report

## Overview

This document describes the **current evaluation metrics** implemented in the ReqFlow project for assessing span annotation quality on software requirements.

---

## Evaluation Modes

The project supports **two matching modes**:

| Mode | Description |
|------|-------------|
| **Exact Match** | Span is correct only if `tag`, `start`, and `end` match exactly |
| **Relaxed Match** | Uses IoU + text similarity scoring with a configurable threshold |

---

## Core Metrics

### Precision, Recall, F1

```
Precision = TP / (TP + FP)
    "Of all spans I predicted, how many are correct?"

Recall = TP / (TP + FN)
    "Of all spans that exist in gold, how many did I find?"

F1 = 2 × Precision × Recall / (Precision + Recall)
    "Harmonic mean of precision and recall"
```

### Definitions

| Term | Definition |
|------|------------|
| **TP (True Positive)** | Predicted span matches a gold span |
| **FP (False Positive)** | Predicted span has no matching gold span |
| **FN (False Negative)** | Gold span has no matching prediction |

---

## Exact Match Implementation

From `src/tools/evaluate.py`:

```python
def to_set(spans: Any):
    s = set()
    for sp in spans or []:
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        if tag in TAGS and isinstance(st, int) and isinstance(en, int):
            s.add((tag, st, en))
    return s
```

**TP/FP/FN Detection:**

```python
ps = to_set(pred[rid].get("spans"))   # Predicted spans as set
gs = to_set(gold[rid].get("spans"))   # Gold spans as set

for tag in TAGS:
    p_tag = {x for x in ps if x[0] == tag}
    g_tag = {x for x in gs if x[0] == tag}
    
    TP = len(p_tag & g_tag)   # Set intersection
    FP = len(p_tag - g_tag)   # In pred but not gold
    FN = len(g_tag - p_tag)   # In gold but not pred
```

**Example:**
```
Gold spans:    {(Main_actor, 0, 8), (Action, 20, 30)}
Pred spans:    {(Main_actor, 0, 8), (Action, 22, 30)}

Main_actor: TP=1, FP=0, FN=0  (exact match)
Action:     TP=0, FP=1, FN=1  (start differs: 22 ≠ 20)
```

---

## Relaxed Match Implementation

From `reqflow.py`:

### Step 1: Span Score Calculation

```python
def span_score(a: Span, b: Span, mode: str) -> float:
    if a.tag != b.tag:
        return 0.0
    
    if mode == "Exact":
        return 1.0 if (a.start == b.start and a.end == b.end) else 0.0
    
    # Relaxed mode
    s_iou = iou(a, b)
    if s_iou <= 0:
        return 0.0
    s_txt = text_sim(a, b)
    
    return 0.65 * s_iou + 0.35 * s_txt
```

**Score Formula:**
```
Score = 0.65 × IoU + 0.35 × TextSimilarity
```

### Step 2: IoU (Intersection over Union)

```python
def iou(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    
    la = max(0, a.end - a.start)
    lb = max(0, b.end - b.start)
    union = la + lb - inter
    
    return (inter / union) if union > 0 else 0.0
```

**IoU Example:**
```
Gold:  [10, 26]  "the system shall"  (16 chars)
Pred:  [14, 26]  "system shall"      (12 chars)

Overlap = min(26,26) - max(10,14) = 26 - 14 = 12 chars
Union = 16 + 12 - 12 = 16 chars
IoU = 12/16 = 0.75
```

### Step 3: Text Similarity

```python
def text_sim(a: Span, b: Span) -> float:
    return SequenceMatcher(None, a.text, b.text).ratio()
```

Uses Python's `difflib.SequenceMatcher` for string similarity (0.0 to 1.0).

### Step 4: Greedy Matching Algorithm

```python
def match_counts(pred: List[Span], gold: List[Span], mode: str, threshold: float):
    # Build candidate matches
    cands = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            sc = span_score(p, g, mode=mode)
            if sc >= threshold:
                cands.append((sc, i, j))
    
    # Sort by score descending (greedy)
    cands.sort(key=lambda x: x[0], reverse=True)
    
    # One-to-one matching
    used_p = set()
    used_g = set()
    tp = 0
    
    for _sc, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    
    fp = len(pred) - tp
    fn = len(gold) - tp
    
    return tp, fp, fn
```

**Matching Logic:**
1. Calculate scores for all pred-gold pairs where `tag` matches
2. Filter pairs where `score >= threshold`
3. Sort by score (highest first)
4. Greedily assign matches (each span matched at most once)
5. Unmatched predictions → FP, Unmatched gold → FN

---

## P/R/F1 Calculation

```python
def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1
```

---

## Aggregation Methods

### Micro F1 (Used in GUI)

Aggregates TP/FP/FN across all tags, then calculates F1:

```python
overall_tp = overall_fp = overall_fn = 0

for tag in TAGS:
    tp, fp, fn = ... # per-tag counts
    overall_tp += tp
    overall_fp += fp
    overall_fn += fn

p_all, r_all, f1_all = prf(overall_tp, overall_fp, overall_fn)
```

### Macro F1 (Used in CLI evaluate.py)

Calculates F1 per tag, then averages:

```python
macro = 0.0
for tag in TAGS:
    p, r, f1 = prf(tp, fp, fn)
    macro += f1

macro_f1 = macro / len(TAGS)
```

---

## Threshold Curve

The GUI generates threshold sweep curves for Relaxed mode:

```python
def threshold_curve(pred_items, gold_map, mode):
    if mode == "Exact":
        # Single point at threshold=1.0
        return single_point_df
    
    # Sweep from 0.00 to 1.00 in 0.05 increments
    ths = [round(i / 20, 2) for i in range(0, 21)]
    
    for t in ths:
        summary, _ = evaluate_run(pred_items, gold_map, mode=mode, threshold=t)
        # Collect precision, recall, f1 at each threshold
```

---

## Summary

### Current Evaluation Configuration

| Setting | Value |
|---------|-------|
| **Primary Metric** | Micro F1 |
| **Matching Modes** | Exact, Relaxed |
| **Relaxed Score Formula** | `0.65 × IoU + 0.35 × TextSim` |
| **Default Threshold** | 0.80 (configurable 0.00-1.00) |
| **Aggregation** | Per-tag + Overall (Micro) |

### Files Involved

| File | Role |
|------|------|
| `reqflow.py` | GUI evaluation (lines 415-591) |
| `src/tools/evaluate.py` | CLI evaluation (exact match only) |

### Limitations of Current System

1. **CLI evaluate.py** only supports exact match
2. **No text content verification** in CLI (only checks offsets)
3. **Fixed weight ratio** (0.65/0.35) not configurable
4. **Greedy matching** may not find optimal assignment in edge cases
