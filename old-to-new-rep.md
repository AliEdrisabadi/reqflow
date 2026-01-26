# Evaluation System Implementation Report

## From Old to New: Comprehensive Evaluation Metrics Update

**Date:** January 26, 2026  
**Project:** ReqFlow - Span Annotation for Software Requirements  
**Scope:** Evaluation metrics implementation update

---

## Executive Summary

This report documents the implementation of the new evaluation metrics system for the ReqFlow project. The changes align the evaluation framework with industry-standard NER (Named Entity Recognition) evaluation practices, providing more comprehensive and fair assessment of span annotation quality.

### Key Changes Summary

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| Default Threshold | 0.80 | **0.50** (NER standard) |
| F1 Metrics (GUI) | Micro only | **Micro + Macro** |
| F1 Metrics (CLI) | Macro only | **Micro + Macro** |
| CLI Relaxed Mode | Not supported | **Fully supported** |
| Baseline Comparison | Not shown | **Exact Match F1** |
| User Guidance | None | **Tooltips added** |

---

## 1. Problem Statement

### Issues with the Old Implementation

1. **Overly Strict Threshold (0.80)**
   - LLMs often produce semantically correct spans with minor boundary differences
   - Threshold of 0.80 unfairly penalized correct predictions
   - Example: "The administrator" vs "administrator" would often fail despite being semantically equivalent

2. **Incomplete Metrics Reporting**
   - GUI only showed Micro F1
   - CLI only showed Macro F1
   - No way to compare both in a single evaluation

3. **Missing Baseline**
   - No Exact Match F1 shown for reference
   - Difficult to assess the benefit of relaxed matching

4. **Limited CLI Functionality**
   - CLI `evaluate.py` only supported exact matching
   - No IoU or text similarity scoring in CLI

---

## 2. Implementation Details

### 2.1 Default Threshold Change

**File:** `reqflow.py`  
**Location:** Line 1124

```python
# OLD
self.eval_thr.setCurrentText("0.80")

# NEW
self.eval_thr.setCurrentText("0.50")  # Changed from 0.80 to 0.50 (NER standard)
```

**Rationale:**
- 0.50 is the standard threshold in NER evaluation (CoNLL, SemEval)
- Provides a balance between precision and tolerance for boundary variance
- Users can still select stricter thresholds (0.8) when needed

---

### 2.2 Enhanced `evaluate_run()` Function

**File:** `reqflow.py`  
**Location:** Lines 515-588

**Changes:**
- Added Macro F1 calculation (average of per-tag F1 scores)
- Returns both `micro_f1` and `macro_f1` in summary dict

```python
def evaluate_run(pred_items, gold_map, mode, threshold):
    """
    Now returns:
    - micro_f1: Aggregated TP/FP/FN → single F1 (weighted by frequency)
    - macro_f1: Average of per-tag F1 scores (equal weight per tag)
    """
    # ... existing code ...
    
    # NEW: Collect per-tag F1 for Macro calculation
    tag_f1_scores: List[float] = []
    
    for tag in TAGS:
        # ... compute per-tag metrics ...
        tag_f1_scores.append(f1)
    
    # Micro F1 (from aggregated counts)
    p_micro, r_micro, f1_micro = prf(overall_tp, overall_fp, overall_fn)
    
    # NEW: Macro F1 (average of per-tag F1)
    f1_macro = sum(tag_f1_scores) / len(tag_f1_scores)
    
    summary = {
        # ... existing fields ...
        "micro_f1": round(f1_micro, 4),
        "macro_f1": round(f1_macro, 4),  # NEW
    }
```

---

### 2.3 New `evaluate_comprehensive()` Function

**File:** `reqflow.py`  
**Location:** Lines 591-608

**Purpose:** Compute all recommended metrics in a single call for comparison display.

```python
def evaluate_comprehensive(pred_items, gold_map, threshold=0.5):
    """
    Returns:
    - primary: Relaxed at user-selected threshold
    - strict: Relaxed at threshold 0.8
    - exact: Exact match baseline
    - df: Per-tag breakdown DataFrame
    """
    summ_relaxed, df = evaluate_run(..., mode="Relaxed", threshold=threshold)
    summ_strict, _ = evaluate_run(..., mode="Relaxed", threshold=0.8)
    summ_exact, _ = evaluate_run(..., mode="Exact", threshold=1.0)
    
    return {
        "primary": summ_relaxed,
        "strict": summ_strict,
        "exact": summ_exact,
        "df": df,
        "threshold": threshold,
    }
```

---

### 2.4 Updated `compute_evaluation()` Display

**File:** `reqflow.py`  
**Location:** Lines 1409-1481

**New 4-Line Summary Format:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Line 1: Run: run_20260126_123456 | Pred: Pipeline (few)       │
│  Line 2: ▸ Current (Relaxed, thr=0.5): Micro F1=0.75 |         │
│            Macro F1=0.68 | P=0.78 | R=0.72                      │
│  Line 3: ▸ Comparison: Exact Match F1=0.45 | Relaxed@0.8=0.62  │
│  Line 4: ▸ Counts: Common=50 | TP=234 | FP=66 | FN=91          │
└─────────────────────────────────────────────────────────────────┘
```

**Code:**
```python
# Line 2: Primary metric (current selection)
line2 = (
    f"▸ Current ({summ['mode']}, thr={summ['threshold']}): "
    f"Micro F1={summ['micro_f1']} | Macro F1={summ['macro_f1']} | "
    f"P={summ['precision']} | R={summ['recall']}"
)

# Line 3: Comparison metrics
if mode == "Relaxed":
    exact_f1 = comp['exact']['micro_f1']
    strict_f1 = comp['strict']['micro_f1']
    line3 = f"▸ Comparison: Exact Match F1={exact_f1} | Relaxed@0.8 F1={strict_f1}"
else:
    relaxed_f1 = comp['primary']['micro_f1']
    relaxed_macro = comp['primary']['macro_f1']
    line3 = f"▸ Comparison: Relaxed@{threshold} Micro F1={relaxed_f1} | Macro F1={relaxed_macro}"
```

---

### 2.5 UI Enhancements with Tooltips

**File:** `reqflow.py`  
**Location:** Lines 1107-1125

**Mode Dropdown:**
```python
mode_label.setToolTip(
    "Exact: spans must match exactly (tag, start, end)\n"
    "Relaxed: uses IoU + text similarity (recommended)"
)
self.eval_mode.setToolTip(
    "Relaxed mode recommended for LLM evaluation\n"
    "(handles boundary variance)"
)
```

**Threshold Dropdown:**
```python
thr_label.setToolTip(
    "Minimum score for a match in Relaxed mode.\n"
    "0.50 is the NER standard (recommended).\n"
    "0.80 is stricter."
)
self.eval_thr.setToolTip(
    "0.50 recommended (NER standard)\n"
    "0.80 for stricter evaluation"
)
```

---

### 2.6 CLI `evaluate.py` Complete Rewrite

**File:** `src/tools/evaluate.py`  
**Lines:** 1-371 (complete rewrite)

#### New Features:

| Feature | Implementation |
|---------|---------------|
| `--mode` argument | `choices=["exact", "relaxed"], default="relaxed"` |
| `--threshold` argument | `type=float, default=0.5` |
| IoU function | `iou(a, b)` - Intersection over Union |
| Text similarity | `text_sim(a, b)` - SequenceMatcher ratio |
| Relaxed scoring | `0.65 * IoU + 0.35 * TextSim` |
| Greedy matching | Sort by score, one-to-one assignment |
| Dual F1 output | Both Micro and Macro F1 in CSV |

#### New CLI Usage:

```bash
# Recommended: Relaxed at threshold 0.5
python evaluate.py --pred pred.json --gold gold.json --mode relaxed --threshold 0.5

# Exact match only
python evaluate.py --pred pred.json --gold gold.json --mode exact

# Stricter relaxed evaluation
python evaluate.py --pred pred.json --gold gold.json --mode relaxed --threshold 0.8
```

#### Sample CLI Output:

```
Evaluating 50 common requirements...
Mode: RELAXED, Threshold: 0.5
------------------------------------------------------------

Results:
  Micro Precision: 0.7800
  Micro Recall:    0.7200
  Micro F1:        0.7500 (PRIMARY)
  Macro F1:        0.6800
  Exact Match F1:  0.4500 (baseline)

Wrote results_f1.csv
```

#### CSV Output Format:

```csv
tag,tp,fp,fn,precision,recall,f1
Main_actor,45,12,8,0.7895,0.8491,0.8182
Entity,38,15,12,0.7170,0.7600,0.7379
...

# Summary
# Mode: RELAXED
# Threshold: 0.5
# Common requirements: 50
# TP: 234, FP: 66, FN: 91
MICRO_F1,,,,,, 0.7500
MACRO_F1,,,,,, 0.6800
EXACT_F1,,,,,, 0.4500
```

---

## 3. Testing Verification

### Files Modified

| File | Lines Changed | Type |
|------|--------------|------|
| `reqflow.py` | ~100 lines | Modified |
| `src/tools/evaluate.py` | 371 lines | Rewritten |

### Verification Points

| Check | Status |
|-------|--------|
| Default threshold = 0.50 | ✅ Verified |
| `evaluate_run()` returns micro_f1 | ✅ Verified |
| `evaluate_run()` returns macro_f1 | ✅ Verified |
| `evaluate_comprehensive()` exists | ✅ Verified |
| GUI shows Micro + Macro F1 | ✅ Verified |
| GUI shows Exact Match baseline | ✅ Verified |
| Tooltips on Mode dropdown | ✅ Verified |
| Tooltips on Threshold dropdown | ✅ Verified |
| CLI supports --mode argument | ✅ Verified |
| CLI supports --threshold argument | ✅ Verified |
| CLI reports Micro + Macro F1 | ✅ Verified |

---

## 4. Benefits of New Implementation

### 4.1 Fairer Evaluation

```
OLD: Threshold 0.80 too strict
     Gold: "administrator" [4, 17]
     Pred: "The administrator" [0, 17]
     Score = 0.80 → Barely passes at 0.80, fails at 0.81

NEW: Threshold 0.50 is standard
     Same prediction → Score = 0.80 → Clearly passes
     More fair to semantically correct LLM outputs
```

### 4.2 Complete Picture

```
OLD: GUI shows only Micro F1
     - Rare tags (Exception, Precondition) underweighted
     - Can't tell if model ignores rare tags

NEW: Shows both Micro F1 and Macro F1
     - Micro F1: Real-world weighted performance
     - Macro F1: Equal weight per tag (catches rare tag issues)
```

### 4.3 Reference Baseline

```
OLD: No baseline comparison
     - Hard to know if relaxed matching helps
     
NEW: Always shows Exact Match F1
     - Can see improvement from relaxed matching
     - Example: Exact=0.45, Relaxed@0.5=0.75 → +30% improvement
```

### 4.4 User Guidance

```
OLD: No guidance on settings
     - Users unsure what threshold to use

NEW: Tooltips explain everything
     - "0.50 is NER standard (recommended)"
     - "Relaxed mode recommended for LLM evaluation"
```

---

## 5. Alignment with Industry Standards

| Standard | Our Implementation |
|----------|-------------------|
| **CoNLL-2003** (Span F1, Exact) | ✅ Exact mode available |
| **SemEval** (Relaxed F1, IoU) | ✅ Relaxed mode with IoU |
| **seqeval** (Python library) | ✅ Compatible output format |
| **ACL Papers** (Report both) | ✅ Micro + Macro F1 |

---

## 6. Conclusion

The new evaluation implementation provides:

1. **Industry-standard threshold** (0.50) for fair LLM evaluation
2. **Comprehensive metrics** (Micro F1 + Macro F1)
3. **Baseline comparison** (Exact Match F1 always shown)
4. **Full CLI support** for relaxed matching
5. **User guidance** through tooltips

These changes ensure that ReqFlow's evaluation is fair, comprehensive, and aligned with academic and industry standards for NER/span annotation evaluation.

---

## Appendix: Quick Reference

### GUI Evaluation Settings

| Setting | Recommended | Alternative |
|---------|-------------|-------------|
| Mode | Relaxed | Exact (baseline) |
| Threshold | 0.50 | 0.80 (stricter) |

### Metrics Interpretation

| Metric | Meaning | When High Is Good |
|--------|---------|-------------------|
| Micro F1 | Weighted by span frequency | Always |
| Macro F1 | Equal weight per tag | Model handles all tags well |
| Exact F1 | Strict boundary matching | High precision needed |

### CLI Quick Commands

```bash
# Standard evaluation (recommended)
python evaluate.py --pred p.json --gold g.json

# Stricter evaluation
python evaluate.py --pred p.json --gold g.json --threshold 0.8

# Exact match only
python evaluate.py --pred p.json --gold g.json --mode exact
```
