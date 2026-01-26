# Comparison: Current vs Suggested Evaluation System

## Overview

This document compares the **current evaluation implementation** (old-ev.md) with the **suggested metrics** (Evaluation-metric.md) for the ReqFlow span annotation project.

---

## Side-by-Side Comparison

| Aspect | Current (old-ev.md) | Suggested (Evaluation-metric.md) |
|--------|---------------------|----------------------------------|
| **Matching Modes** | Exact + Relaxed | Exact + Relaxed |
| **Relaxed Score Formula** | `0.65 × IoU + 0.35 × TextSim` | `0.65 × IoU + 0.35 × TextSim` |
| **Primary Metric** | Micro F1 | Micro F1 (Relaxed, threshold=0.5) |
| **Default Threshold** | 0.80 | 0.50 (recommended) |
| **CLI Support** | Exact match only | Both modes recommended |
| **Reporting** | Per-tag + Overall | Per-tag + Micro F1 + Macro F1 + Threshold Curve |

---

## Key Differences

### 1. Default Threshold

| Current | Suggested |
|---------|-----------|
| **0.80** | **0.50** |

**Analysis:**
- Current (0.80) is **too strict** for LLM outputs
- Suggested (0.50) is the **standard for NER evaluation**
- A threshold of 0.80 will reject many semantically correct predictions with minor boundary differences

**Example Impact:**
```
Gold:  "administrator" [4, 17]
Pred:  "The administrator" [0, 17]

IoU = 13/17 = 0.765
TextSim = 0.87
Score = 0.65 × 0.765 + 0.35 × 0.87 = 0.80

At threshold 0.80: ✅ Barely passes
At threshold 0.50: ✅ Clearly passes
At threshold 0.85: ❌ Rejected (unfair!)
```

**Winner: Suggested** - threshold 0.50 is more appropriate for LLM evaluation.

---

### 2. Reporting Completeness

| Current | Suggested |
|---------|-----------|
| Per-tag P/R/F1 + Overall Micro | Per-tag + Micro F1 + Macro F1 + Multiple thresholds + Exact baseline |

**Suggested recommends reporting:**
1. **Primary**: Micro F1 (Relaxed, threshold=0.5)
2. **Secondary**: Macro F1 (ensures rare tags matter)
3. **Additional**: Micro F1 at threshold=0.8 (stricter)
4. **Baseline**: Exact Match F1 (for comparison)

**Winner: Suggested** - more comprehensive reporting.

---

### 3. Macro F1 Support

| Current GUI | Current CLI | Suggested |
|-------------|-------------|-----------|
| Micro only | Macro only | Both |

**Analysis:**
- GUI calculates **Micro F1** (weighted by frequency)
- CLI calculates **Macro F1** (equal weight per tag)
- Suggested recommends **both** for complete picture

**Why both matter:**
```
Tag Distribution Example:
  Action:     150 spans (30%)  - F1: 0.85
  Exception:   10 spans (2%)   - F1: 0.30

Micro F1 ≈ 0.76 (dominated by frequent tags)
Macro F1 = 0.61 (rare tags drag it down)

→ Micro shows "real-world" performance
→ Macro shows if model ignores rare tags
```

**Winner: Suggested** - reporting both gives fuller picture.

---

### 4. CLI Evaluation Capability

| Current | Suggested |
|---------|-----------|
| Exact match only | Both Exact and Relaxed |

**Current CLI (`evaluate.py`):**
```python
# Only uses (tag, start, end) tuple - no text similarity
s.add((tag, st, en))
```

**Winner: Suggested** - CLI should support relaxed matching too.

---

## What's the Same (Both Are Good)

| Feature | Status |
|---------|--------|
| Score formula `0.65 × IoU + 0.35 × TextSim` | ✅ Same |
| IoU calculation | ✅ Same |
| Text similarity via SequenceMatcher | ✅ Same |
| Greedy matching algorithm | ✅ Same |
| Per-tag breakdown | ✅ Both support |
| Threshold curve visualization | ✅ GUI supports |

---

## Summary: Which is Better?

| Category | Winner |
|----------|--------|
| **Core Formula** | Tie (same) |
| **Default Threshold** | **Suggested** (0.50 vs 0.80) |
| **Reporting Completeness** | **Suggested** |
| **Macro + Micro Support** | **Suggested** |
| **CLI Capability** | **Suggested** |
| **Documentation** | **Suggested** (clearer rationale) |

---

## Verdict

**The Suggested metrics (Evaluation-metric.md) are better** because:

1. **More appropriate threshold** (0.50) - Standard for NER, fairer to LLM boundary variance
2. **More complete reporting** - Both Micro and Macro F1
3. **Better justified** - Clear rationale for each design choice
4. **Industry aligned** - References CoNLL, SemEval standards

---

## Recommended Changes to Current System

To align with suggested metrics:

| Change | Priority |
|--------|----------|
| Change default threshold from 0.80 to **0.50** | High |
| Add Macro F1 to GUI summary | Medium |
| Add relaxed matching to CLI `evaluate.py` | Medium |
| Report both Micro and Macro F1 by default | Medium |
| Add exact match F1 as baseline comparison | Low |

---

## Implementation Checklist

- [ ] Update `reqflow.py`: Change default threshold combo from "0.80" to "0.50"
- [ ] Update `reqflow.py`: Add Macro F1 to evaluation summary display
- [ ] Update `src/tools/evaluate.py`: Add relaxed matching mode
- [ ] Update `src/tools/evaluate.py`: Add IoU and text similarity functions
- [ ] Update CLI output to show both Micro and Macro F1
- [ ] Add exact match F1 as reference baseline in reports
