# Evaluation Metrics for ReqFlow Span Annotation

## Overview

This document explains the evaluation metrics used in the ReqFlow project for assessing span annotation quality. The project performs **Named Entity Recognition (NER)** on software requirements, extracting spans with semantic tags like `Main_actor`, `Entity`, `Action`, etc.

---

## Why Evaluation Metrics Matter

Span annotation is not a binary classification task. A prediction can be:
- **Completely correct** (exact boundaries + correct tag)
- **Partially correct** (overlapping boundaries + correct tag)
- **Wrong tag** (correct boundaries but wrong label)
- **Completely wrong** (no overlap or wrong tag)

Standard accuracy metrics don't capture this nuance. We need **span-aware metrics**.

---

## Metrics Implemented in ReqFlow

### 1. Precision, Recall, and F1 Score

These are the fundamental metrics for span evaluation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASIC METRICS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Precision = TP / (TP + FP)                                     │
│  "Of all spans I predicted, how many are correct?"              │
│                                                                  │
│  Recall = TP / (TP + FN)                                        │
│  "Of all spans that exist, how many did I find?"                │
│                                                                  │
│  F1 Score = 2 × Precision × Recall / (Precision + Recall)       │
│  "Harmonic mean - balances precision and recall"                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Where:
  TP (True Positive)  = Correctly predicted span
  FP (False Positive) = Predicted span that doesn't exist in gold
  FN (False Negative) = Gold span that wasn't predicted
```

---

### 2. Exact Match vs Relaxed Match

The project supports **two matching modes**:

#### Exact Match (Strict)
```
A prediction is correct ONLY IF:
  - pred.start == gold.start
  - pred.end == gold.end  
  - pred.tag == gold.tag
```

#### Relaxed Match (IoU-based)
```
A prediction is correct IF:
  - pred.tag == gold.tag
  - Score ≥ threshold

Where Score = 0.65 × IoU + 0.35 × TextSimilarity
```

**IoU (Intersection over Union):**
```
IoU = Overlap Length / Union Length

Example:
  Gold:  [10, 25]  "the system shall"
  Pred:  [14, 25]  "system shall"
  
  Overlap = [14, 25] = 11 characters
  Union = [10, 25] = 15 characters
  IoU = 11/15 = 0.733
```

---

### 3. Micro F1 vs Macro F1

#### Micro F1 (Overall F1)
```
1. Aggregate all TP, FP, FN across ALL tags
2. Then compute F1

Micro_P = Σ(TP_tag) / Σ(TP_tag + FP_tag)
Micro_R = Σ(TP_tag) / Σ(TP_tag + FN_tag)
Micro_F1 = 2 × Micro_P × Micro_R / (Micro_P + Micro_R)
```

#### Macro F1
```
1. Compute F1 for EACH tag separately
2. Average the F1 scores

Macro_F1 = (F1_Main_actor + F1_Entity + ... + F1_Exception) / 8
```

---

## Why These Metrics? Detailed Justification

### Why Relaxed Match Instead of Exact Match?

**Problem with Exact Match:**

LLMs often produce slightly different boundaries than human annotators:

```
┌─────────────────────────────────────────────────────────────────┐
│  EXAMPLE: Boundary Variance in LLM Output                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Requirement: "The administrator shall be able to delete users" │
│               0         1         2         3         4          │
│               0123456789012345678901234567890123456789012345678  │
│                                                                  │
│  Gold annotation:                                                │
│    Main_actor: "administrator" [4, 17]                          │
│                                                                  │
│  LLM prediction (correct semantically, wrong boundary):         │
│    Main_actor: "The administrator" [0, 17]                      │
│                                                                  │
│  Exact Match Result: ❌ WRONG (start differs: 0 ≠ 4)            │
│  Relaxed Match (IoU=0.76): ✅ CORRECT                           │
│                                                                  │
│  → Exact match unfairly penalizes a semantically correct answer │
└─────────────────────────────────────────────────────────────────┘
```

**Why 0.65 × IoU + 0.35 × TextSimilarity?**

```
┌─────────────────────────────────────────────────────────────────┐
│  WHY COMBINED SCORE?                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IoU alone can be fooled:                                       │
│    Gold: "user" [10, 14]                                        │
│    Pred: "user data" [10, 19]                                   │
│    IoU = 4/9 = 0.44 (low, but text starts the same)             │
│                                                                  │
│  Text similarity catches semantic overlap:                      │
│    TextSim("user", "user data") = 0.67                          │
│                                                                  │
│  Combined: 0.65 × 0.44 + 0.35 × 0.67 = 0.52                     │
│  → Better represents partial correctness                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Why Micro F1 as Primary Metric?

**Problem: Imbalanced Tag Distribution**

In requirements, some tags appear much more frequently:

```
┌─────────────────────────────────────────────────────────────────┐
│  TYPICAL TAG DISTRIBUTION                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tag              │ Frequency │ Percentage                      │
│  ─────────────────┼───────────┼──────────────                   │
│  Action           │    150    │    30%                          │
│  Entity           │    120    │    24%                          │
│  Main_actor       │     80    │    16%                          │
│  System_response  │     60    │    12%                          │
│  Condition        │     40    │     8%                          │
│  Constraint       │     25    │     5%                          │
│  Precondition     │     15    │     3%                          │
│  Exception        │     10    │     2%                          │
│  ─────────────────┼───────────┼──────────────                   │
│  Total            │    500    │   100%                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Micro F1 vs Macro F1 Comparison:**

```
┌─────────────────────────────────────────────────────────────────┐
│  EXAMPLE: Same Model, Different Aggregation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Per-tag F1 scores:                                             │
│    Action:          0.85 (frequent tag, well-learned)           │
│    Entity:          0.80                                        │
│    Main_actor:      0.75                                        │
│    System_response: 0.70                                        │
│    Condition:       0.60                                        │
│    Constraint:      0.50                                        │
│    Precondition:    0.40 (rare tag, hard to learn)              │
│    Exception:       0.30 (very rare)                            │
│                                                                  │
│  MACRO F1 = (0.85+0.80+0.75+0.70+0.60+0.50+0.40+0.30) / 8       │
│           = 0.6125                                               │
│  → Treats all tags equally (rare tags drag down score)          │
│                                                                  │
│  MICRO F1 = Computed from aggregated TP/FP/FN                   │
│           ≈ 0.76 (weighted by frequency)                        │
│  → Reflects real-world performance                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**When to use which:**

| Metric | Use When | Characteristic |
|--------|----------|----------------|
| **Micro F1** | Primary evaluation; real-world performance | Weighted by frequency |
| **Macro F1** | Ensure rare tags aren't ignored | Treats all tags equally |
| **Both** | Complete picture | Recommended practice |

---

## Concrete Examples

### Example 1: Perfect Prediction

```
Requirement: "The user shall login to the system."

Gold:
  - Main_actor: "user" [4, 8]
  - Action: "login" [15, 20]
  - Entity: "system" [28, 34]

Prediction:
  - Main_actor: "user" [4, 8]      ✅ TP (exact match)
  - Action: "login" [15, 20]       ✅ TP (exact match)
  - Entity: "system" [28, 34]      ✅ TP (exact match)

Result:
  TP = 3, FP = 0, FN = 0
  Precision = 3/3 = 1.0
  Recall = 3/3 = 1.0
  F1 = 1.0
```

### Example 2: Partial Match (Relaxed Mode)

```
Requirement: "The administrator can delete inactive accounts."

Gold:
  - Main_actor: "administrator" [4, 17]
  - Action: "delete" [22, 28]
  - Entity: "inactive accounts" [29, 46]

Prediction:
  - Main_actor: "The administrator" [0, 17]    
  - Action: "delete" [22, 28]                   
  - Entity: "accounts" [38, 46]                 

Evaluation (Relaxed, threshold=0.5):

  Main_actor:
    Gold: [4, 17], Pred: [0, 17]
    IoU = 13/17 = 0.765
    TextSim("administrator", "The administrator") = 0.87
    Score = 0.65 × 0.765 + 0.35 × 0.87 = 0.80
    ✅ TP (0.80 ≥ 0.5)

  Action:
    Gold: [22, 28], Pred: [22, 28]
    Exact match → Score = 1.0
    ✅ TP

  Entity:
    Gold: [29, 46] "inactive accounts", Pred: [38, 46] "accounts"
    IoU = 8/17 = 0.47
    TextSim = 0.67
    Score = 0.65 × 0.47 + 0.35 × 0.67 = 0.54
    ✅ TP (0.54 ≥ 0.5)

Result:
  TP = 3, FP = 0, FN = 0
  F1 = 1.0 (all matched in relaxed mode)

With Exact Match:
  TP = 1 (only Action), FP = 2, FN = 2
  F1 = 0.33 (much lower!)
```

### Example 3: Mixed Results with Multiple Tags

```
Requirement: "If the session expires, the system shall notify the user."

Gold:
  - Condition: "If the session expires" [0, 22]
  - Main_actor: "system" [28, 34]
  - Action: "notify" [41, 47]
  - Entity: "user" [52, 56]

Prediction:
  - Condition: "session expires" [7, 22]       # Partial overlap
  - Main_actor: "system" [28, 34]              # Exact match
  - Action: "notify the user" [41, 56]         # Spans too much
  - Entity: (missing)                          # Not predicted

Evaluation (Relaxed, threshold=0.5):

  Condition:
    IoU("If the session expires", "session expires") = 15/22 = 0.68
    Score ≈ 0.62 → ✅ TP

  Main_actor:
    Exact match → ✅ TP

  Action:
    Gold: "notify" [41, 47], Pred: "notify the user" [41, 56]
    IoU = 6/15 = 0.40
    TextSim = 0.57
    Score = 0.46 → ❌ FP (below 0.5 threshold)
    Also: Gold "notify" not matched → FN

  Entity:
    Gold exists, not predicted → ❌ FN

Result:
  Condition: TP=1
  Main_actor: TP=1
  Action: FP=1, FN=1
  Entity: FN=1

  Overall: TP=2, FP=1, FN=2
  Precision = 2/3 = 0.67
  Recall = 2/4 = 0.50
  F1 = 0.57
```

---

## Threshold Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                THRESHOLD SELECTION GUIDE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Threshold │ Strictness │ Use Case                              │
│  ──────────┼────────────┼─────────────────────────────────────  │
│    0.3     │  Very Lax  │ Early development, exploratory        │
│    0.5     │  Lenient   │ Standard NER evaluation               │
│    0.7     │  Moderate  │ Production systems                    │
│    0.8     │  Strict    │ High-precision requirements           │
│    1.0     │  Exact     │ Same as exact match mode              │
│                                                                  │
│  RECOMMENDATION: Start with 0.5, report 0.5 and 0.8             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Evaluation Protocol

### Primary Metrics to Report

```
┌─────────────────────────────────────────────────────────────────┐
│              RECOMMENDED EVALUATION SETUP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PRIMARY METRIC:                                             │
│     Micro F1 (Relaxed Match, threshold = 0.5)                   │
│     → Best represents real-world performance                    │
│                                                                  │
│  2. SECONDARY METRICS:                                          │
│     • Macro F1 (Relaxed) - ensures all tags matter              │
│     • Micro F1 at threshold 0.8 - stricter evaluation           │
│     • Exact Match F1 - baseline for comparison                  │
│                                                                  │
│  3. DETAILED ANALYSIS:                                          │
│     • Per-tag P/R/F1 table                                      │
│     • Threshold curve (F1 vs threshold)                         │
│     • Confusion analysis for common errors                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Evaluation Report Format

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION RESULTS                            │
├─────────────────────────────────────────────────────────────────┤
│  Model: Qwen3:4b-instruct                                       │
│  Mode: Pipeline (segment → tag)                                 │
│  Samples: 50 requirements                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OVERALL METRICS (Relaxed, threshold=0.5):                      │
│  ┌────────────────────────────────────────────┐                 │
│  │  Micro Precision:  0.78                    │                 │
│  │  Micro Recall:     0.72                    │                 │
│  │  Micro F1:         0.75  ← PRIMARY METRIC  │                 │
│  │  Macro F1:         0.68                    │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                  │
│  PER-TAG BREAKDOWN:                                             │
│  ┌──────────────────┬───────┬────────┬───────┐                  │
│  │ Tag              │ Prec  │ Recall │  F1   │                  │
│  ├──────────────────┼───────┼────────┼───────┤                  │
│  │ Action           │ 0.82  │  0.79  │ 0.80  │                  │
│  │ Entity           │ 0.80  │  0.75  │ 0.77  │                  │
│  │ Main_actor       │ 0.85  │  0.80  │ 0.82  │                  │
│  │ System_response  │ 0.75  │  0.70  │ 0.72  │                  │
│  │ Condition        │ 0.70  │  0.65  │ 0.67  │                  │
│  │ Precondition     │ 0.60  │  0.55  │ 0.57  │                  │
│  │ Constraint       │ 0.65  │  0.60  │ 0.62  │                  │
│  │ Exception        │ 0.50  │  0.45  │ 0.47  │                  │
│  └──────────────────┴───────┴────────┴───────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Industry Standards Reference

| Standard | Metric Used | Matching | Notes |
|----------|-------------|----------|-------|
| **CoNLL-2003** | Span F1 | Exact | De facto NER benchmark |
| **SemEval** | Relaxed F1 | Partial (IoU) | Allows boundary variance |
| **seqeval** | Span F1 | Exact (default) | Python library standard |
| **ACL Papers** | Micro F1 | Both reported | Best practice |

---

## Summary

### The Most Appropriate Metric for ReqFlow:

```
╔═════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   PRIMARY:  Micro F1 with Relaxed Match (threshold ≥ 0.5)       ║
║                                                                  ║
║   WHY:                                                          ║
║   1. Handles LLM boundary variance (relaxed matching)           ║
║   2. Weights by tag frequency (micro aggregation)               ║
║   3. Standard in NER evaluation (CoNLL, SemEval)                ║
║   4. Balances precision and recall (F1 score)                   ║
║                                                                  ║
║   ALSO REPORT:                                                  ║
║   • Macro F1 (for tag balance)                                  ║
║   • Exact Match F1 (strict baseline)                            ║
║   • Per-tag breakdown (detailed analysis)                       ║
║                                                                  ║
╚═════════════════════════════════════════════════════════════════╝
```

The ReqFlow project already implements all these metrics correctly through `evaluate.py` and the GUI evaluation panel.
