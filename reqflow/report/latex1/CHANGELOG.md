# LaTeX Paper Revision Changelog

## Summary of Changes

This document describes the changes made to `acl_latex.tex` to improve accuracy, clarity, and alignment with the actual ReqFlow project implementation.

---

## Revision 3 (Latest) - Real Experimental Results

### Executed Actual Experiments
All results are now based on **real experiments** run on the full 50-requirement dataset:

**Experiments Run:**
- Baseline Zero-shot (50 requirements)
- Baseline One-shot (50 requirements)  
- Baseline Few-shot (50 requirements)
- Pipeline Zero-shot (50 requirements)
- Pipeline One-shot (50 requirements)
- Pipeline Few-shot (50 requirements)

**Model Used:** Qwen3 4B via Ollama (temperature=0.2)

### Corrected Table 1 (Dataset Distribution)
Based on actual counts from `dataset.csv`:

| Complexity | Previous (Wrong) | Actual (Correct) |
|------------|------------------|------------------|
| Simple | 22 | **23** |
| Conditional | 14 | **16** |
| Nested | 14 | **11** |
| **Total** | 50 | **50** |

### Added Three Results Tables with Real Data

**Table 2: Zero-shot Results**
| Metric | Baseline | Pipeline |
|--------|----------|----------|
| Micro-F1 | **0.65** | 0.40 |
| Macro-F1 | 0.57 | 0.41 |

**Table 3: One-shot Results**
| Metric | Baseline | Pipeline |
|--------|----------|----------|
| Micro-F1 | **0.50** | 0.46 |
| Macro-F1 | 0.39 | 0.39 |

**Table 4: Few-shot Results**
| Metric | Baseline | Pipeline |
|--------|----------|----------|
| Micro-F1 | **0.51** | 0.48 |
| Macro-F1 | 0.46 | 0.42 |

### Major Finding: Baseline > Pipeline

The real experiments revealed a **surprising result**: 
- **Baseline consistently outperforms Pipeline** in all configurations
- **Zero-shot outperforms Few-shot** (contrary to typical expectations)

This required rewriting the paper's narrative and conclusions.

### Updated Abstract
- Changed from claiming pipeline is better to accurately stating baseline performs best
- Updated best Micro-F1 from illustrative "0.75" to actual "0.65"

### Updated Discussion Section
- Added analysis of why pipeline underperforms (error propagation, context loss)
- Added analysis of why zero-shot works best (no example bias)
- Acknowledged these counter-intuitive findings

### Updated Conclusion
- Reversed the main finding: simpler approaches work better
- Added nuanced discussion about when complex workflows might not help

---

## Revision 2

### Added Architecture Diagram (Figure 2)
- TikZ-based diagram comparing Baseline vs Pipeline approaches

### Added Dataset Distribution Table
- Table showing complexity distribution

### Text Streamlined
- Reduced verbosity for 6-page limit

---

## Revision 1 (Initial)

### Title Revision
- **Before**: "Generating Software Requirements Through Abstractions"
- **After**: "Semantic Decomposition of Software Requirements Using Large Language Models"

### Dataset Size Correction
- **Before**: "35 functional requirements"
- **After**: "50 functional and non-functional requirements"

### Semantic Tags Corrected
- Fixed to match actual implementation: Main_actor, Entity, Action, System_response, Condition, Precondition, Constraint, Exception

### Added Related Work Section

### Removed Placeholder Content
- Removed empty appendix

---

## Experimental Results Summary

### Per-Tag Performance (Zero-shot Baseline - Best Configuration)

| Tag | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| Main_actor | 0.84 | 0.82 | 0.83 |
| Entity | 0.60 | 0.65 | 0.63 |
| Action | 0.43 | 0.77 | 0.55 |
| System_response | 0.74 | 0.93 | 0.82 |
| Condition | 0.56 | 0.96 | 0.71 |
| Precondition | 0.00 | 0.00 | 0.00 |
| Constraint | 0.58 | 0.23 | 0.33 |
| Exception | 1.00 | 0.50 | 0.67 |

**Note:** Precondition consistently fails (F1=0.00) across all configurations, suggesting tag schema refinement is needed.

---

## Files Modified

| File | Status |
|------|--------|
| `acl_latex.tex` | Major revision with real results |
| `CHANGELOG.md` | Updated (this file) |

## Experimental Data Location

All experimental results are stored in:
```
reqflow/project/result_eval/
├── zero/
│   ├── baseline/zero/baseline_spans.json
│   ├── pipeline/zero/pipeline_spans.json
│   ├── baseline_zero_eval.csv
│   └── pipeline_zero_eval.csv
├── one/
│   ├── baseline/one/baseline_spans.json
│   ├── pipeline/one/pipeline_spans.json
│   ├── baseline_one_eval.csv
│   └── pipeline_one_eval.csv
└── few/
    ├── baseline/few/baseline_spans.json
    ├── pipeline/few/pipeline_spans.json
    ├── baseline_few_eval.csv
    └── pipeline_few_eval.csv
```

---

## Verification Checklist

- [x] Title matches paper content
- [x] Abstract reflects actual findings
- [x] Dataset size correct (50 requirements)
- [x] Dataset distribution correct (23, 16, 11)
- [x] Tags match code implementation
- [x] **Results from real experiments**
- [x] Three tables (zero, one, few-shot)
- [x] Conclusions match actual findings
- [x] Within 6-page limit
