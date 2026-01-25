"""
Compare Relaxed Match results across threshold range (0 to 1) vs Exact Match
"""
import sys
sys.path.insert(0, 'src')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

# Load files
PRED_PATH = r"D:\Final LLM\reqflow\project\result_cli\run_20260125_132523\baseline\zero\baseline_spans.json"
GOLD_PATH = r"D:\Final LLM\reqflow\project\data\gold_10samples.json"

TAGS = [
    "Main_actor", "Entity", "Action", "System_response",
    "Condition", "Precondition", "Constraint", "Exception",
]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def index_by_id(items):
    return {int(it["id"]): it for it in items if "id" in it}

def to_span_list(spans):
    result = []
    for sp in spans or []:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        text = sp.get("text", "")
        if tag in TAGS and isinstance(st, int) and isinstance(en, int):
            result.append({"tag": tag, "start": st, "end": en, "text": text})
    return result

def to_set(spans):
    s = set()
    for sp in spans or []:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        if tag in TAGS and isinstance(st, int) and isinstance(en, int):
            s.add((tag, st, en))
    return s

def compute_iou(start1, end1, start2, end2):
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    len1 = end1 - start1
    len2 = end2 - start2
    union = len1 + len2 - intersection
    return intersection / union if union > 0 else 0.0

def compute_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1

def evaluate_exact(pred, gold, common):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for rid in common:
        ps = to_set(pred[rid].get("spans"))
        gs = to_set(gold[rid].get("spans"))
        for tag in TAGS:
            p_tag = {x for x in ps if x[0] == tag}
            g_tag = {x for x in gs if x[0] == tag}
            counts[tag]["tp"] += len(p_tag & g_tag)
            counts[tag]["fp"] += len(p_tag - g_tag)
            counts[tag]["fn"] += len(g_tag - p_tag)
    return counts

def evaluate_relaxed(pred, gold, common, threshold):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for rid in common:
        pred_spans = to_span_list(pred[rid].get("spans"))
        gold_spans = to_span_list(gold[rid].get("spans"))
        
        for tag in TAGS:
            p_tag = [s for s in pred_spans if s["tag"] == tag]
            g_tag = [s for s in gold_spans if s["tag"] == tag]
            
            matched_gold = set()
            matched_pred = set()
            
            for pi, p_span in enumerate(p_tag):
                best_match = -1
                best_score = 0.0
                
                for gi, g_span in enumerate(g_tag):
                    if gi in matched_gold:
                        continue
                    
                    iou = compute_iou(p_span["start"], p_span["end"], 
                                     g_span["start"], g_span["end"])
                    text_sim = compute_text_similarity(p_span["text"], g_span["text"])
                    score = max(iou, text_sim)
                    
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_match = gi
                
                if best_match >= 0:
                    matched_gold.add(best_match)
                    matched_pred.add(pi)
            
            tp = len(matched_pred)
            fp = len(p_tag) - tp
            fn = len(g_tag) - len(matched_gold)
            
            counts[tag]["tp"] += tp
            counts[tag]["fp"] += fp
            counts[tag]["fn"] += fn
    
    return counts

def compute_metrics(counts):
    tag_f1_scores = []
    total_tp = total_fp = total_fn = 0
    
    for tag in TAGS:
        tp, fp, fn = counts[tag]["tp"], counts[tag]["fp"], counts[tag]["fn"]
        _, _, f1 = prf(tp, fp, fn)
        tag_f1_scores.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    macro_f1 = sum(tag_f1_scores) / len(tag_f1_scores) if tag_f1_scores else 0.0
    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": micro_p,
        "recall": micro_r,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

# Load data
print("Loading data...")
pred = index_by_id(load_json(PRED_PATH))
gold = index_by_id(load_json(GOLD_PATH))
common = sorted(set(pred.keys()) & set(gold.keys()))
print(f"Common IDs: {len(common)}")

# Evaluate exact match
print("\nEvaluating Exact Match...")
exact_counts = evaluate_exact(pred, gold, common)
exact_metrics = compute_metrics(exact_counts)
print(f"  Micro F1: {exact_metrics['micro_f1']:.4f}")
print(f"  Macro F1: {exact_metrics['macro_f1']:.4f}")

# Evaluate across thresholds
thresholds = np.arange(0.0, 1.05, 0.05)
results = []

print("\nEvaluating Relaxed Match across thresholds...")
for thr in thresholds:
    counts = evaluate_relaxed(pred, gold, common, thr)
    metrics = compute_metrics(counts)
    results.append({
        "threshold": thr,
        **metrics
    })
    print(f"  Threshold {thr:.2f}: Micro F1 = {metrics['micro_f1']:.4f}, Macro F1 = {metrics['macro_f1']:.4f}")

# Create visualization
print("\nGenerating comparison chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Baseline Zero-Shot: Relaxed Match vs Exact Match\nAcross Threshold Range (0.0 - 1.0)', 
             fontsize=14, fontweight='bold')

thrs = [r["threshold"] for r in results]
micro_f1s = [r["micro_f1"] for r in results]
macro_f1s = [r["macro_f1"] for r in results]
precisions = [r["precision"] for r in results]
recalls = [r["recall"] for r in results]

# 1. Micro F1 vs Threshold
ax1 = axes[0, 0]
ax1.plot(thrs, micro_f1s, 'b-o', linewidth=2, markersize=4, label='Relaxed Match')
ax1.axhline(y=exact_metrics['micro_f1'], color='red', linestyle='--', linewidth=2, 
            label=f'Exact Match ({exact_metrics["micro_f1"]:.4f})')
ax1.axvline(x=0.5, color='green', linestyle=':', alpha=0.7, label='Recommended (0.5)')
ax1.fill_between(thrs, micro_f1s, exact_metrics['micro_f1'], alpha=0.2, color='blue')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Micro F1 Score')
ax1.set_title('Micro F1 (PRIMARY) vs Threshold', fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.8)
ax1.grid(True, alpha=0.3)

# 2. Macro F1 vs Threshold
ax2 = axes[0, 1]
ax2.plot(thrs, macro_f1s, 'purple', linestyle='-', marker='s', linewidth=2, markersize=4, label='Relaxed Match')
ax2.axhline(y=exact_metrics['macro_f1'], color='red', linestyle='--', linewidth=2,
            label=f'Exact Match ({exact_metrics["macro_f1"]:.4f})')
ax2.axvline(x=0.5, color='green', linestyle=':', alpha=0.7, label='Recommended (0.5)')
ax2.fill_between(thrs, macro_f1s, exact_metrics['macro_f1'], alpha=0.2, color='purple')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Macro F1 Score')
ax2.set_title('Macro F1 (SECONDARY) vs Threshold', fontweight='bold')
ax2.legend(loc='upper right')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.8)
ax2.grid(True, alpha=0.3)

# 3. Precision vs Recall at different thresholds
ax3 = axes[1, 0]
scatter = ax3.scatter(recalls, precisions, c=thrs, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
ax3.scatter([exact_metrics['recall']], [exact_metrics['precision']], 
           c='red', s=200, marker='X', label='Exact Match', edgecolors='black', linewidth=1)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Threshold')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Trade-off', fontweight='bold')
ax3.legend(loc='lower left')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)

# Add annotations for key thresholds
for r in results:
    if r["threshold"] in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax3.annotate(f'{r["threshold"]:.2f}', (r["recall"], r["precision"]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

# 4. Summary metrics at key thresholds
ax4 = axes[1, 1]
key_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
key_results = [r for r in results if r["threshold"] in key_thresholds]
key_labels = [f'Thr={r["threshold"]:.2f}' for r in key_results] + ['Exact']

key_micro = [r["micro_f1"] for r in key_results] + [exact_metrics["micro_f1"]]
key_macro = [r["macro_f1"] for r in key_results] + [exact_metrics["macro_f1"]]

x = np.arange(len(key_labels))
width = 0.35

bars1 = ax4.bar(x - width/2, key_micro, width, label='Micro F1', color='#3498db', alpha=0.8)
bars2 = ax4.bar(x + width/2, key_macro, width, label='Macro F1', color='#9b59b6', alpha=0.8)

ax4.set_ylabel('F1 Score')
ax4.set_title('F1 Scores at Key Thresholds', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(key_labels, rotation=45, ha='right')
ax4.legend(loc='upper right')
ax4.set_ylim(0, 0.8)

for bar, val in zip(bars1, key_micro):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', 
             ha='center', fontsize=8)
for bar, val in zip(bars2, key_macro):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', 
             ha='center', fontsize=8)

plt.tight_layout()

output_path = r"D:\Final LLM\reqflow\project\result_cli\run_20260125_132523\threshold_comparison_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nChart saved to: {output_path}")

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY: Relaxed Match vs Exact Match across Thresholds")
print("=" * 70)
print(f"{'Threshold':<12} {'Micro F1':<12} {'Macro F1':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 70)
for r in results:
    if r["threshold"] in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"{r['threshold']:<12.2f} {r['micro_f1']:<12.4f} {r['macro_f1']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")
print("-" * 70)
print(f"{'Exact Match':<12} {exact_metrics['micro_f1']:<12.4f} {exact_metrics['macro_f1']:<12.4f} {exact_metrics['precision']:<12.4f} {exact_metrics['recall']:<12.4f}")
print("=" * 70)
print(f"\nRecommended: Threshold = 0.5")
thr_05 = [r for r in results if abs(r["threshold"] - 0.5) < 0.01][0]
print(f"  Micro F1: {thr_05['micro_f1']:.4f} (vs Exact: {exact_metrics['micro_f1']:.4f}, improvement: +{(thr_05['micro_f1']/exact_metrics['micro_f1']-1)*100:.1f}%)")
print(f"  Macro F1: {thr_05['macro_f1']:.4f} (vs Exact: {exact_metrics['macro_f1']:.4f}, improvement: +{(thr_05['macro_f1']/exact_metrics['macro_f1']-1)*100:.1f}%)")
