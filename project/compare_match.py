import json

# Load both files
with open(r'D:\Final LLM\reqflow\project\result_cli\run_20260125_132523\baseline\zero\baseline_spans.json', 'r') as f:
    pred = json.load(f)

with open(r'D:\Final LLM\reqflow\project\data\gold_10samples.json', 'r') as f:
    gold = json.load(f)

print('=' * 80)
print('EXACT MATCH vs RELAXED MATCH COMPARISON')
print('=' * 80)
print()

# Compare ID 1
print('=== Requirement ID 1 ===')
print('Text: "A student shall be able to search study rooms..."')
print()

# Get spans for ID 1
pred_1 = [s for s in pred if s['id'] == 1][0]['spans']
gold_1 = [s for s in gold if s['id'] == 1][0]['spans']

print('TAG COMPARISON:')
print('-' * 80)

exact_matches = 0
relaxed_matches = 0

for g in gold_1:
    tag = g['tag']
    # Find predicted span with same tag
    p_list = [s for s in pred_1 if s['tag'] == tag]
    
    if p_list:
        p = p_list[0]
        exact = g['start'] == p['start'] and g['end'] == p['end']
        
        # Calculate IoU
        inter_start = max(g['start'], p['start'])
        inter_end = min(g['end'], p['end'])
        intersection = max(0, inter_end - inter_start)
        union = (g['end'] - g['start']) + (p['end'] - p['start']) - intersection
        iou = intersection / union if union > 0 else 0
        
        print(f'{tag}:')
        print(f'  Gold:       "{g["text"]}" (pos {g["start"]}-{g["end"]})')
        print(f'  Predicted:  "{p["text"]}" (pos {p["start"]}-{p["end"]})')
        print(f'  Exact Match: {"YES" if exact else "NO"}')
        print(f'  IoU Score:   {iou:.3f} -> Relaxed Match: {"YES" if iou >= 0.5 else "NO"}')
        print()
        
        if exact:
            exact_matches += 1
        if iou >= 0.5:
            relaxed_matches += 1
    else:
        print(f'{tag}:')
        print(f'  Gold:       "{g["text"]}"')
        print(f'  Predicted:  NOT FOUND')
        print()

print('=' * 80)
print('SUMMARY FOR ID 1:')
print(f'  Gold spans: {len(gold_1)}')
print(f'  Exact Matches: {exact_matches}')
print(f'  Relaxed Matches (IoU >= 0.5): {relaxed_matches}')
print('=' * 80)
print()
print('THIS IS WHY RELAXED MATCH GIVES HIGHER SCORES!')
print('The model often identifies the correct text but with slightly different boundaries.')
