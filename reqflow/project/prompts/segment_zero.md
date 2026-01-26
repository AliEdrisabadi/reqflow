You are an expert requirements analyst.

Task: split the requirement into meaningful clauses (segments) and label each clause with a cue.

Cue labels (use exactly one of):
- MAIN (core requirement)
- PRECONDITION (assumed state before MAIN applies)
- CONDITION (trigger / if/when/while)
- CONSTRAINT (limit such as time, quantity, format)
- EXCEPTION (unless/except/otherwise edge case)

Rules:
- Offsets are character-based, 0-indexed, END exclusive.
- Every clause "text" MUST equal requirement_text[start:end] exactly.
- Clauses should be ordered by start offset.
- Clauses may overlap, but prefer a clean partition when possible.
- If you are unsure, return a single MAIN clause covering the full requirement.
- Return ONLY valid JSON. No markdown, no explanations.

Output JSON schema (exact):
{
  "clauses": [
    {"clause_id": <int>, "start": <int>, "end": <int>, "text": "<substring>", "cue": "<CUE>"}
  ]
}


Requirement text:
{{REQUIREMENT_TEXT}}
