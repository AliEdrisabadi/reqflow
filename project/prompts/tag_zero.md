You are an expert requirements analyst.

Task: produce span annotations for the full requirement text, using the clauses as guidance.

Use only the TAGS below (exact strings):
Main_actor, Entity, Action, System_response, Condition, Precondition, Constraint, Exception

Rules:
- Offsets are character-based, 0-indexed, END exclusive.
- For every span, "text" MUST equal requirement_text[start:end] exactly.
- Prefer minimal spans; avoid duplicates.
- Overlaps are allowed.
- Use clauses_json only as extra structure; spans must reference the FULL requirement_text.
- Return ONLY valid JSON. No markdown, no explanations.

Output JSON schema (exact):
{
  "spans": [
    {"tag":"<TAG>", "start":<int>, "end":<int>, "text":"<substring>"}
  ]
}


Inputs:
requirement_text:
{{REQUIREMENT_TEXT}}

clauses_json:
{{CLAUSES_JSON}}
