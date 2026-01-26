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


ONE EXAMPLE
requirement_text:
When the user clicks 'Submit', the system shall validate the form and display an error message if any required field is missing.

clauses_json:
[
  {
    "clause_id": 1,
    "start": 0,
    "end": 30,
    "text": "When the user clicks 'Submit',",
    "cue": "CONDITION"
  },
  {
    "clause_id": 2,
    "start": 31,
    "end": 94,
    "text": "the system shall validate the form and display an error message",
    "cue": "MAIN"
  },
  {
    "clause_id": 3,
    "start": 95,
    "end": 128,
    "text": "if any required field is missing.",
    "cue": "CONDITION"
  }
]

expected output JSON:
{
  "spans": [
    {
      "tag": "Condition",
      "start": 0,
      "end": 30,
      "text": "When the user clicks 'Submit',"
    },
    {
      "tag": "Main_actor",
      "start": 9,
      "end": 17,
      "text": "the user"
    },
    {
      "tag": "Action",
      "start": 14,
      "end": 29,
      "text": "clicks 'Submit'"
    },
    {
      "tag": "Main_actor",
      "start": 31,
      "end": 41,
      "text": "the system"
    },
    {
      "tag": "System_response",
      "start": 42,
      "end": 64,
      "text": "shall validate the form"
    },
    {
      "tag": "Entity",
      "start": 56,
      "end": 64,
      "text": "the form"
    },
    {
      "tag": "System_response",
      "start": 69,
      "end": 92,
      "text": "display an error message"
    },
    {
      "tag": "Entity",
      "start": 80,
      "end": 93,
      "text": "error message"
    },
    {
      "tag": "Condition",
      "start": 95,
      "end": 127,
      "text": "if any required field is missing"
    },
    {
      "tag": "Entity",
      "start": 102,
      "end": 116,
      "text": "required field"
    }
  ]
}

NOW DO THE TASK FOR THIS INPUT
requirement_text:
{{REQUIREMENT_TEXT}}

clauses_json:
{{CLAUSES_JSON}}
