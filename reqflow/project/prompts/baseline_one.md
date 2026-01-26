You are an expert requirements analyst. Your task: annotate the requirement text with labeled spans.

TAGS (use exactly these strings):
- Main_actor: primary actor that initiates the requirement (user, admin, system component).
- Entity: noun phrases / objects involved (UI elements, data items, resources).
- Action: actor action / intent (typically a verb phrase; often initiated by Main_actor).
- System_response: what the system shall do (responses, outputs, state changes).
- Condition: if/when/while clauses that condition behavior (including triggers).
- Precondition: assumed state before the requirement applies (login required, configuration, prior events).
- Constraint: quantitative/qualitative limits (time, format, performance, permissions).
- Exception: unless/except/otherwise clauses (edge cases, error conditions, alternative flow).


General rules:
- Offsets are character-based, 0-indexed, and END is exclusive (Python slicing).
- For every span, "text" MUST equal requirement_text[start:end] exactly.
- Prefer minimal spans (do not include extra surrounding punctuation/spaces unless they are part of the phrase).
- Overlapping spans are allowed (e.g., an Entity inside a System_response).
- Return ONLY valid JSON. No markdown, no explanations.


Output JSON schema (exact):
{
  "spans": [
    {"tag": "<TAG>", "start": <int>, "end": <int>, "text": "<substring>"}
  ]
}

ONE EXAMPLE
Requirement text:
When the user clicks 'Submit', the system shall validate the form and display an error message if any required field is missing.

Expected output JSON:
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

NOW DO THE TASK FOR THIS REQUIREMENT
Requirement text:
{{REQUIREMENT_TEXT}}
