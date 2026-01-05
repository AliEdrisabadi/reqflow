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

FEW-SHOT EXAMPLES

Example 1
Requirement text:
When the user clicks 'Submit', the system shall validate the form and display an error message if any required field is missing.
Output JSON:
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

Example 2
Requirement text:
After three failed login attempts, the system shall lock the account for 15 minutes and notify the user via email.
Output JSON:
{
  "spans": [
    {
      "tag": "Precondition",
      "start": 0,
      "end": 34,
      "text": "After three failed login attempts,"
    },
    {
      "tag": "Main_actor",
      "start": 35,
      "end": 45,
      "text": "the system"
    },
    {
      "tag": "System_response",
      "start": 46,
      "end": 68,
      "text": "shall lock the account"
    },
    {
      "tag": "Entity",
      "start": 57,
      "end": 68,
      "text": "the account"
    },
    {
      "tag": "Constraint",
      "start": 69,
      "end": 83,
      "text": "for 15 minutes"
    },
    {
      "tag": "System_response",
      "start": 88,
      "end": 111,
      "text": "notify the user via email"
    },
    {
      "tag": "Main_actor",
      "start": 95,
      "end": 103,
      "text": "the user"
    },
    {
      "tag": "Entity",
      "start": 108,
      "end": 113,
      "text": "email"
    }
  ]
}

Example 3
Requirement text:
The user shall be able to reset the password unless the account is suspended.
Output JSON:
{
  "spans": [
    {
      "tag": "Main_actor",
      "start": 0,
      "end": 8,
      "text": "The user"
    },
    {
      "tag": "Action",
      "start": 26,
      "end": 44,
      "text": "reset the password"
    },
    {
      "tag": "Entity",
      "start": 32,
      "end": 44,
      "text": "the password"
    },
    {
      "tag": "Exception",
      "start": 45,
      "end": 76,
      "text": "unless the account is suspended"
    },
    {
      "tag": "Entity",
      "start": 52,
      "end": 63,
      "text": "the account"
    }
  ]
}

NOW DO THE TASK FOR THIS REQUIREMENT
Requirement text:
{{REQUIREMENT_TEXT}}
