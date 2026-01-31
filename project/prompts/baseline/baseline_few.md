# Baseline (Single-Agent) — A3 Abstraction Extractor — ZERO-SHOT

You are a strict requirements analyst.

## Input
A single software requirement (one or more sentences):

{REQUIREMENT_TEXT}

## Task
Extract spans for the A3 abstractions:
- Purpose
- Trigger
- Precondition
- Condition
- Action
- System_response
- Entity
- Main_actor

### Definitions (aligned with the course slide)
- Entity: any involved object/role/system/component (human or non-human).
- Main_actor: the primary initiating role/actor (subset of Entity).
- Action: any action/event described (verb/verb-phrase).
- System_response: actions performed by the system (subset of Action).
- Condition: any limiting context/constraint (time, authorization, system state, ranges, availability, etc.).
- Precondition: a condition that MUST hold before the requirement applies (subset of Condition).
- Trigger: the event that activates the behavior (subset of Condition), often introduced by "when/upon/after/once".
- Purpose: intent/goal, typically expressed with "so that", "in order to", "for the purpose of".

## Output format (STRICT)
Return ONLY valid JSON (no markdown, no explanations) with this exact shape:
{
  "spans": [
    {"tag": "<one of: Purpose|Trigger|Precondition|Condition|Action|System_response|Entity|Main_actor>", "text": "<exact substring from input>"},
    ...
  ]
}

## Hard constraints
- Each "text" MUST be an exact substring of the input (copy-paste).
- Do NOT paraphrase, normalize, or add words not present.
- Use the minimum span that uniquely identifies the item (avoid large spans).
- Deduplicate exact duplicates (same tag + same text).
- If an abstraction is not explicitly present, omit it (do NOT hallucinate).

## Consistency constraints (must satisfy)
- Every System_response span must also appear as an Action span (same text).
- Every Main_actor span must also appear as an Entity span (same text).
- Every Precondition span must also appear as a Condition span (same text).
- Every Trigger span must also appear as a Condition span (same text).## Few-shot examples (3)

Example 1
Input:
"When a student submits a request, the system shall notify the administrator by email."
Output:
{"spans":[
  {"tag":"Condition","text":"When a student submits a request"},
  {"tag":"Trigger","text":"When a student submits a request"},
  {"tag":"Entity","text":"a student"},
  {"tag":"Main_actor","text":"a student"},
  {"tag":"Entity","text":"the system"},
  {"tag":"Entity","text":"the administrator"},
  {"tag":"Action","text":"a student submits a request"},
  {"tag":"Action","text":"the system shall notify the administrator by email"},
  {"tag":"System_response","text":"the system shall notify the administrator by email"}
]}

Example 2
Input:
"As long as the user is authenticated, the system shall allow booking edits during business hours."
Output:
{"spans":[
  {"tag":"Condition","text":"As long as the user is authenticated"},
  {"tag":"Precondition","text":"As long as the user is authenticated"},
  {"tag":"Condition","text":"during business hours"},
  {"tag":"Entity","text":"the user"},
  {"tag":"Main_actor","text":"the user"},
  {"tag":"Entity","text":"the system"},
  {"tag":"Action","text":"the system shall allow booking edits"},
  {"tag":"System_response","text":"the system shall allow booking edits"}
]}

Example 3
Input:
"The system shall log all access attempts so that administrators can audit security incidents."
Output:
{"spans":[
  {"tag":"Entity","text":"The system"},
  {"tag":"Entity","text":"all access attempts"},
  {"tag":"Entity","text":"administrators"},
  {"tag":"Action","text":"The system shall log all access attempts"},
  {"tag":"System_response","text":"The system shall log all access attempts"},
  {"tag":"Purpose","text":"so that administrators can audit security incidents"}
]}
