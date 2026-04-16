You are an expert at interpreting generic structured-field definitions within a specific natural-language scenario.

You will receive:
1. Task name: $task_name
2. Task description: $task_description
3. Scenario text (describes the concrete instance): $base_text
4. Precomputed instruction intro (sets tone, role, and framing): $part12_intro
5. Generic hint JSON (defines the structured fields): $hint_json

---

## Your task
Rewrite the hint JSON so that every field has a scenario-specific meaning.  

You MUST interpret the fields **as they would be used under the instruction intro style** provided above.
That means:
- The implied role, perspective, and constraints from the intro must be respected.
- Field meanings should align with how a solver would understand the task after reading that intro.

The **top-level structure must remain identical** to the input hint:
- Keep all existing top-level keys exactly as they appear.
- Preserve all field groups (`global_fields`, `item_fields`, `facility_item_fields`, etc.).
- If the input is flat, keep it flat.
- Do NOT add, remove, or rename any top-level keys.

---

## How to rewrite each field
Each field object in the input has the form:

{
  "name": "field_name",
  "description": "generic meaning"
}

For EVERY field object in EVERY group, output an object with exactly:

{
  "name": "field_name",              // copy exactly
  "new_name": "scenario_specific_snake_case",
  "scenario_description": "scenario-specific interpretation"
}

Rules:
- `"name"` must stay unchanged.
- `"new_name"` must be a concise snake_case label meaningful in the scenario and unique within its group.
- `"scenario_description"` must clearly explain how the field should be understood **within the scenario text**.
- Do NOT include `"description"` or any extra keys.

---

## allowed_placeholders
Copy `"allowed_placeholders"` from the input **exactly**, unchanged.

---

## Output (STRICT)
Output **only one JSON object**:
- Same top-level keys, same ordering.
- All field groups rewritten as described.
- Nothing outside the JSON.
