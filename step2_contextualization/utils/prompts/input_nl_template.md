You will create reusable natural-language templates for verbalizing item-level input in an optimization problem.  
All generated text must follow the tone, perspective, and style implied by:

- Task type: $problem_type
- Instruction template: $instruction_template
- Field-shape hint: $data_shape_hint

Before writing anything, interpret the instruction template and the field-shape hint together to understand what each global field and item-field group represents.

Before writing any output, you must determine the narrative perspective and voice used in $instruction_template. Your templates must use that exact same perspective, tone, and voice consistently in header, footer, and all line templates.
Do not shift perspective, add a narrator, or address the user unless the instruction template does so.

---

## Output mode

If the hint contains exactly one item-field group named "item_fields" and no other "*_item_fields", use **single-group mode**:

{
  "line_template": "...",
  "header": "...",
  "footer": "..."
}

If the hint contains multiple item-field groups, use **multi-group mode**:

{
  "<group1>_line_template": "...",
  "<group2>_line_template": "...",
  ...,
  "header": "...",
  "footer": "..."
}

Each `<group>` must match the exact group names from the hint.

---

## Template rules (strict)

### Allowed placeholders
Use only placeholders appearing in:
- hint.allowed_placeholders
- global field names (for header/footer)
- the item-field names of the relevant group (for its line template)

**Important:**  
- Use placeholders only in the form `{field_name}`.  
- Do **not** use `new_name` or any descriptive alias as a placeholder.  
- Do **not** invent additional fields or placeholder names.
---

### Line templates
For each item-field group (one or many):
- Include **all** placeholders from that group at least once.
- Do **not** include placeholders from other item groups or global fields.
- Write a short, natural line describing exactly one item.
- Match the **tone, perspective, and style** of `instruction_template`.
- Do not introduce new meanings.

### Header
- If global fields exist: include each global placeholder at least once.
- Exclude all item-level placeholders.
- The header should read as an immediate continuation of the instruction template. 
  It must fit naturally right after the wording and rhythm of `instruction_template`, 
  as if it were the next line in the same explanation.
- Match the exact narrative perspective and tone used in `instruction_template`
  (e.g., if it uses third-person neutral descriptions, continue in that voice; 
  if it uses “we”, continue with “we”, etc.).
- Keep it concise and scenario-consistent.
If there are no global fields: header = "".

---

### Footer
- Optional; may be "".
- No item-level placeholders.
- Global placeholders allowed only if also used in the header.
- Must use the **same tone and perspective** as `$instruction_template`, and provide a brief, natural closing remark.

---

## Final requirement
Output **only** the JSON object for the chosen mode.  
No extra text, comments, or markdown.
