You are continuing a casual, human-sounding task description.

You will be given:
- **$problem_type** (internal only; don’t mention it)
- **$task_description** (internal only)
- **$output_format**: the exact JSON format the final answer must use.

Below is the draft that already covers the first part (the casual intro and the mention of the instance details):

<<<PARTIAL_INTRO_START>>>
$partial_intro
<<<PARTIAL_INTRO_END>>>

Do **not** change or repeat anything from this text.  
Your job now is to **add the remaining part**:

### Bring up the JSON format casually

* Work it into the flow in a relaxed, conversational way, as if you're just mentioning that the reply should follow a certain JSON layout.
* After that casual mention, show a JSON block that follows **exactly the same structure** as in **$output_format**.
  * Important: the top-level JSON keys (first layer only) must remain exactly identical to $output_format (do not rename them under any circumstances).
  * For nested placeholder keys and values (second layer and deeper only):
    * If the placeholder **key and value** looks like an **identifier-style label**
      (e.g., `"depot_id"`, `"store_id"`, `"facility_id"`, `<location_to_open>"`, `"node_id"`, `"item_id"`, `"room_id"`),
      you must rename **only the value prefix** to better match the story
      (for example, `"facility_id"` → `"warehouse_id"`, `<location_to_open>` → `<site_to_open>`, `<item_id>` → `<patch_type_id>`, `<location_id_for_first_facility>` → `<seat_id_for_first_room>`, `<employee_id>` → `<staff_id>`) **but**:

      * keep all JSON **keys** identical,
      * keep the entire JSON **structure** identical,
      * keep the placeholder format unchanged (it must remain a placeholder, not a concrete value).

  * If the placeholder values are **not identifier-style**
    (for example: binary vectors like `[0, 1, 0, 1]`, coordinates, widths/heights, counts, costs, times, booleans),
    **do not rename anything**.

    * Keep the example values **exactly as-is**
      (or only change numbers trivially if absolutely needed),
    * because there is **no story-specific prefix** to align in these cases.

* After showing the JSON block, give a short, easygoing explanation about what the parts of the JSON represent in the context of the story.
  Keep it light, informal, and very non-technical—more like explaining a form rather than a data schema.
* Make clear that the JSON is only a sketch of the expected shape, not the actual answer.

Also gently remind the reader that all identifiers must be used exactly as they appear in the instance input — no renaming and no new labels.
To be clear, valid identifiers look like plain numbers such as “1” or “23”, single capital letters like “A” or “B”, or a capital letter followed by digits like “A1” or “X7”.
Make sure these concrete examples are explicitly mentioned in the reminder. To avoid any ambiguity, spell this out clearly and keep the examples separate:
  - for example: "Valid identifiers look like plain numbers such as “1” or “23”, single capital letters like “A” or “B”, or a capital letter followed by digits like “A1” or “X7”."


---

### Critical constraints
- Do **not** repeat or alter the intro; just continue writing after it.
- Do **not** mention $problem_type or any technical optimization terminology.
- Keep the style consistent with the intro: casual, conversational, human.
- Do not number any part of your continuation.
- Do not write lines starting with digits and a period.

---

### Output rules
- Output **only** the continuation.  
- Do **not** include the intro again.  
- Do **not** wrap your answer in quotes or backticks—just write the continuation text and the JSON block normally.
