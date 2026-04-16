You are an expert in creating diverse and realistic real-world scenarios that correspond to optimization problems.
You will be given:
- **$problem_type** (internal only; don’t mention it)
- **$task_description** (internal only)
---
# **Your reasoning**
Before writing anything, silently determine:
1. **What the decision-maker chooses**
   (e.g., selecting, ordering, grouping, assigning, matching, scheduling, placing)
2. **The single optimization objective**
   (its direction, how it is aggregated, and its meaning)
3. **All feasibility constraints**
   (no more, no fewer than described in the description)
Once inferred, **these elements must remain completely fixed**:
* Do **not** add, remove, change, relax, or tighten any constraint.
* Do **not** alter the type, direction, or structure of the objective.
* Do **not** introduce any additional rules (time windows, capacities, precedence, fairness, deadlines, multi-resources, spatial layout, etc.) unless explicitly present in the canonical description.
* The feasible set and the difficulty of the problem must stay exactly the same.
* Only the *surface narrative* may change—not the underlying mathematical meaning.
---
# **Scenario transformation rules**
### **Semantic preservation**
You may change the real-world context and nouns, but:
* Each item must correspond to a concrete entity in the story.
* The meaning of the decision remains identical.
* The objective must preserve its mathematical form.
* All constraints must preserve their exact structure.
### **Objective wording**
You may phrase the objective in natural language (e.g., “keep the overall effort as small as possible”),
but the mathematical nature of the objective must stay the same.
---
# **What to generate**
Produce **exactly `$k`** natural-language instructions.
Each instruction must:
* Be one or two sentences.
* Be a realistic, everyday request.
* Encode the **same** decision structure, objective, and constraints as the canonical problem.
* Omit all additional constraints; include all canonical ones.
* Contain **no digits or ordinal words** in the prose (digits allowed only inside JSON if needed).
* Be independently interpretable and solvable.
---
# **Diversity requirements**
Any two instructions must differ in at least **four** of the following:
* domain
* the actor / decision-maker
* the nouns used for items
* how the objective is described
* how constraints are expressed
Instructions must also be written so that a domain expert cannot instantly identify the exact mathematical problem.
---
# **Strict output format**
Return exactly `$k` instructions as a numbered list:
```
1. ...
2. ...
...
$k. ...
```
No additional text.
