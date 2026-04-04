---
name: readable-py
description: Readable Python code rules inspired by The Art of Readable Code. Use when writing, reviewing, or refactoring Python code. Enforces short functions, flat control flow, clear naming, readable structure, and Pythonic idioms.
---

# Readable Python Rules (/readable-py)

Apply these rules when writing, reviewing, or refactoring Python code. Inspired by *The Art of Readable Code* by Dustin Boswell and Trevor Foucher.

**Core principle: Code should be easy to understand.** The time it takes someone else (or future you) to understand the code is the ultimate metric.

## 1. Keep Functions Short and Focused

- A function should do **one thing**. If you can describe what it does with "and", split it.
- Aim for functions that fit on one screen (~15-25 lines). If it's longer, extract sub-tasks.
- Each function should operate at a **single level of abstraction** — don't mix high-level logic with low-level details in the same function.

## 2. Flatten Control Flow — No Deep Nesting

- **Never nest more than 2 levels deep.** If you have a loop inside a loop, or an `if` inside a loop inside an `if`, extract the inner block into a helper function with a descriptive name.
- Use **early returns / guard clauses** to handle edge cases at the top, keeping the main logic flat.
- Prefer `continue` or `break` to skip iterations rather than wrapping the body in a conditional.
- Replace complex conditionals with well-named helper functions or variables that explain the intent.

```
# Bad: nested and hard to follow
for user in users:
    if user.is_active:
        for order in user.orders:
            if order.is_pending:
                process(order)

# Good: flat, each function name explains what it does
active_users = get_active_users(users)
for user in active_users:
    process_pending_orders(user.orders)
```

## 3. Name Things Clearly

- **Pack information into names.** Use specific, concrete words — `fetch_page` not `get`, `num_retries` not `n`.
- **Avoid generic names** like `tmp`, `data`, `result`, `val`, `info`, `handle` — unless the scope is tiny (2-3 lines).
- **Use names that can't be misconstrued.** If a range is inclusive, say `max_items` not `limit`. If a boolean, use `is_`, `has_`, `should_`, `can_` prefixes.
- **Match the name length to the scope.** Short names for small scopes, descriptive names for wide scopes.
- **Don't use abbreviations** unless they're universally understood (`num`, `max`, `min`, `err` are fine; `svc_mgr_cfg` is not).

## 4. Make Control Flow Easy to Follow

- Put the **changing/interesting value on the left** side of comparisons: `if length > 10` not `if 10 < length`.
- Order `if/else` blocks: **positive case first**, simpler case first, or the more interesting case first.
- Minimize the number of variables the reader has to track. Reduce the **mental footprint** of each block.
- Avoid the **ternary operator** for anything non-trivial — if it's not immediately obvious, use an `if/else`.

## 5. Break Down Giant Expressions

- Use **explaining variables** to break complex expressions into named pieces.
- Use **summary variables** to capture a long expression that's used more than once.
- Apply **De Morgan's laws** to simplify negated boolean expressions.

```
# Bad
if not (age >= 18 and has_id and not is_banned):
    deny()

# Good
is_eligible = age >= 18 and has_id and not is_banned
if not is_eligible:
    deny()
```

## 6. Extract Unrelated Subproblems

- If a block of code is solving a **subproblem unrelated to the main goal** of the function, extract it.
- The helper function should be **pure and self-contained** — it shouldn't need to know about the calling context.
- This is the single most effective way to improve readability: separate *what* you're doing from *how*.

## 7. One Task at a Time

- Each section of code should do **one task**. If a function is doing parsing AND validation AND transformation, split them into separate steps.
- List the tasks a function does. If there's more than one, reorganize so each task is in its own block or function.

## 8. Reduce Variable Scope

- **Declare variables close to where they're used.** Don't declare at the top of a function if it's only used 30 lines later.
- **Minimize the "live time" of a variable** — the fewer lines between its assignment and last use, the easier it is to follow.
- **Prefer write-once variables.** Variables that are assigned once and never modified are easier to reason about.
- **Eliminate unnecessary variables.** If a variable is used only once and doesn't clarify anything, inline it.

## 9. No Magic Numbers or Strings

- Replace **magic numbers and strings** with named constants: `if retries > MAX_RETRIES` not `if retries > 3`.
- If a value has meaning, give it a name. The name documents the intent.
- Group related constants together.

## 10. Fewer Function Arguments

- Aim for **3 or fewer arguments** per function. More than that is a smell.
- Group related arguments into an **object, dataclass, or named tuple**.
- If a function needs many config-like options, pass a single config/options object.
- Boolean flag arguments are a sign the function does two things — split it instead.

## 11. Consistency

- If the codebase does something one way, **do it the same way**. Don't mix styles.
- Consistent naming patterns, consistent structure, consistent error handling.
- When joining an existing codebase, **match the existing conventions** even if you'd prefer a different style.
- Surprise is the enemy of readability — predictable code is readable code.

## 12. Write Less Code

- The best code is **no code at all**. Question whether a feature is truly needed before implementing.
- **Don't over-engineer.** Solve the problem at hand, not hypothetical future problems.
- Remove dead code. Commented-out code is dead code.
- Use standard libraries before writing custom solutions.

## 13. Comments: Explain Why, Not What

- Don't comment **what** the code does — the code already says that. Comment **why** it does it.
- Comment **flaws and workarounds**: `// TODO:`, `// HACK:`, `// XXX:` with explanation.
- Comment **surprising behavior** or non-obvious decisions — things where a reader would ask "why?".
- **Don't comment bad code — rewrite it.** If you need a comment to explain what a block does, extract it into a well-named function instead.

---

# Python-Specific Rules

## 14. Prefer Comprehensions — But Keep Them Simple

- Use list/dict/set comprehensions for **simple transforms and filters**.
- If a comprehension needs a nested loop AND a conditional, it's too complex — use a regular loop or extract a helper.
- Generator expressions for large sequences to avoid materializing the whole list.

```python
# Good: simple and readable
names = [user.name for user in users if user.is_active]

# Bad: too much going on
result = [transform(item) for group in data for item in group.items if item.valid and item.type == "A"]

# Good: break it up
valid_items = get_valid_items(data, item_type="A")
result = [transform(item) for item in valid_items]
```

## 15. Use Unpacking

- **Tuple unpacking** over index access: `name, age = get_user()` not `result[0], result[1]`.
- **Star unpacking** for head/tail: `first, *rest = items`.
- **Dict unpacking** with `**` for merging dicts.
- Unpacking makes the structure of the data explicit in the code.

## 16. Use `enumerate`, `zip`, and Itertools

- Use `enumerate(items)` — never track indices manually with `i += 1`.
- Use `zip(a, b)` to iterate in parallel — never index into parallel lists.
- Use `itertools` (`chain`, `groupby`, `islice`) before writing manual iteration logic.

## 17. Use Dataclasses and NamedTuples Over Raw Dicts/Tuples

- If a dict always has the same keys, it should be a `dataclass` or `NamedTuple`.
- If a function returns more than 2 values, return a `dataclass` or `NamedTuple` — not a raw tuple.
- This gives you names, type hints, and readable attribute access for free.

## 18. Use `pathlib` for File Paths

- Use `pathlib.Path` instead of `os.path.join` and string manipulation.
- `Path` objects are readable, composable (`/` operator), and cross-platform.
