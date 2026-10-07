# Demo task

Send this to the agent running in the demo project. It is written for any CLI
that can start subagents. Swap the model names for ones the CLI offers (keep a
mix of a small, a medium and a large one) and its own words for "Explore",
"Plan" or "background" if it has them. What matters for agents-tree is the
shape: five subagents at once, one in the background, one nested, different
models.

---

Demo task for this Acme Bookshop playground: please do the work below through
subagents, started in parallel, so it shows up as a tree of agents.

This is a throwaway demo repo; edit files freely, but do not commit. pytest may
not be installed system-wide; run tests with `uvx pytest`.

Spawn these five subagents in one go:

1. Explore agent, small model, description "Map bookshop modules": map the
   codebase (catalog, cart, inventory) and how a checkout flows from Cart to
   Inventory. Report only, no edits.

2. General agent, medium model, run in the background, description "Write cart
   and inventory tests": write pytest tests for bookshop/cart.py and
   bookshop/inventory.py in tests/, covering totals, discount codes,
   reservations and OutOfStock. Do not change anything under bookshop/. Run them
   with `uvx pytest` and report which fail (some should, the code has known
   bugs).

3. Plan agent, description "Plan configurable discounts": design how discount
   codes could come from configuration instead of being hard-coded in
   Cart.total_cents. A plan only, no code.

4. General agent, large model, description "Review checkout path": review
   Cart.checkout and Inventory.reserve/release/ship for correctness. Before
   writing the review, it must itself spawn one Explore subagent (small model,
   description "Find reserve and release callers") to find every caller of
   Inventory.reserve and Inventory.release. Report findings only, no edits.

5. General agent, small model, description "Fix negative cart total": fix the
   "may be a float, and may go below zero" bug in Cart.total_cents with the
   smallest change (integer cents, never below zero).

When all of them are done, give a short summary: what each found or changed,
and which tests still fail.
