
This code is a function called `analyze_noreturn` that takes two arguments: `core`, which is the R2 instance, and `request_fcn`, which is an optional argument. The function starts by initializing some lists and hashtables needed for its work. It then loops over the xrefs of the given address (or all xrefs if `request_fcn` is not set) and looks for calls to functions that have been marked as noreturn.

Here's a step-by-step breakdown of what each line does:

1. `RList *todo = r_list_new ();`: Creates an empty list called `todo`.
2. `HtUu *done = ht_uu_new (NULL, free);`: Creates a hashtable to keep track of the functions that have already been analyzed. The function is not actually used, so it is passed as NULL.
3. `RList *xrefs = r_anal_xrefs_get (core->anal, noret_addr);`: Gets a list of xrefs for the given address. If `request_fcn` is set, this will only get xrefs to functions in that specific function.
4. `RAnalRef *xref;`: Declares a variable called `xref` that will be used to loop over the elements of the `xrefs` list.
5. `r_list_foreach (xrefs, iter, xref) { ... }`: Loops over each element in the `xrefs` list and does something with it. The `iter` variable is not actually used, so it is just a placeholder.
6. `RAnalOp *xrefop = r_core_op_anal (core, xref->addr, R_ANAL_OP_MASK_ALL);`: Gets the disassembly of an instruction at the given address using the `r_core_op_anal` function.
7. `ut64 call_addr = xref->addr;`: Saves the current address as a variable called `call_addr`.
8. `ut64 chop_addr = call_addr + xrefop->size;`: Calculates the next address to analyze by adding the size of the instruction at `call_addr` to it.
9. `r_anal_op_free (xrefop);`: Frees the memory used by the disassembly.
10. `if (xref->type != R_ANAL_REF_TYPE_CALL) { ... }`: If the type of xref is not a call, skips to the end of the block.
11. `RList *blocks = r_anal_get_blocks_in (core->anal, call_addr);`: Gets a list of all blocks that contain instructions at or after `call_addr`.
12. `if (!blocks) { ... }`: If there are no blocks, skips to the end of the block.
13. `RAnalBlock *block = NULL;`: Declares a variable called `block` and initializes it to NULL.
14. `RListIter *bit;`: Declares a variable called `bit` that will be used to loop over the elements of the `blocks` list.
15. `RAnalBlock *block_cur;`: Declares a variable called `block_cur` that will be used to store each block as it is looped over.
16. `r_list_foreach (blocks, bit, block_cur) { ... }`: Loops over each element in the `blocks` list and does something with it. The `bit` variable is not actually used, so it is just a placeholder.
17. `if (r_anal_block_op_starts_at (block_cur, call_addr)) { ... }`: If the current block starts at the address of the xref, sets the `block` variable to that block and breaks out of the loop.
18. `if (block) { ... }`: If there is a block, does something with it.
19. `r_anal_block_unref (block);`: Releases the memory used by the `block`.
20. `r_list_free (blocks);`: Frees the memory used by the `blocks` list.
21. `RList *block_fcns = r_list_clone (block->fcns);`: Creates a copy of the `block->fcns` list and saves it to a variable called `block_fcns`.
22. `if (request_fcn && !r_list_contains (block_fcns, request_fcn)) { ... }`: If the `request_fcn` argument is set and the current block does not contain that function, skips to the end of the block.
23. `RListIter *iter;`: Declares a variable called `iter` that will be used to loop over the elements of the `block_fcns` list.
24. `void *function;`: Declares a variable called `function` that will store each function as it is looped over.
25. `r_list_foreach (block_fcns, iter, function) { ... }`: Loops over each element in the `block_fcns` list and does something with it. The `iter` variable is not actually used, so it is just a placeholder.
26. `RAnalFunction *anal_function = r_anal_get_function (core->anal, function);`: Gets the analysis of the given function using the `r_anal_get_function` function.
27. `if (!anal_function || !r_anal_function_has_flag (anal_function, R_ANAL_FUNC_FLAG_NORETURN)) { ... }`: If the current block does not contain a function marked as noreturn, skips to the end of the block.
28. `RAnalFunction *func = anal_function;`: Saves the analysis of the current function to a variable called `func`.
29. `r_list_append (todo, func);`: Adds the current function to the `todo` list.
30. `}`: Ends the loop over each element in the `block_fcns` list.
31. `r_list_free (xrefs);`: Frees the memory used by the `xrefs` list.
32. `return todo;`: Returns the `todo` list.