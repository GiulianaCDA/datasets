 This function is a part of Radare2, an open-source reverse engineering framework. It's used to analyze functions in a binary that do not return, i.e., they never end with a return statement. Here's a step-by-step explanation:

1. The function takes a `core` object as its argument, which contains various components such as the analysis component (`anal`). It also takes an optional `request_fcn`, which is a specific function to analyze.

2. It initializes two data structures, `todo` and `done`. `todo` is a list of addresses that still need to be processed, and `done` is a hash table that keeps track of the addresses that have already been analyzed.

3. The function then enters a loop that continues as long as there are entries in the `todo` list. In each iteration of the loop, it pops an address from the `todo` list and retrieves all cross-references (xrefs) to that address.

4. For each xref, it checks if the reference type is a call. If it's not, the function moves on to the next xref.

5. The function then finds the block that has an instruction at exactly the xref addr. If such a block is found, it creates a list of functions contained in that block.

6. If a specific function (`request_fcn`) was requested, the function checks if it's in the list of functions contained in the block. If not, or if no specific function was requested, the block is chopped at the xref address to separate any non-returning code into its own block.

7. The function then iterates over all functions contained in the block. For each function, it checks if it has already been analyzed (by checking the `done` hash table). If not, it calls the `analyze_noreturn_function` function to analyze the function and determine if it's a non-returning function.

8. If the function is determined to be a non-returning function, its address is added to the `todo` list and marked as done in the `done` hash table. This ensures that any functions called by this non-returning function are also analyzed.

9. The loop continues until all addresses in the `todo` list have been processed. Finally, the `todo` and `done` data structures are freed to avoid memory leaks.