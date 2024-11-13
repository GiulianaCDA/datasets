**Description of the CapRTL Character Set**

The CapRTL character set is a testing character set used in conjunction with the reference implementation of the Fribidi algorithm. It contains all Unicode character types within the ASCII range (1-127) and provides explicit escape strings for each character.

**Character Types:**

The CapRTL character set includes the following 32 character types:

| Code | Character | Bidi Type |
|---|---|---|
| 0 | ^@ | LRM |
| 1 | ^A | RLM |
| 2 | ^B | LRE |
| 3 | ^C | RLE |
| 4 | ^D | PDF |
| 5 | ^E | LRO |
| 6 | ^F | RLO |
| 7 | ^G | LRI |
| 8 | ^H | RLI |
| 9 | ^I | FSI |
| 10 | ^J | PDI |
| 11 | ^K | _ |
| 12 | ^L | _ |
| 13 | ^M | _ |
| 14 | ^N | _ |
| 15 | ^O | _ |
| 16 | ^P | _ |
| 17 | ^Q | _ |
| 18 | ^R | _ |
| 19 | ^S | _ |
| 20 | ^T | _ |
| 21 | ^U | _ |
| 22 | ^V | _ |
| 23 | ^W | _ |
| 24 | ^X | _ |
| 25 | ^Y | _ |
| 26 | ^Z | _ |
| 27 | ^[ | _ |
| 28 | ^\ | _ |
| 29 | ^] | _ |
| 30 | ^^ | _ |
| 31 | ^_ | _ |

**Escape Sequences:**

The following escape sequences are used to represent specific character types in the CapRTL character set:

| Escape Sequence | Character Type |
|---|---|
| _> | LRM |
| _< | RLM |
| _l | LRE |
| _r | RLE |
| _L | LRO |
| _R | RLO |
| _o | PDF |
| _i | LRI |
| _y | RLI |
| _f | FSI |
| _I | PDI |

**Usage:**

The CapRTL character set can be used to test the functionality of the Fribidi algorithm and to demonstrate the different character types. It is also used in the reference implementation of the algorithm.

**Note:**

The CapRTL character set is subject to change.