 These code snippets are from a C program, likely written for a terminal or console application. They seem to be part of a library that handles text display and manipulation in a constrained environment, possibly within a limited screen size or character buffer. Here's an overview of each function:

1. **pappend(c, new_pos)**: This function appends a single character `c` to the line buffer, updating its position to `new_pos`. It returns whether the character was successfully appended (0 if successful, 1 if not).

2. **pappstr(str)**: This function appends an entire string `str` to the line buffer. If the string is too long to fit within the screen's width, it truncates the beginning of the string and continues appending until the whole string has been processed. It returns 0 if all characters were successfully appended, or 1 if any character did not fit.

3. **load_line(str)**: This function loads a given string `str` into the line buffer. If the string is too long to fit on the screen, it adjusts the display by shifting the text left until the entire string fits within the screen's boundaries. It then terminates the line with a null character.

4. **rrshift(VOID_PARAM)**: This function calculates and returns the necessary right shift (`rshift`) required to display the end of the longest line fully on the screen. If no lines are longer than the screen width, it returns 0.

5. **color_index(attr)**: A helper function that maps a given attribute `attr` to its corresponding color map index. This is used for mapping text attributes like bold, underline, etc., to specific colors or effects in the terminal.

6. **set_color_map(attr, colorstr)** and **get_color_map(attr)**: These functions manage a color map where each attribute (like AT_COLOR_PROMPT, AT_BOLD, etc.) is associated with a specific color or effect string. `set_color_map` sets the color for an attribute, while `get_color_map` retrieves it.

These snippets suggest that the program deals with text manipulation and display in a highly constrained environment where screen size, character buffer limitations, and possibly limited color support are significant factors. The functions handle dynamic adjustment of displayed text based on available space and manage various attributes like colors, shifts, and line wrapping to ensure optimal presentation within these constraints.