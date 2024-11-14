 It seems like this is a snippet of code written in C that deals with screen display and text processing. The functions included handle operations such as loading strings into a line buffer, shifting the display to show the end of the longest displayed line, managing colors for different types of text attributes, and setting or getting color maps.

Here's a brief overview:

1. `pappstr` function checks if a given string can be appended to the screen without overflowing. It returns 1 if it doesn't fit and 0 otherwise.
2. The `load_line` function loads a string into the line buffer, ensuring that the entire string fits on the screen. If it doesn't, it truncates the beginning of the string to fit.
3. The `rrshift` function calculates the necessary shift to display the end of the longest line in case where the line is longer than the screen width.
4. The `color_index` function returns the color map index associated with a given attribute, which is then used for rendering text on the screen.
5. The `set_color_map` and `get_color_map` functions allow setting and getting color strings for different attributes.