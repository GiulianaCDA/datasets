 This script appears to be a part of a larger system, possibly a web application. It defines several functions related to the layout and display of a footer section on a page, as well as a function for rendering a scroll-up button. Here's a brief overview:

1. `layout_footer()`: This function is responsible for generating the footer section of a webpage. It includes the following steps:
   - If certain configuration options are enabled (show_timer, show_memory_usage, and/or show_queries_count), it calculates statistics about database queries executed during the page load, such as the total number of queries, unique queries, and total query execution time. These statistics are then displayed in a small text format.
   - It calls `log_print_to_page()` to print any log events that need to be displayed on the page.
   - Finally, it generates the closing markup for the footer section.

2. `layout_footer_begin()`: This function outputs the opening HTML tags and div structures needed to start rendering a footer section.

3. `layout_footer_end()`: This function closes the previously opened div structures, effectively ending the footer section.

4. `layout_scroll_up_button()`: This function generates an anchor tag with an icon that, when clicked, scrolls the user back to the top of the page. It appears that this function is not currently used or called within the provided code snippet.