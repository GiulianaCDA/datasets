**Page Statistics Display**

The provided code snippet is responsible for rendering the footer section of a web page, including performance statistics.

**Key Features:**

* Memory usage (peak memory usage)
* Query count (total and unique queries)
* Query execution time
* Timer information

**Functionality:**

* **Memory Usage:** Calculates the peak memory usage in kilobytes and displays it in a small element with a lightning bolt icon.
* **Query Count:** Counts the total number of queries executed and displays it in a small element with a database icon. If `db_log_queries` is enabled, it also displays the number of unique queries executed.
* **Query Execution Time:** Calculates the total query execution time and displays it in a small element with a clock icon.
* **Timer:** If enabled, displays a timer element with the current time.

**Layout:**

The footer section is enclosed within `<div>` elements with the following classes:

* `clearfix`
* `space-20`
* `footer` (noprint)
* `footer-inner`
* `footer-content`

**Additional Features:**

* Scroll-up button
* Log event printing (via `log_print_to_page()` function)

**Configuration:**

The visibility of the statistics is controlled by configuration options:

* `show_timer`
* `show_memory_usage`
* `show_queries_count`

**Usage:**

The code is included in a PHP file and executed within a function called `layout_footer()`.

**Example Output:**

The footer section will display performance statistics such as:

* Memory usage (e.g., "Memory Usage: 1234 KB")
* Query count (e.g., "Total Queries: 100")
* Query execution time (e.g., "Total Query Execution Time: 0.5 seconds")
* Timer (e.g., "Timer: 2023-10-26 12:00:00")

**Conclusion:**

The provided code snippet provides a mechanism for displaying performance statistics in the footer section of a web page. It is configured to show relevant metrics and helps provide insights into the performance of the application.