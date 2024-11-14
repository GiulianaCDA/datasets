 This is a module that contains helper methods for working with queries in Redmine, such as filtering, sorting, and rendering the query results and sidebar queries. Here's a breakdown of each method:

* `build_query`: Builds a new query object based on the provided parameters, such as filters, columns, totalable names, group by, and sort criteria. If no parameters are provided, it attempts to retrieve them from the session using the `retrieve_query_from_session` method.
* `retrieve_query_from_session`: Retrieves a query object from the session based on the stored data. It first checks if an ID is present in the session data, and if so, it retrieves the query by ID using ActiveRecord's `find_by_id` method. If no ID is present, it creates a new query object with the provided filters, group by, column names, totalable names, and sort criteria.
* `query_as_hidden_field_tags`: Returns a string of hidden field tags that represent the query definition. This allows the query to be passed as parameters in a form submission.
* `query_hidden_sort_tag`: Returns a hidden field tag for the query's sort criteria, which can be used in a form submission.
* `sidebar_queries`: Returns an array of queries that should be rendered in the sidebar based on the current project and user permissions.
* `query_links`: Renders a group of queries as a list of links with appropriate classes for styling.
* `render_sidebar_queries`: Renders the list of queries for the sidebar by calling `sidebar_queries` and `query_links`. It first renders private queries, followed by public queries.