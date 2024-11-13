# In the controller:
@query = issue_query_helper.retrieve_query_from_session

# In the view:
<%= query_links('My Queries', issue_query_helper.sidebar_queries(IssueQuery, @project)) %>
