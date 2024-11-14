[PYTHON]
class QueryHelper:
    def initialize(context):
        self.context = context
    
    def query_hidden_sort_tag(query):
        return hidden_field_tag("sort", query.sort_criteria.to_param, id=None)

    def sidebar_queries(klass, project):
        return klass.visible.global_or_on_project(project).sorted().to_a()
    
    def render_sidebar_queries(klass, project):
        queries = sidebar_queries(klass, project)
        
        out = ''
        out += query_links(l('label_my_queries'), queries.select(&:is_private?))
        out += query_links(l('label_query_plural'), queries.reject(&:is_private?))
        return out
    
    def query_links(title, queries):
        if queries.empty():
            return ''
        
        # links to #index on issues/show
        url_params = controller_name == 'issues' ? {'controller': 'issues', 'action': 'index', 'project_id': self.project} : {}
        
        content_tag('h3', title) + "\n" + \
            content_tag('ul',
                queries.collect(lambda query:
                    content_tag('li', link_to(query.name, url_params.merge({'query_id': query}), {'class': 'query' if query == self.query else 'query selected'})), \
                        class_='queries')
            ) + "\n"
[/PYTHON]
