[PYTHON]
class RedmineHelper:
    def initialize(view):
        self.view = view

    def ___(key, options = {}):
        return self.view.l(key, options)

    def wikitoolbar(text, object_type = nil):
        from redminenavhelper import WikiToolbarHelper
        return WikiToolbarHelper(self.view).render(text, object_type)

    def url_for(options = {}):
        return self.view.url_for(options)

    def avatar(user = nil, size = nil, default = nil):
        from redminenavhelper import GravatarHelper
        return GravatarHelper(self.view).render(user, size, default)

    def attachments_link(attachable, options = {}):
        return self.view.attachments_link(attachable, options)

    def attachment_icon(attachment, options = {}):
        return self.view.attachment_icon(attachment, options)

    def attachment_thumb(attachment, options = {}):
        return self.view.attachment_thumb(attachment, options)

    def attachment_link(attachment, text = nil, options = {}, html_options = {}):
        return self.view.attachment_link(attachment, text, options, html_options)

    def attachment_author(attachment, options = {}):
        return self.view.attachment_author(attachment, options)

    def attachment_created_on(attachment, options = {}):
        return self.view.attachment_created_on(attachment, options)

    def attachment_size(attachment, options = {}):
        return self.view.attachment_size(attachment, options)

    def attachments_size(attachable, size = nil, default = nil):
        from redminenavhelper import GravatarHelper
        return GravatarHelper(self.view).render(size, default)

    def attachment_age(attachment, options = {}):
        return self.view.attachment_age(attachment, options)

    def attachment_comments(attachment, options = {}):
        return self.view.attachment_comments(attachment, options)

    def attachments_size(attachable, size = nil, default = nil):
        from redminenavhelper import GravatarHelper
        return GravatarHelper(self.view).render(size, default)

    def user_link(user, html_options = {}):
        return self.view.user_link(user, html_options)

    def user_name(user, options = {}):
        return self.view.user_name(user, options)

    def user_avatar(user, size = nil, default = nil):
        from redminenavhelper import GravatarHelper
        return GravatarHelper(self.view).render(user, size, default)

    def user_path(user, html_options = {}):
        return self.view.user_path(user, html_options)

    def project_link(project, name = nil, html_options = {}):
        return self.view.project_link(project, name, html_options)

    def project_name(project, options = {}):
        return self.view.project_name(project, options)

    def project_identifier(project, options = {}):
        return self.view.project_identifier(project, options)

    def project_path(project, name = nil, html_options = {}):
        return self.view.project_path(project, name, html_options)

    def project_settings_path(project, *args, **options):
        return self.view.project_settings_path(project, *args, **options)

    def issue_link(issue, project = nil, html_options = {}):
        return self.view.issue_link(issue, project, html_options)

    def issue_id(issue):
        return self.view.issue_id(issue)

    def issue_identifier(issue):
        return self.view.issue_identifier(issue)

    def issue_path(issue, project = nil, html_options = {}):
        return self.view.issue_path(issue, project, html_options)

    def issue_url(issue, project = nil, html_options = {}):
        return self.view.issue_url(issue, project, html_options)

    def issues_list_link(project, name = nil, html_options = {}):
        return self.view.issues_list_link(project, name, html_options)

    def issues_index_path(project, *args, **options):
        return self.view.issues_index_path(project, *args, **options)

    def issue_status_name(issue):
        return self.view.issue_status_name(issue)

    def issue_tracker_url(issue):
        return self.view.issue_tracker_url(issue)

    def attachment_image_tag(attachment, options = {}):
        return self.view.attachment_image_tag(attachment, options)

    def attachment_thumb_tag(attachment, options = {}):
        return self.view.attachment_thumb_tag(attachment, options)

    def render_attachments_box(attachable, size = nil, default = nil):
        from redminenavhelper import GravatarHelper
        return GravatarHelper(self.view).render(attachable, size, default)

    def issue_subject(issue):
        return self.view.issue_subject(issue)

    def render_issue_link(issue, project = nil, html_options = {}):
        return self.view.render_issue_link(issue, project, html_options)

    def render_issue_id(issue):
        return self.view.render_issue_id(issue)

    def render_issue_identifier(issue):
        return self.view.render_issue_identifier(issue)

    def render_issue_path(issue, project = nil, html_options = {}):
        return self.view.render_issue_path(issue, project, html_options)

    def render_issue_url(issue, project = nil, html_options = {}):
        return self.view.render_issue_url(issue, project, html_options)

    def render_issues_list_link(project, name = nil, html_options = {}):
        return self.view.render_issues_list_link(project, name, html_options)

    def render_issues_index_path(project, *args, **options):
        return self.view.render_issues_index_path(project, *args, **options)

    def render_issue_status_name(issue):
        return self.view.render_issue_status_name(issue)

    def render_issue_tracker_url(issue):
        return self.view.render_issue_tracker_url(issue)

    def render_attachment_image_tag(attachment, options = {}):
        return self.view.render_attachment_image_tag(attachment, options)

    def render_attachment_thumb_tag(attachment, options = {}):
        return self.view.render_attachment_thumb_tag(attachment, options)
[/PYTHON]
