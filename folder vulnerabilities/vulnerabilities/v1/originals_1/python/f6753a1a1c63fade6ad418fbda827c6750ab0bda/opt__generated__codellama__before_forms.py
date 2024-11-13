

[PYTHON]
from django import forms
from django.utils.translation import gettext as _
from django.contrib.auth.models import Group
from weblate.models import Project, User, ProjectToken

class WeblateDateInput(forms.DateInput):
    input_type = 'date'

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name", "description"]
        widgets = {
            "name": forms.TextInput(attrs={"autofocus": True}),
            "description": forms.Textarea(),
        }

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "full_name", "email"]
        widgets = {
            "username": forms.TextInput(attrs={"autofocus": True}),
            "full_name": forms.TextInput(),
            "email": forms.EmailInput(),
        }

class ProjectTokenDeleteForm(forms.Form):
    token = forms.ModelChoiceField(
        queryset=ProjectToken.objects.none(),
        widget=forms.HiddenInput,
        required=True,
    )

    def __init__(self, project, *args, **kwargs):
        self.project = project
        super().__init__(*args, **kwargs)
        self.fields["token"].queryset = project.projecttoken_set.all()

class ProjectTokenCreateForm(forms.ModelForm):
    class Meta:
        model = ProjectToken
        fields = ["name", "expires"]
        widgets = {
            "expires": WeblateDateInput(),
            "project": forms.HiddenInput,
        }

    def __init__(self, project, *args, **kwargs):
        self.project = project
        kwargs["initial"] = {"project": project}
        super().__init__(*args, **kwargs)

    def clean_expires(self):
        expires = self.cleaned_data["expires"]
        expires = expires.replace(hour=23, minute=59, second=59, microsecond=999999)
        if expires < timezone.now():
            raise forms.ValidationError(gettext("Expiry cannot be in the past!"))
        return expires
[/PYTHON]
