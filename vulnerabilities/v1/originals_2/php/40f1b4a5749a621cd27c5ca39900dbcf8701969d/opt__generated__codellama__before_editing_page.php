 
[PYTHON]
import os
from flask import redirect, url_for, render_template, request, flash, session
from flask_login import login_required, current_user
from .forms import FileForm
from .. import db
from ..models import User

@bp.route('/gallery/edit', methods=['GET', 'POST'])
@login_required
def gallery_edit():
    form = FileForm()
    if form.validate_on_submit():
        file_name = secure_filename(form.file.data.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        form.file.data.save(file_path)
        flash('File successfully uploaded')
        return redirect(url_for('main.gallery'))
    return render_template('edit.html', title='Edit', form=form)
[/PYTHON]
