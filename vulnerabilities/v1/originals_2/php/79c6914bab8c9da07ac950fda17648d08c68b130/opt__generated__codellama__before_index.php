
[PYTHON]
def get_module_settings(module):
    settings = {}
    if module:
        settings['id'] = module.get('id')
        settings['type'] = module.get('type')
        settings['data-type'] = module.get('data-type')
        settings['is_layout'] = 'y' if module.get('data-is-layout') else 'n'
    return settings

def get_module(id):
    module = False
    if id:
        module = mw.tools.get_element('#' + id)
    return module

def save_settings(data):
    params = data.copy()
    module = get_module(params['id'])
    if not module:
        return
    module.attrs(params)
    mw.reload_module('#' + params['id'], 1, 1)

def main():
    module = get_module(mw.url.window_hash())
    if not module:
        return ''
    settings = get_module_settings(module)
    form = '<div id="settings-main">'
    form += '<div id="settings-container">'
    form += '<form method="get" onsubmit="save_settings(this); return false;">'
    form += '  <input type="hidden" name="id" value="{id}" />'.format(**settings)
    for key in settings:
        form += '  <label>{key}</label>'.format(key=key)
        if key == 'type':
            select = '<select name="{key}">'.format(key=key)
            for option in ['mw-editor', 'mw-text-editor']:
                selected = ''
                if option == settings[key]:
                    selected = 'selected'
                select += '<option {selected}>{option}</option>'.format(option=option, selected=selected)
            select += '</select>'
            form += select
        else:
            form += '  <input type="text" name="{key}" value="{value}" />'.format(key=key, value=settings[key])
    form += '  <button>Save settings</button>'
    form += '</form>'
    form += '</div></div>'
    return form.format(**settings)

print(main())
[/PYTHON]
