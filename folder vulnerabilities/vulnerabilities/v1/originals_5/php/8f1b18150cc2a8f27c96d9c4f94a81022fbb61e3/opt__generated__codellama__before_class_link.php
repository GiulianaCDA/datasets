
 

[PYTHON]
import os

class LinkHelper:
    @staticmethod
    def get_link(page_i, page_e, action='', params='', systemid=''):
        if page_i == page_e and not action and not params and not systemid:
            return ''

        str_params = LinkHelper.parse_params(params)

        if systemid:
            str_params += '&systemid={}'.format(systemid)

        if action:
            str_params += '&action={}'.format(action)

        return '{page_i}?{params}'.format(page_i=page_i, params=str_params)

    @staticmethod
    def get_link_popup(page_i, page_e, action='', params='', systemid='', title='', width=500, height=500):
        if page_i == page_e and not action and not params and not systemid:
            return ''

        str_params = LinkHelper.parse_params(params)

        if systemid:
            str_params += '&systemid={}'.format(systemid)

        if action:
            str_params += '&action={}'.format(action)

        return '<a href="{page_i}?{params}" onclick="return !window.open(\'{page_i}?{params}\',\'{title}\',\'scrollbars=yes,resizable=yes,width={width},height={height}\')" title="{title}">{title}</a>'.format(page_i=page_i, params=str_params, title=title, width=width, height=height)

    @staticmethod
    def parse_params(params):
        if not params:
            return ''

        str_params = '&' + params

        # Remove first ampersand from params
        if str_params[0] == '&':
            str_params = str_params[1:]

        arr_params = []
        for param in str_params.split('&'):
            param = param.strip()

            if not param:
                continue

            if '=' in param:
                key, value = param.split('=')
                value = urllib.parse.quote(value)
                arr_params.append('{}={}'.format(key, value))
            else:
                arr_params.append(param)

        return '&'.join(arr_params)
[/PYTHON]
