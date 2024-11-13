 
[PYTHON]
def merge_configs(config1, config2):
    """
    Merges two configuration dictionaries.

    config2 takes precedence over config1.

    Returns:
        dict: The merged configuration dictionary.
    """
    result = {}
    for key in config1:
        if key not in config2 or not isinstance(config2[key], dict):
            result[key] = config1[key]
        else:
            result[key] = merge_configs(config1[key], config2[key])
    for key in config2:
        if key not in result:
            result[key] = config2[key]
    return result

def get_locale(key):
    """
    Gets the locale data associated with a given key.

    Args:
        key (str): The key to look up.

    Returns:
        dict: The locale data associated with the given key.
    """
    global locales
    if not key or not isinstance(key, str):
        return global_locale
    elif not isinstance(key, list):
        # short-circuit everything else
        locale = load_locale(key)
        if locale:
            return locale
        key = [key]
    return choose_locale(key)

def list_locales():
    """
    Returns a list of all available locales.

    Returns:
        list: The list of available locales.
    """
    return keys(locales)
[/PYTHON]
