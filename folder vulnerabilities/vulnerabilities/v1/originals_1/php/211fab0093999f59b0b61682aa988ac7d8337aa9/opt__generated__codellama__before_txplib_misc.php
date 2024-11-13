 
[PYTHON]
def txp_filter(content, pattern):
    import re
    from txpath.lib.list_unique import do_list_unique
    from txpath.lib.list import do_list
    
    # Initialize variables
    cond = False
    dlmPool = ['|', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
    separator = pattern.get('separator') if pattern.get('separator') != None else False
    
    # Check for null value
    if content != None:
        # Get the match type and value from the pattern
        what = pattern.get('what')
        match = pattern.get('match')
        value = pattern.get('value')
        
        # Check if the match type is 'any' or 'all' and set the condition accordingly
        if match == 'any':
            values = do_list_unique(value)
            cond = False
            cf_contents = separator and not isinstance(content, list) and content.find(separator) != 0 and content.rfind(separator) != len(content) - 1 and content or do_list(content, separator)
            
            for term in values:
                if isinstance(cf_contents, list) and term in cf_contents or str(term) in str(cf_contents):
                    cond = True
                    break
        elif match == 'all':
            values = do_list_unique(value)
            cond = True
            cf_contents = separator and not isinstance(content, list) and content.find(separator) != 0 and content.rfind(separator) != len(content) - 1 and content or do_list(content, separator)
            
            for term in values:
                if isinstance(cf_contents, list) and term not in cf_contents or str(term) not in str(cf_contents):
                    cond = False
                    break
        else:
            # Match type is 'equal', 'not equal', 'less than', 'greater than', 'less than or equal to' or 'greater than or equal to'
            if what == [] and not isinstance(content, list):
                cond = False
            elif match == 'pattern':
                # Check if the pattern delimiter is available in the value attribute, if not then dynamically assign a delimiter that minimizes the possibility of a TXP-initiated preg_match error while still preserving errors outside TXP's control (e.g. mangled user-submitted PCRE pattern)
                dlm = separator and separator not in value and separator or value.find(separator) == 0 and separator or value.rfind(separator) == len(value) - 1 and separator or separator + value + separator
                
                cond = re.search(dlm, content)
            else:
                # Convert the condition to a boolean value based on the match type
                if what == [] and isinstance(content, list):
                    cond = len(content) > 0
                elif isinstance(what, list):
                    cond = True
                else:
                    cond = False
                    
                    # Check for empty values
                    if not value or (isinstance(value, str) and value == '') or (isinstance(value, list) and len(value) == 0):
                        return cond
                        
                    # Check for the match type
                    if match == 'equal':
                        cond = what == value
                    elif match == 'not equal':
                        cond = what != value
                    elif match == 'less than':
                        cond = what < value
                    elif match == 'greater than':
                        cond = what > value
                    elif match == 'less than or equal to':
                        cond = what <= value
                    elif match == 'greater than or equal to':
                        cond = what >= value
                        
                    # Check if the condition is true and return it
                    return cond
    
    # Return the final condition
    return cond

# -------------------------------------------------------------

def txp_break(wraptag):
    if wraptag == 'ul' or wraptag == 'ol':
        return 'li'
    elif wraptag == 'p':
        return 'br'
    elif wraptag == 'h1' or wraptag == 'h2' or wraptag == 'h3' or wraptag == 'h4' or wraptag == 'h5' or wraptag == 'h6':
        return 'p'
    else:
        return wraptag

# -------------------------------------------------------------

def txp_sort(content, pattern):
    import re
    
    # Initialize variables
    sort = False
    value = pattern.get('value')
    
    if content != None:
        sort = True
        
        if len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], list):
            for i in range(len(content)):
                for j in range(len(value[0])):
                    if value[0][j] == content[i]:
                        sort = False
                        break
                for j in range(len(value[1])):
                    if value[1][j] == i + 1:
                        sort = True
                        break
        else:
            for i in range(len(content)):
                if content[i] == value:
                    sort = False
                    break
    
    # Return the final condition
    return sort

# -------------------------------------------------------------

def txp_unique(content, pattern):
    import re
    
    # Initialize variables
    unique = True
    value = pattern.get('value')
    
    if content != None:
        unique = False
        
        if isinstance(value, str) and len(value) > 0:
            for i in range(len(content)):
                if re.search(value, content[i]):
                    unique = True
                    break
        elif isinstance(value, list) and len(value) > 0:
            for i in range(len(content)):
                for j in range(len(value)):
                    if value[j] == content[i]:
                        unique = False
                        break
    
    # Return the final condition
    return unique

# -------------------------------------------------------------

def txp_slice(content, pattern):
    import re
    
    # Initialize variables
    slice = None
    value = pattern.get('value')
    
    if content != None:
        slice = []
        
        for i in range(len(content)):
            if isinstance(value, str) and len(value) > 0:
                if re.search(value, content[i]):
                    slice.append(content[i])
            elif isinstance(value, list) and len(value) > 0:
                for j in range(len(value)):
                    if value[j] == content[i]:
                        slice.append(content[i])
    
    # Return the final condition
    return slice

# -------------------------------------------------------------

def txp_wrap(content, pattern):
    import re
    
    # Initialize variables
    wrap = []
    value = pattern.get('value')
    
    if content != None:
        wrap = [value]
    
    # Return the final condition
    return wrap

# -------------------------------------------------------------

def txp_unpack(content, pattern):
    import re
    
    # Initialize variables
    unpack = []
    value = pattern.get('value')
    
    if content != None:
        unpack = content[0]
    
    # Return the final condition
    return unpack

# -------------------------------------------------------------

def txp_call(content, pattern):
    import re
    
    # Initialize variables
    call = []
    value = pattern.get('value')
    
    if content != None:
        call = [content]
    
    # Return the final condition
    return call
[/PYTHON]
