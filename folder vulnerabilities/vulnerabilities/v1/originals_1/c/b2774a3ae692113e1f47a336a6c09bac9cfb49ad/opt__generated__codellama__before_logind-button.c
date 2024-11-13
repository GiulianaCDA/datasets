[PYTHON]
import ctypes as ct

def get_struct_union_field(obj, field):
    # Get the address of a struct or union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return obj.__struct__.ctype._type_._fields_[field[0]]

def set_struct_union_field(obj, field, value):
    # Set the value of a struct or union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.addressof(obj)[field[1]] = value

def get_array_field(obj, field):
    # Get the value of an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents

def set_array_field(obj, field, value):
    # Set the value of an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents = value

def get_bitmask_field(obj, field):
    # Get the value of a bitmask field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents.value

def set_bitmask_field(obj, field, value):
    # Set the value of a bitmask field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents.value = value

def get_array_size(obj, field):
    # Get the size of an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return int((field[1] + (field[2] * ct.sizeof(field[3])) - 1) / (field[2] * ct.sizeof(field[3])))

def get_array_element(obj, field, index):
    # Get an element of an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return ct.cast(ct.addressof(obj) + (index * field[2] * ct.sizeof(field[3])), ct.POINTER(field[3])).contents

def set_array_element(obj, field, index, value):
    # Set an element of an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.cast(ct.addressof(obj) + (index * field[2] * ct.sizeof(field[3])), ct.POINTER(field[3])).contents = value

def get_array_string(obj, field):
    # Get a string from an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return ct.cast(ct.addressof(obj) + field[1], ct.c_char * get_array_size(obj, field)).value

def set_array_string(obj, field, value):
    # Set a string for an array field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.cast(ct.addressof(obj) + field[1], ct.c_char * get_array_size(obj, field)).value = value

def get_bitmask_string(obj, field):
    # Get a string from a bitmask field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return bin(get_bitmask_field(obj, field))[2:].zfill(get_array_size(obj, field) * 8)

def set_bitmask_string(obj, field, value):
    # Set a string for a bitmask field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    set_bitmask_field(obj, field, int(value.zfill(get_array_size(obj, field) * 8), 2))

def get_union_field(obj, field):
    # Get the value of a union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents

def set_union_field(obj, field, value):
    # Set the value of a union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents = value

def get_field(obj, field):
    # Get the value of a struct or union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        return getattr(obj, field)
    else:
        return ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents

def set_field(obj, field, value):
    # Set the value of a struct or union field.
    # The field can be specified by name or offset.
    if isinstance(field, str):
        return setattr(obj, field, value)
    else:
        ct.cast(ct.addressof(obj) + field[1], ct.POINTER(field[2])).contents = value

def get_bitmask_flag(obj, field, flag):
    # Get the value of a bitmask field as a boolean indicating whether the specified flag is set.
    # The field and flag can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    return bool(get_bitmask_field(obj, field) & (1 << flag))

def set_bitmask_flag(obj, field, flag, value):
    # Set the value of a bitmask field by setting or clearing a specific flag.
    # The field and flag can be specified by name or offset.
    if isinstance(field, str):
        field = obj.__struct__.fields[field]
    set_bitmask_field(obj, field, (get_bitmask_field(obj, field) & ~(1 << flag)) | (value << flag)))
[/PYTHON]
