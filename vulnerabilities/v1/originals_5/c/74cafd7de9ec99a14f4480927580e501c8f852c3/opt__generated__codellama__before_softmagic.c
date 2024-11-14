
[PYTHON]
def handle_annotation(ms: "magic_set", m: "magic") -> bool:
    """
    Handle the annotation of a magic entry.

    :param ms: The magic set to use.
    :param m: The magic entry to check.
    :return: Whether an annotation was found and handled successfully.
    """
    if (ms.flags & MAGIC_APPLE):
        if file_printf(ms, "%.8s", m.apple) == -1:
            return -1
        return True
    elif (ms.flags & MAGIC_MIME_TYPE) and m.mimetype[0]:
        if file_printf(ms, "%s", m.mimetype) == -1:
            return -1
        return True
    else:
        return False

def match(ms: "magic_set", m: "magic") -> bool:
    """
    Check if the file matches a magic entry.

    :param ms: The magic set to use.
    :param m: The magic entry to check.
    :return: Whether the file matched successfully.
    """
    v = extract_bits(ms, m)
    if (ms.flags & MAGIC_DEBUG):
        print_d(f"{m.offset}: {v} ({get_flag_name(m)})")
    if not v:
        return False
    matched = 0
    if (ms.flags & MAGIC_MIME) and m.mimetype[0]:
        return True
    for r in get_relations(ms, m):
        # TODO: check the relation is valid
        l = r.offset
        matched |= check_relation(ms, v, m, l)
    if (ms.flags & MAGIC_DEBUG):
        print_d(f"matched: {matched}")
    return bool(matched)

def extract_bits(ms: "magic_set", m: "magic") -> int:
    """
    Extract the bits from a file that match a magic entry.

    :param ms: The magic set to use.
    :param m: The magic entry to check.
    :return: The extracted bits.
    """
    # TODO: handle offsets and sizes properly
    if m.flag & OFFSET:
        return 0
    elif m.flag & UNSIGNED:
        return ms.offset
    else:
        return ms.offset >> 1

def check_relation(ms: "magic_set", v: int, m: "magic", l: int) -> bool:
    """
    Check the relation between two values.

    :param ms: The magic set to use.
    :param v: The first value.
    :param m: The magic entry to check.
    :param l: The second value.
    :return: Whether the relation is true.
    """
    if (ms.flags & MAGIC_DEBUG):
        print_d(f"checking {m.reln} {v} {l}")
    matched = 0
    if m.flag & UNSIGNED:
        if m.reln == '>':
            matched = v > l
        elif m.reln == '<':
            matched = v < l
        elif m.reln == '&':
            matched = (v & l) == l
        elif m.reln == '^':
            matched = (v & l) != l
    else:
        if m.reln == '>':
            matched = (v > l) or ((v < 0) and (l >= -2147483648))
        elif m.reln == '<':
            matched = (v < l) or ((v >= 0) and (l <= 2147483647))
        elif m.reln == '&':
            matched = (v & l) == l
        elif m.reln == '^':
            matched = (v & l) != l
    if (ms.flags & MAGIC_DEBUG):
        print_d(f"matched: {matched}")
    return matched

def get_flag_name(m: "magic") -> str:
    """
    Get the name of a magic flag.

    :param m: The magic entry to use.
    :return: The name of the flag.
    """
    if m.flag & OFFSET:
        return "OFFSET"
    elif m.flag & UNSIGNED:
        return "UNSIGNED"
    else:
        return "SIGNED"

def get_relations(ms: "magic_set", m: "magic") -> list:
    """
    Get the relations for a magic entry.

    :param ms: The magic set to use.
    :param m: The magic entry to check.
    :return: A list of relations.
    """
    if (m.flag & UNSIGNED) or ((ms.flags & MAGIC_DEBUG) and not isinstance(m, "magic")):
        return [m]
    else:
        return []
[/PYTHON]
