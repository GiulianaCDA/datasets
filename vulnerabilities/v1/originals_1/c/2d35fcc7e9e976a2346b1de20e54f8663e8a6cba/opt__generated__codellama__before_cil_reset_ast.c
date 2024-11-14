[PYTHON]
def cil_reset_ast(current):
    """Reset the AST by calling cil_reset_node for each node in the tree.

    Args:
        current: root of the AST 

    Returns:
        SEPOL_OK on success
        SEPOL_ERR on failure
    """
    if not current:
        return SEPOL_ERR

    rc = cil_tree_walk(current, cil_reset_node, None, None, None)
    if rc != SEPOL_OK:
        cil_log("Failed to reset AST\n")
        return SEPOL_ERR

    return SEPOL_OK
[/PYTHON]
