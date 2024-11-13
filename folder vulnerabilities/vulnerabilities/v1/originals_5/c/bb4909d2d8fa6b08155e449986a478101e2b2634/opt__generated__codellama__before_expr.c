[PYTHON]
def expr_resolve_mask(ctx: 'Context', expr: 'ExprDef') -> bool:
    # Check if the expression is an integer literal.
    if expr.expr.op == EXPR_IDENT:
        name = xkb_atom_text(ctx, expr.ident.ident)
        val = int(name[1:])
        if val < 0 or val >= 256:
            return False
        mask_rtrn = 1 << val
        return True
    # Check if the expression is an unary operation.
    elif expr.expr.op == EXPR_UNARY_PLUS or \
            expr.expr.op == EXPR_NEGATE:
        left = expr.unary.child
        if not expr_resolve_mask(ctx, left):
            return False
        mask_rtrn = ~mask_rtrn
        return True
    # Check if the expression is a binary operation.
    elif expr.expr.op == EXPR_ADD or \
            expr.expr.op == EXPR_DIVIDE or \
            expr.expr.op == EXPR_MULTIPLY:
        left = expr.binary.left
        right = expr.binary.right
        if not expr_resolve_mask(ctx, left):
            return False
        if not expr_resolve_mask(ctx, right):
            return False
        mask_rtrn = mask_rtrn + mask_rtrn
        return True
    # Check if the expression is an assignment operation.
    elif expr.expr.op == EXPR_ASSIGN:
        log_wsgo(ctx, "Assignment operator not implemented yet\n")
        return False
    # Check if the expression is a bitwise operation.
    elif expr.expr.op == EXPR_BITWISE_AND or \
            expr.expr.op == EXPR_BITWISE_OR:
        left = expr.binary.left
        right = expr.binary.right
        if not expr_resolve_mask(ctx, left):
            return False
        if not expr_resolve_mask(ctx, right):
            return False
        mask_rtrn = mask_rtrn & mask_rtrn
        return True
    # Check if the expression is a logical operation.
    elif expr.expr.op == EXPR_LOGICAL_AND or \
            expr.expr.op == EXPR_LOGICAL_OR:
        left = expr.binary.left
        right = expr.binary.right
        if not expr_resolve_mask(ctx, left):
            return False
        if not expr_resolve_mask(ctx, right):
            return False
        mask_rtrn = mask_rtrn & mask_rtrn
        return True
    # Check if the expression is a negation operation.
    elif expr.expr.op == EXPR_INVERT:
        left = expr.unary.child
        if not expr_resolve_mask(ctx, left):
            return False
        mask_rtrn = ~mask_rtrn
        return True
    # Check if the expression is a key code operation.
    elif expr.expr.op == EXPR_KEYCODE:
        log_wsgo(ctx, "Keycode operator not implemented yet\n")
        return False
    # Check if the expression is an unsupported operation.
    else:
        log_wsgo(ctx, "Unknown operator %d in ResolveMask\n",
                expr.expr.op)
        return False
[/PYTHON]
