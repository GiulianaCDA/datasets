// Resolve the expression "1 + 2" to an integer value.
int value;
ExprResolveInteger(ctx, expr, &value);

// Resolve the expression "/mask" to a bit mask.
unsigned int mask;
ExprResolveMask(ctx, expr, &mask);
