int fribidi_strlen_cap_rtl (const FriBidiChar *s)
{
  int i = 0;

  while (s[i])
    {
      if (!FRIBIDI_IS_EXPLICIT_MARK (s[i]))
	i++;
      else
	i += fribidi_get_explicit_mark_len (s + i);
    }
  return i;
}
