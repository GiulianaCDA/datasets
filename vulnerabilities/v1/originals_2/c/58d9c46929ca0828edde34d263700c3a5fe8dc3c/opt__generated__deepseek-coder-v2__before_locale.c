void LocaleUpper(char *string)
{
  register char *q;
  assert(string != (char *) NULL); // Ensure the input string is valid
  for (q = string; *q != '\0'; q++)
    *q = (char) LocaleUppercase((int) *q); // Convert each character using LocaleUppercase()
}
