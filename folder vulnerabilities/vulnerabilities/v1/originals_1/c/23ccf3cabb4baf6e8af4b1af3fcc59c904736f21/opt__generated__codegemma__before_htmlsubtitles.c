char *html_text = "<font color='red'><b>Hello world!</b></font>";
char *latex_text;

html_to_latex(html_text, &latex_text);

// latex_text will contain the following:
// {\\color{red}}{\\bfseries Hello world!}
