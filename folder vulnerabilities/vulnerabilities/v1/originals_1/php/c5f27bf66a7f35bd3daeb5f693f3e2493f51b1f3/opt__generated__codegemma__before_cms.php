$html = "This is an internal page link: ipl://my-page";
$processedHtml = replaceInternalPageLinks($html);

echo $processedHtml; // Output: This is an internal page link: /my-page
