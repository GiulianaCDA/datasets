$link = new Link();

// Set data from editmode
$link->setDataFromEditmode([
    'path' => '/my-link',
    'linktype' => 'internal',
]);

// Get the link class
echo $link->getClass(); // Output: ''

// Get the link attributes
echo $link->getAttributes(); // Output: ''

// Get the link path
echo $link->getPath(); // Output: '/my-link'

// Get the link type
echo $link->getLinkType(); // Output: 'internal'
