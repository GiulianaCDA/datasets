// Create a new instance of the News class
$news = new News($config);

// Get a list of news entries in the current language
$newsEntries = $news->getNews();

// Add a new news entry
$news->addNewsEntry($newsData);

// Update a news entry
$news->updateNewsEntry($newsId, $newsData);

// Delete a news entry
$news->deleteNews($newsId);
