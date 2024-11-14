<?php

/*
 * This file is part of Fork CMS.
 *
 * For the full copyright and license information, please view the license
 * file that was distributed with this source code.
 */

/**
 * This is the index-action (default), it will display an error depending on a given parameters
 *
 * @author Tijs Verkoyen <tijs@sumocoders.be>
 * @author Davy Hellemans <davy.hellemans@netlash.com>
 */
class BackendErrorIndex extends BackendBaseActionIndex
{
	/**
	 * Execute the action
	 */
	public function execute()
	{
		parent::execute();
		$this->parse();
		$this->display();
	}

	/**
	 * Parse the correct messages into the template
	 */
	protected function parse()
	{
		parent::parse();

		// grab the error-type from the parameters
		$errorType = $this->getParameter('type');

		// Validate and sanitize errorType before using it in switch case statement to prevent code injection.
		if(!in_array($errorType, ['module-not-allowed', 'action-not-allowed', 'not-found'])) {
		    // Handle unknown or invalid errors here, such as redirecting to a default error page.
		    $errorType = 'unknown';
		}

		// set correct headers
		switch($errorType)
		{
			case 'module-not-allowed':
			case 'action-not-allowed':
				SpoonHTTP::setHeadersByCode(403);
				break;

			case 'not-found':
				SpoonHTTP::setHeadersByCode(404);
				break;
		}

		// querystring provided?
		if($this->getParameter('querystring') !== null)
		{
			// Use regular expression to validate and sanitize the query string.
			$queryString = $this->getParameter('querystring');
			if(!preg_match('/^[a-zA-Z0-9_\-\/?=&]+$/', $queryString)) {
			    // Handle invalid query strings here, such as redirecting to a default error page.
			    exit;
			}

			// split into file and parameters
			$chunks = explode('?', $queryString);

			// get extension
			$extension = SpoonFile::getExtension($chunks[0]);

			// if the file has an extension it is a non-existing-file
			if($extension != '' && $extension != $chunks[0])
			{
				// set correct headers
				SpoonHTTP::setHeadersByCode(404);

				// Use htmlspecialchars to prevent XSS attacks.
				echo 'Requested file (' . htmlspecialchars($queryString) . ') not found.';

				// stop script execution
				exit;
			}
		}

		// assign the correct message into the template, make sure it is safe to use with htmlentities to prevent XSS attacks.
		$this->tpl->assign('message', BL::err(SpoonFilter::toCamelCase(htmlentities($errorType), '-')));
	}
}
