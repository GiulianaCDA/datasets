 /////////////////////////////////////
	//
	// Save any access token that is in the current page URL
	// Handle any response solicited through iframe hash tag following an API request
	//
	/////////////////////////////////////

	hello.utils.responseHandler(window, window.opener || window.parent);