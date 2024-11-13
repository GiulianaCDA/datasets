$cartProductModel = new CartProductModel();

// Get the discount for a cart product
$discount = $cartProductModel->getRelatedItemsDiscount($cartProduct);

// Get the title of a cart product
$title = $cartProductModel->getItemTitle($cartProduct);

// Convert a cart product to an API array
$apiArray = $cartProductModel->cartProductToApiArray($cartProduct);
