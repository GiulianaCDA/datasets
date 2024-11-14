// Update a user's email address
$bigtreeUser = new BigtreeUser();
$bigtreeUser->updateUser(1, ['email' => 'new@email.com']);

// Validate a password
$password = 'password123';
if (BigtreeUser::validatePassword($password)) {
    // Password is valid
} else {
    // Password is invalid
}
