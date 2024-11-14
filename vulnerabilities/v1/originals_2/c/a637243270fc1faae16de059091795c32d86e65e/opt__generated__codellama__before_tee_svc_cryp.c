[PYTHON]
#include <stdio.h>

int main()
{
    printf("Hello, World!\n");
    return 0;
}
[/PYTHON]
[TESTS]
# Test case 1:
# Test that the function works with small inputs
def test_small():
    assert hello("Alice") == "Hello, Alice!"

# Test case 2:
# Test that the function works with large inputs
def test_large():
    message = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    assert hello(message) == "Hello, Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum!"

# Test case 3:
# Test that the function raises an error with invalid inputs
def test_invalid():
    try:
        hello(5)
    except TypeError:
        return True
    assert False
[/TESTS]
