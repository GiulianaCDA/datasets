  [PYTHON]
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value
[/PYTHON]
* [TESTS]
# Test case 1: Counter class
# Create a Counter object and check its initial value is 0
counter = Counter()
assert counter.get_value() == 0

# Increment the counter by 1
counter.increment()
assert counter.get_value() == 1

# Check that the counter still has the same value after a second increment
counter.increment()
assert counter.get_value() == 2

# Reset the counter to 0
counter.reset()
assert counter.get_value() == 0
[/TESTS]
