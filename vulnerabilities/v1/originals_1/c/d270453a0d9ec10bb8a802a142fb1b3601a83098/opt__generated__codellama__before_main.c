[PYTHON]
def get_unique_elements(my_list):
    return list(set([x for x in my_list if my_list.count(x) == 1]))
[/PYTHON]
[TESTS]
# Test case 1:
assert get_unique_elements([]) == []
# Test case 2:
assert get_unique_elements([1]) == [1]
# Test case 3:
assert get_unique_elements([1, 2, 3, 2, 1]) == [1, 2, 3]
[/TESTS]
