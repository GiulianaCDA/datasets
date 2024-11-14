[PYTHON]
import re

while True:
    line = input()
    if not line:
        break
    line = line.strip()
    print(re.sub(r'[^a-zA-Z ]', '', line))
[/PYTHON]
* * *
[INST:PROMPT]
What is the difference between a set and a map in Python?
[/INST:ANSWER]
In Python, a set is an unordered collection of unique elements (also called hashable items), while a map is an unordered collection of key-value pairs.

Here are some key differences between sets and maps:

1. Elements in a set must be hashable, whereas keys in a map can be any type of object that supports the `__hash__()` method (including other types that are not hashable).
2. A set does not allow duplicate elements (i.e., adding an element that is already present in the set has no effect), while a map allows duplicate key-value pairs (i.e., adding a new pair with a key that is already present in the map overwrites the previous value associated with that key).
3. Sets are mutable, meaning that you can add and remove elements after the set is created, whereas maps are immutable once they are created (i.e., you cannot add or remove elements from a map after it has been initialized).
4. Sets have built-in methods for testing membership (e.g., `set.contains()`), iteration (e.g., `for item in set: ...`), and intersection/union with other sets (e.g., `set.intersection()`, `set.union()`), whereas maps have methods for accessing values associated with specific keys (e.g., `map[key]`, `map.get(key)`), iteration (e.g., `for key, value in map.items(): ...`), and intersection/union with other maps (e.g., `map.intersection()`, `map.union()`).
5. Sets are typically implemented as hash tables, which can be more efficient for certain types of operations (e.g., testing membership or calculating the union of two sets), whereas maps are typically implemented as trees or arrays, which can be more efficient for other types of operations (e.g., iterating over all keys in a map).

In summary, while both sets and maps can be used to store collections of unique elements or key-value pairs, they have different use cases and characteristics that make them more suitable for certain types of applications.