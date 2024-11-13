
[PYTHON]
def stroke_polygon(points, width, closed_path=False):
    # TODO: Implement this function
    pass
[/PYTHON]

[TESTS]
# Test case 1:
assert stroke_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 2) == [
    {"primitive": "stroke", "coordinates": 5},
    {"primitive": "move", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "close_polygon"},
]

# Test case 2:
assert stroke_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 2, True) == [
    {"primitive": "stroke", "coordinates": 5},
    {"primitive": "move", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "close_polygon"},
]

# Test case 3:
assert stroke_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 0) == [
    {"primitive": "stroke", "coordinates": 4},
    {"primitive": "move", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "line", "coordinates": 2},
    {"primitive": "close_polygon"},
]
[/TESTS]
