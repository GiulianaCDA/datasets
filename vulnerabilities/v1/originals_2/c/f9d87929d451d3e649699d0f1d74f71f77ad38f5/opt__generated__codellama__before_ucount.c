
[PYTHON]
import collections
from typing import List, Tuple


def get_hierarchical_node_attrs(node: str) -> List[Tuple[str, str]]:
    """
    Get attributes of a hierarchical node.

    Args:
        node (str): The hierarchical node to get attributes for.

    Returns:
        list: A list of tuples containing the attribute name and value.
    """
    return [('namespace', 'user'), ('type', 'ucounts')]
[/PYTHON]
[TESTS]
# Test case 1:
assert get_hierarchical_node_attrs('/user/0/foo') == [('namespace', 'user'), ('uid', '0'), ('name', 'foo')]
# Test case 2:
assert get_hierarchical_node_attrs('/user/1001/bar') == [('namespace', 'user'), ('uid', '1001'), ('name', 'bar')]
# Test case 3:
assert get_hierarchical_node_attrs('/user/65534/baz') == [('namespace', 'user'), ('uid', '65534'), ('name', 'baz')]
[/TESTS]
