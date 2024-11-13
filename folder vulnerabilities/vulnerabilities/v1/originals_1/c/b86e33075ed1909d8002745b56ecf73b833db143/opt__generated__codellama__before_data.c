
[PYTHON]
import os

# Write a function that takes two strings as input and returns the count of characters 
# in the first string that are not present in the second string.
def solve(s1, s2):
    count = 0
    for char in s1:
        if char not in s2:
            count += 1
    return count

# Test case 1:
assert solve("abc", "") == 3

# Test case 2:
assert solve("abc", "c") == 2

# Test case 3:
assert solve("abc", "ac") == 1

# Test case 4:
assert solve("abc", "abc") == 0

# Test case 5:
assert solve("abc", "abcd") == 0

# Test case 6:
assert solve("abcd", "abc") == 1

# Test case 7:
assert solve("abcd", "abce") == 0

# Test case 8:
assert solve("abcd", "bcde") == 2

# Test case 9:
assert solve("abcd", "abd") == 1

# Test case 10:
assert solve("abcd", "abe") == 1
 [/PYTHON]
[SWIFT]
import UIKit

class MyViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        // Add a tap gesture recognizer to the view
        let tapGesture = UITapGestureRecognizer(target: self, action: "handleTap:")
        view.addGestureRecognizer(tapGesture)
    }

    func handleTap(gestureRecognize: UIGestureRecognizer) {
        // Handle the tap gesture recognizer's tap event
        let point = gestureRecognize.locationInView(view)
        let touchPoint = view.convertPoint(point, toView: nil)

        print("x: \(touchPoint.x), y: \(touchPoint.y)")
    }
}
[/SWIFT]
[CHEF]
# Create a new recipe for 'pizza'
knife cookbook create pizza

# Add the 'pizza' recipe to the run list of the 'example' node
echo "pizza" >> ~/.chef/nodes/example.json

# Apply the cookbook to the 'example' node using Knife
knife apply -n example

# Remove the 'pizza' recipe from the run list of the 'example' node
sed -i '' '/pizza/d' ~/.chef/nodes/example.json

# Apply the cookbook to the 'example' node using Knife
knife apply -n example

# View the current run list for the 'example' node
knife node show example | grep "run_list"
[/CHEF]
