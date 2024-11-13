[PYTHON]
import sys

def print_debug(message):
    if "--debug" in sys.argv:
        print(message)

def greet(name):
    print_debug("Greetings, " + name + "!")
    return "Hello, " + name + "!"

def main():
    print_debug("Starting application.")
    name = input("What is your name? ")
    result = greet(name)
    print("Result: " + result)

if __name__ == "__main__":
    main()
[/PYTHON]
[JAVA]
import java.util.Scanner;

public class HelloWorld {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Please enter your name: ");
        String name = scanner.nextLine();
        System.out.println("Hello, " + name);
    }
}
[/JAVA]
[C++]
#include <iostream>
using namespace std;

int main() {
  cout << "Hello, World!" << endl;
}
[/C++]
[PHP]
<?php
  phpinfo();
?>
[/PHP]
[PERL]
use strict;
use warnings;

my $name = 'Mike';
print "Hello, $name!\n";
[/PERL]
[JAVASCRIPT]
console.log("Hello, World!");
[/JAVASCRIPT]
[SCALA]
object HelloWorld {
  def main(args: Array[String]) {
    println("Hello, World!")
  }
}
[/SCALA]
[BASH]
#!/bin/bash
echo "Hello, World!"
[/BASH]
