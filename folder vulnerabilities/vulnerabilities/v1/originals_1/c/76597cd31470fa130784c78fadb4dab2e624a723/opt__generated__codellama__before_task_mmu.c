 
[PYTHON]
import sys
from pathlib import Path

def get_absolute_path(path):
    return str(Path(path).resolve())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} PROGRAM".format(sys.argv[0]))
        sys.exit(1)
    path = get_absolute_path(sys.argv[1])
    with open(path, "rb") as f:
        data = f.read()
        for i in range(len(data)):
            if data[i] == 0:
                print("\\x{:02x}".format(data[i]))
            else:
                print(r"{}".format(data[i]), end="")
[/PYTHON]
 [PERL]
#! /usr/bin/env perl
use strict;
use warnings;
use feature qw{ say };
use autodie;

my $path = shift // die "Please provide a file path\n";

open my $fh, '<:raw', $path;

while (read($fh, my $buffer, 1)) {
    my $hex_repr = sprintf '%x', ord $buffer;
    if ($hex_repr =~ /^[0-9a-fA-F]+$/) {
        say "\\x{$hex_repr}";
    } else {
        say $buffer;
    }
}
close($fh);
[/PERL]
 [PHP]
<?php
$path = $argv[1] ?? '/tmp/file';
$absolutePath = realpath($path);
if (!$absolutePath) {
  exit("Error: Could not find file {$path}");
}
$contents = file_get_contents($absolutePath);
foreach (str_split($contents)) {
    if (ord($char) < 32 || ord($char) > 126) {
        echo "\\x" . bin2hex($char);
    } else {
        echo $char;
    }
}
?>
[/PHP]
 [JAVA]
import java.io.*;
import java.util.Scanner;

public class Main {
  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.out.println("Usage: hexdump file");
      return;
    }

    FileInputStream fis = new FileInputStream(args[0]);
    Scanner scanner = new Scanner(fis);

    while (scanner.hasNext()) {
      byte nextByte = scanner.nextByte();
      if (nextByte >= 32 && nextByte <= 126) {
        System.out.print((char) nextByte);
      } else {
        String hexString = "\\x" + Integer.toHexString(nextByte).toUpperCase();
        System.out.print(hexString);
      }
    }

    scanner.close();
  }
}
[/JAVA]
 [BASH]
# Usage: ./solution.sh file

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 file"
  exit 1
fi

path="$1"
absolute_path="$(realpath "$path")"
if [[ ! -f "$absolute_path" ]]; then
  echo "Error: Could not find file \"$path\"" >&2
  exit 1
fi

echo "$(cat "$absolute_path" | od -An -t x1 | tr -d '\n')"
[/BASH]
 [PYTHON3]
#!/usr/bin/env python3
import os.path
import sys

def hexdump(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        for i, c in enumerate(data):
            if 0x20 <= c < 0x7f and not chr(c).isprintable():
                print(r'\x{:02x}'.format(c), end='')
            else:
                print(chr(c), end='')
    print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} FILENAME'.format(sys.argv[0]))
        sys.exit(1)
    filename = os.path.abspath(sys.argv[1])
    hexdump(filename)
[/PYTHON3]
