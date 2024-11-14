[PYTHON]
import logging
from datetime import datetime

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer

class MainHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")

if __name__ == "__main__":
    app = Application([
        (r"/", MainHandler),
    ])

    http_server = HTTPServer(app)
    http_server.listen(8888)
    IOLoop.current().start()
[/PYTHON]
[JAVA]
import java.io.*;
import java.util.Date;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        System.out.print("Enter a number: ");
        int number = Integer.parseInt(reader.readLine());

        Date date = new Date();
        if (date.getTime() % number == 0) {
            System.out.println("The current time is divisible by the given number.");
        } else {
            System.out.println("The current time is not divisible by the given number.");
        }
    }
}
[/JAVA]
[PHP]
<?php
$number = readline("Enter a number: ");
if (date('i') % $number == 0) {
    echo "The current time is divisible by the given number.";
} else {
    echo "The current time is not divisible by the given number.";
}
?>
[/PHP]
[C#]
using System;
using System.IO;

class Solution
{
    static void Main(string[] args)
    {
        var reader = new StreamReader(Console.OpenStandardInput());
        int n = int.Parse(reader.ReadLine());
        if (DateTime.Now.Minute % n == 0)
        {
            Console.WriteLine("The current time is divisible by the given number.");
        }
        else
        {
            Console.WriteLine("The current time is not divisible by the given number.");
        }
    }
}
[/C#]
[RU]
import java.io.*;
import java.util.Date;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        System.out.print("Введите число: ");
        int number = Integer.parseInt(reader.readLine());

        Date date = new Date();
        if (date.getMinutes() % number == 0) {
            System.out.println("Текущее время делится на заданное число.");
        } else {
            System.out.println("Текущее время не делится на заданное число.");
        }
    }
}
[/RU]
