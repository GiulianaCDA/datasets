[PYTHON]
import sys
from datetime import datetime

def print_usage():
    print("Usage: python", sys.argv[0], "<year> <month> <day>")
    print("Print a happy birthday message to standard output.")
    print("Example: python", sys.argv[0], "1985 04 12")

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def days_in_month(year, month):
    if month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 28 if not is_leap_year(year) else 29
    else:
        return 31

def calculate_day(year, month, day):
    total_days = 0
    for y in range(year, month + 1):
        total_days += (365 if not is_leap_year(y) else 366)
    for m in range(1, month):
        total_days += days_in_month(year, m)
    return total_days + day

def happy_birthday(year, month, day):
    today = datetime.today()
    diff = calculate_day(today.year, today.month, today.day) - calculate_day(year, month, day)
    if diff == 0:
        print("Happy Birthday! You are", diff, "days old.")
    elif diff > 0:
        print("You are", diff, "days older.")
    else:
        print("You are", abs(diff), "days younger.")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)
    happy_birthday(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
[/PYTHON]
