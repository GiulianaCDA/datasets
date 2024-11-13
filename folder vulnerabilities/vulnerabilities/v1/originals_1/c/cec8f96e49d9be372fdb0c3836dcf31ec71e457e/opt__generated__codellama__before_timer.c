 
[PYTHON]
import argparse
import datetime
import serial
import time

SERIAL_DEVICE = '/dev/ttyUSB0'
BAUD_RATE = 9600
TIMEOUT = 2

def main():
    # Initialize the serial connection
    with serial.Serial(SERIAL_DEVICE, BAUD_RATE, timeout=TIMEOUT) as port:
        while True:
            line = port.readline()
            if line != '':
                print(datetime.datetime.now().strftime('%H:%M:%S') + ": " + line.decode().strip())
            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=SERIAL_DEVICE, help="Serial device")
    parser.add_argument("--baudrate", type=int, default=BAUD_RATE, help="Baud rate")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
    args = parser.parse_args()
    
    main()
[/PYTHON]
[C++]
int main() {
   // your code goes here
   return 0;
}
[/C++]
