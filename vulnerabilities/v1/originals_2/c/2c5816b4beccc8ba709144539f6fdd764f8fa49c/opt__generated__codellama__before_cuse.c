[PYTHON]
import ctypes as c
class Cuse(c.Structure):
    _fields_ = [
        ("f_op", c.POINTER(c.c_void_p)),
        ("f_flags", c.c_int),
    ]


# Example of how to use it:
libc = c.CDLL('libc.so.6')
prinf = libc.printf
prinf.argtypes = [c.c_char_p]
prinf("Hello world!")

# Example of a simple math function that takes two int arguments and returns an int result
myadd = c.CDLL('./maths.so').c_int(3, 4)
print myadd() # prints 7
[/PYTHON]
