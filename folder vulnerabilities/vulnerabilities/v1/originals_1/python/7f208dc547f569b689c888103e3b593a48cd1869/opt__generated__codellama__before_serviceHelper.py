[PYTHON]
# This script is part of the BackInTime package and is not intended for use outside of it.
import sys
from PyQt5.QtCore import QCoreApplication, Qt

class UdevRules:
    def __init__(self, bus, path):
        self.path = path

    @dbus.service.method(bus_name='net.launchpad.backintime.serviceHelper',
                         interface='net.launchpad.backintime.UdevRules')
    def clean(self, sender=None, conn=None):
        print("Cleaning up previous cached rules.")

if __name__ == '__main__':
    dbus.mainloop.pyqt5.DBusQtMainLoop(set_as_default=True)

    app = QCoreApplication([])

    bus = dbus.SystemBus()
    name = dbus.service.BusName("net.launchpad.backintime.serviceHelper", bus)
    object = UdevRules(bus, '/UdevRules')

    print("Running BIT service.")
    app.exec_()
[/PYTHON]
[GDB]
break main
commands 1
step
end
c
n
cont
info breakpoints
backtrace
list
where
bt
frame
up
down
info localvars
info args
print $rax
print $rbx
print $rcx
print $rdi
print $rsi
print $rdx
print $r8
print $r9
print $r10
print $r11
print $r12
print $r13
print $r14
print $r15
info registers
end
[/GDB]
