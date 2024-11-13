 
[PYTHON]
import os
import platform
import random
import re
import socket
import subprocess
import sys
from math import ceil

class Utils:
    @staticmethod
    def get_os_type():
        return platform.system().lower()

    @staticmethod
    def is_windows():
        return Utils.get_os_type() == 'windows'

    @staticmethod
    def is_linux():
        return Utils.get_os_type() == 'linux'

    @staticmethod
    def is_mac():
        return Utils.get_os_type() == 'darwin'

    @staticmethod
    def is_64bits():
        return sys.maxsize > 2 ** 32

    @staticmethod
    def get_home_dir():
        return os.path.expanduser('~')

    @staticmethod
    def is_venv_active():
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    @staticmethod
    def get_username():
        return os.getlogin()

    @staticmethod
    def exec_command(cmd):
        return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

    @staticmethod
    def get_ip_addresses():
        addresses = []
        for interface in socket.getaddrinfo(socket.gethostname(), None):
            if interface[0] != 2: # IPv6
                continue
            addr, _port, _, _, _, _ = interface[4]
            addresses.append(addr)
        return addresses

    @staticmethod
    def get_mac_addresses():
        interfaces = ['eth0', 'en0']
        macs = {}
        for interface in interfaces:
            mac = Utils.exec_command('ifconfig %s | grep ether' % interface).split(' ')[1]
            if mac != '' and mac not in macs:
                macs[mac] = interface
        return list(macs)

    @staticmethod
    def get_total_memory():
        total_mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        return int(total_mem / (1024 ** 2))

    @staticmethod
    def get_available_memory():
        available_bytes = os.statvfs('/').f_bsize * os.statvfs('/').f_bavail
        return int(round(available_bytes / (1024 ** 2)))

    @staticmethod
    def get_total_disk_space():
        total_bytes = os.path.getsize('%s' % os.sep) * (1024 ** 2) # Convert to MB
        return int(total_bytes / (1024 ** 2))

    @staticmethod
    def get_available_disk_space():
        total_bytes = os.path.getsize('%s' % os.sep) * (1024 ** 2) # Convert to MB
        available_bytes = os.statvfs('/').f_bsize * os.statvfs('/').f_bavail
        return int(round(available_bytes / (1024 ** 2)))

    @staticmethod
    def get_used_disk_space():
        total_bytes = os.path.getsize('%s' % os.sep) * (1024 ** 2) # Convert to MB
        available_bytes = os.statvfs('/').f_bsize * os.statvfs('/').f_bavail
        return int(round((total_bytes - available_bytes) / (1024 ** 2)))

    @staticmethod
    def get_cpu_count():
        return os.cpu_count()

    @staticmethod
    def get_virtual_memory():
        vm = psutil.virtual_memory()
        return (vm.total / (1024 ** 2), vm.available / (1024 ** 2))

    @staticmethod
    def is_admin():
        if Utils.is_windows():
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0

    @staticmethod
    def get_os_info():
        if Utils.is_windows():
            release = platform.release()
            version = platform.version()
            arch = '32-bit' if platform.architecture()[0] == '32bit' else '64-bit'
            return '%s %s %s' % (release, version, arch)
        else:
            name = platform.linux_distribution()[0]
            release = platform.linux_distribution()[1]
            arch = '32-bit' if platform.architecture()[0] == '32bit' else '64-bit'
            return '%s %s %s' % (name, release, arch)

    @staticmethod
    def get_public_ip():
        ip = Utils.exec_command('curl -s http://checkip.dyndns.org/ | grep -o "Address: [0-9\.]*" | cut -d " " -f 2')
        if not ip:
            return 'N/A'
        else:
            return ip

    @staticmethod
    def get_local_ip():
        addresses = []
        for interface in socket.getaddrinfo(socket.gethostname(), None):
            if interface[0] != 2: # IPv6
                continue
            addr, _port, _, _, _, _ = interface[4]
            addresses.append(addr)
        return addresses

    @staticmethod
    def is_port_open(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5) # Timeout in 1/2 seconds
            result = sock.connect_ex((host, port))
            if result == 0:
                return True
            else:
                return False
        except socket.error as e:
            print('Socket error', str(e))
            return False
        finally:
            sock.close()

    @staticmethod
    def get_random_string(length):
        letters = string.ascii_lowercase + string.digits
        result_str = ''.join((random.choice(letters) for i in range(length)))
        return result_str

    @staticmethod
    def is_ipv6_supported():
        try:
            socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
            return True
        except socket.error:
            return False

    @staticmethod
    def get_mac_addresses(interface):
        macs = {}
        if Utils.is_windows():
            cmd = 'getmac /V %s' % interface
            raw_data = subprocess.check_output(cmd, shell=True)
            for row in raw_data.split('\r\n'):
                data = row.split('  ')
                if len(data) > 2 and 'Physical' in data[0] and 'Ethernet' in data[1]:
                    macs[data[2]] = interface
        else:
            cmd = "ifconfig %s | grep ether" % interface
            raw_data = subprocess.check_output(cmd, shell=True)
            for row in raw_data.split('\n'):
                data = row.split(' ')
                if len(data) > 1 and 'inet' not in data[0]:
                    macs[data[1]] = interface
        return list(macs)

if __name__ == "__main__":
    print(Utils.get_os_type())
    print(Utils.is_windows())
    print(Utils.is_linux())
    print(Utils.is_mac())
    print(Utils.is_64bits())
    print(Utils.get_home_dir())
    print(Utils.is_venv_active())
    print(Utils.get_username())
    print(Utils.exec_command('echo $SHELL'))
    print(Utils.get_ip_addresses())
    print(Utils.get_mac_addresses('eth0'))
    print(Utils.is_port_open('google.com', 80))
    print(Utils.get_random_string(16))
    print(Utils.is_ipv6_supported())
    print(Utils.get_os_info())
    print(Utils.get_public_ip())
    print(Utils.get_local_ip())
```