from msml.core import *

ctx = Context()
print('--------- Hardware Information Begin---------')
print(f'OS name: {ctx.os_name}')
print(f'CPU name: {ctx.cpu_name}')
print(f'CPU virtual cores: {ctx.cpu_virtual_cores}')
print(f'CPU physical cores: {ctx.cpu_physical_cores}')
print(f'CPU sockets: {ctx.cpu_sockets}')
print(f'Physical memory total: {ctx.physical_memory_total / (1 << 30)} GiB')
print(f'Physical memory free: {ctx.physical_memory_free / (1 << 30)} GiB')
print(f'Physical memory used: {ctx.physical_memory_used / (1 << 30)} GiB')
print('--------- Hardware Information End---------')
