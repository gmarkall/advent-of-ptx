# Advent of code 2021
#
# PTX execution harness using Numba
#
# Usage: aoptx.py <day>

from numba import cuda, config
import numpy as np
import os
import sys


def run(day):
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))
    day_dir = os.path.join(base_dir, f'day{day}')
    ptx_path = os.path.join(day_dir, 'solution.ptx')
    input_path = os.path.join(day_dir, 'input')

    with open(input_path, 'r') as f:
        values = np.asarray([int(line.strip()) for line in f.readlines()])

    device_values = cuda.to_device(values)
    data_ptr = device_values.__cuda_array_interface__['data'][0]

    solution = cuda.declare_device('solution', 'void(uint64)')

    @cuda.jit('void(uint64)', link=[ptx_path])
    def wrapper(ptr):
        solution(ptr)

    wrapper[1, 1](data_ptr)
    cuda.synchronize()


def usage():
    print("Usage: aoptx.py <day>")
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()

    try:
        day = int(sys.argv[1])
    except ValueError:
        usage()

    run(day)
