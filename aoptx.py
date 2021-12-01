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
    part1_path = os.path.join(day_dir, 'part1.ptx')
    part2_path = os.path.join(day_dir, 'part2.ptx')
    input_path = os.path.join(day_dir, 'input')

    with open(input_path, 'r') as f:
        values = np.asarray([int(line.strip()) for line in f.readlines()])

    device_values = cuda.to_device(values)
    data_ptr = device_values.__cuda_array_interface__['data'][0]
    data_len = device_values.__cuda_array_interface__['shape'][0]

    part1 = cuda.declare_device('part1', 'void(uint64, uint64)')
    part2 = cuda.declare_device('part2', 'void(uint64, uint64)')

    @cuda.jit('void(uint64, uint64)', link=[part1_path, part2_path])
    def wrapper(data_ptr, data_len):
        part1(data_ptr, data_len)
        part2(data_ptr, data_len)

    wrapper[1, 1](data_ptr, data_len)
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
