# Advent of code 2021
#
# PTX execution harness using Numba
#
# Usage: aoptx.py <Solution PTX>
#
# The PTX file must contain a device function called `solution` that accepts no
# arguments and prints out the solution.

from numba import cuda, config
import os
import sys


def run(filename):
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    solution = cuda.declare_device('solution', 'void()')
    path = os.path.abspath(filename)

    @cuda.jit('void()', link=[path])
    def wrapper():
        solution()

    wrapper[1, 1]()
    cuda.synchronize()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: aoptx.py <Solution PTX>")
        sys.exit(1)

    run(sys.argv[1])
