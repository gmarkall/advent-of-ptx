# Advent of code 2021
#
# PTX execution harness using Numba
#
# Usage: aoptx.py <day>

from numba import cuda, config
import numpy as np
import os
import sys


# Each day has input in a different format, so we define a function to load
# the input for each day.

def day_0_input(path):
    # Day 0 has no input, so just return a dummy value
    return np.zeros(1, dtype=np.uint64)


def day_1_input(path):
    # Day 1 is a list of integers
    with open(path, 'r') as f:
        return np.asarray([int(line.strip()) for line in f.readlines()])


def day_2_input(path):
    # Day 2 is a list of pairs "<command> <value>".
    # Encode as a 64 bit integer with the upper 32 bits holding the command and
    # the lower holding the value. Commands are:
    #
    # 0: Forward
    # 1: Down
    # 2: Up

    with open(path, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines()]

    values = np.zeros(len(pairs), dtype=np.uint64)

    cmds = {
        'forward': 0,
        'down': 1,
        'up': 2
    }

    for i, (cmd, val) in enumerate(pairs):
        values[i] = (cmds[cmd] << 32) | int(val)

    return values


INPUT_LOADERS = (
    day_0_input,
    day_1_input,
    day_2_input,
)


def run(day):
    # Disable low occupancy warnings - because we run only with a single block
    # / thread (because we're not really interested in high performance for
    # these solutions, but convenience for launching a handcoded PTX) Numba
    # will warn us. Setting this elides that warning.
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    # Determine paths of solutions and input for the current day
    base_dir = os.path.dirname(os.path.abspath(__file__))
    day_dir = os.path.join(base_dir, f'day{day}')
    part1_path = os.path.join(day_dir, 'part1.ptx')
    part2_path = os.path.join(day_dir, 'part2.ptx')
    input_path = os.path.join(day_dir, 'input')

    # Read in today's input
    values = INPUT_LOADERS[day](input_path)

    # Move the data to the GPU and get a pointer to the data and the number of
    # entries
    device_values = cuda.to_device(values)
    data_ptr = device_values.__cuda_array_interface__['data'][0]
    data_len = device_values.__cuda_array_interface__['shape'][0]

    # "Forward declaration" of device functions in the solution PTX - these are
    # needed so that Numba knows how to resolve calls to these functions. The
    # declarations here specify functions that accept a pair of uint64 values
    # (the pointer to the data, and the number of items), and return nothing.
    part1 = cuda.declare_device('part1', 'void(uint64, uint64)')
    part2 = cuda.declare_device('part2', 'void(uint64, uint64)')

    # Kernel to call the solutions. It accepts the data pointer and length, and
    # passes them to the solution functions. It does nothing else because the
    # solutions are expected to print their result.
    #
    # The link kwarg is used to instruct Numba to link the PTX for part 1 and
    # part 2 when linking the kernel.
    @cuda.jit('void(uint64, uint64)', link=[part1_path, part2_path])
    def wrapper(data_ptr, data_len):
        part1(data_ptr, data_len)
        part2(data_ptr, data_len)

    # Launch the kernel with a single thread, because we only expect solutions
    # to need a single thread at this point (perhaps this will change for
    # future solutions).
    wrapper[1, 1](data_ptr, data_len)

    # Synchronize to prevent the program exiting before printed solutions get
    # flushed - printing inside kernels can occur asynchronously WRT to the
    # main thread.
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
