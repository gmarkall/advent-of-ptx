# Advent of Code 2021 solutions in handcoded PTX

Solutions to [Advent of Code 2021](https://adventofcode.com/2021) written in
handcoded PTX. This may become, or be used as, a tutorial on writing PTX in
future.

Some remarks on the solutions:

- Solutions will use a single thread - this is a vehicle to exemplify PTX coding
  rather than parallel programming.
- Each part of each day is implemented as a PTX device function that prints its
  result.
- Execution of the solutions is done using a Python test harness that uses
  [Numba](https://numba.pydata.org) - the harness launches a CUDA kernel (the
  *wrapper* that
  calls the solution device functions.
- The wrapper accepts a pointer to the data and its length - we get these by
  reading the input in in Python, moving it to a CUDA Device Array, and then
  getting the pointer and length using the [CUDA Array
  Interface](https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html).
- [The harness](aoptx.py) contains further comments explaining its operation.

Please do send comments / questions / feedback, either in the
[Issues](https://github.com/gmarkall/advent-of-ptx/issues) or on Twitter:
[@gmarkall](https://twitter.com/gmarkall)


## Invocation

```
$ python aoptx.py <day>
```

To test execution, run the "day 0" solution, which just prints `Hello World`:

```
$ python aoptx.py 0
Hello world
Hello world
```

## Notes on solutions

### Day 0

New concepts / constructs / ideas introduced:

- The test harness
- Calling a PTX device function from Numba
- Printing using `vprintf` inside a kernel

See the [PTX source](day0/part1.ptx).

### Day 1

Notable new concepts / constructs / ideas introduced:

- Using local memory to create a stack and address it through local and generic
  address spaces
- Labels
- Branches
- Tests and predicated execution
- Loading parameters into registers

See the [Part 1 solution](day1/part1.ptx) for extensive comments. [Part
2](day1/part2.ptx) also has some comments, but is less thorough - it is just a
slight modification of the code for Part 1.

### Day 2

Notable new concepts:

- Logical operations
- Multiplication (low bits only)

These solutions are a little tidier and better commented than the Day 1
solutions. Refer to [Part 1](day2/part1.ptx) - Part 2 is almost exactly the
same.

## Useful references

- [PTX ISA
  Specification](https://docs.nvidia.com/cuda/parallel-thread-execution/): A
  manual documenting PTX assembly language.
- [Compute Sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/): Like
  Valgrind, but for the GPU. It works even for handcoded PTX, and is very useful
  for debugging.
