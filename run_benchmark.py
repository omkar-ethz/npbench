import argparse
import os

from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--benchmark",
                        type=str,
                        nargs="?",
                        required=True)
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="numpy")
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper'],
                        nargs="?",
                        default='S')
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t",
                        "--timeout",
                        type=float,
                        nargs="?",
                        default=200.0)
    parser.add_argument("-s",
                        "--save-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    parser.add_argument("-l",
                        "--load-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    parser.add_argument("-n",
                        "--num-threads",
                        type=int,
                        nargs="?",
                        default=1)
    args = vars(parser.parse_args())

    os.environ['OMP_NUM_THREADS'] = str(args["num_threads"])
    print("num_threads=", str(args["num_threads"]))
    # set environment variable openmp_tasking
    if args["framework"] == "dace_cpu_tasking":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "true"
    elif args["framework"] == "dace_cpu":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
    elif args["framework"] == "dace_cpu_dynamic_schedule":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
        os.environ['DACE_compiler_cpu_openmp_dynamic_schedule'] = "true"

    # print(args)

    bench = Benchmark(args["benchmark"])
    frmwrk = generate_framework(args["framework"],
                                save_strict=args["save_strict_sdfg"],
                                load_strict=args["load_strict_sdfg"])
    numpy = generate_framework("numpy")
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run(args["preset"], args["validate"], args["repeat"], args["timeout"], num_threads=args["num_threads"])
