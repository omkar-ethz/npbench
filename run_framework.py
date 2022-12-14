import argparse
import os
import pathlib
import sys

from multiprocessing import Process
from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)


def run_benchmark(benchname, fname, preset, validate, repeat, timeout,
                  ignore_errors, save_strict, load_strict, num_threads):
    frmwrk = generate_framework(fname, save_strict, load_strict)
    numpy = generate_framework("numpy")
    bench = Benchmark(benchname)
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run(preset, validate, repeat, timeout, ignore_errors, num_threads=num_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--ignore-errors",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
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
    # set environment variable openmp_tasking
    if args["framework"] == "dace_cpu_tasking":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "true"
    elif args["framework"] == "dace_cpu":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
    elif args["framework"] == "dace_cpu_dynamic_schedule":
        os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
        os.environ['DACE_compiler_cpu_openmp_dynamic_schedule'] = "true"

    parent_folder = pathlib.Path(__file__).parent.absolute()
    bench_dir = parent_folder.joinpath("bench_info")
    pathlist = pathlib.Path(bench_dir).rglob('*.json')
    benchnames = [os.path.basename(path)[:-5] for path in pathlist]
    benchnames.sort()
    
    #these benchmarks timeout
    benchnames.remove('azimint_naive') #times out for dace_cpu_tasking on `S`
    benchnames.remove('durbin') #times out for dace_cpu on `paper`

    failed = []
    for benchname in benchnames:
        p = Process(target=run_benchmark,
                    args=(benchname, args["framework"], args["preset"],
                          args["validate"], args["repeat"], args["timeout"],
                          args["ignore_errors"], args["save_strict_sdfg"],
                          args["load_strict_sdfg"], args["num_threads"]))
        p.start()
        p.join()
        exit_code = p.exitcode
        if exit_code != 0:
            failed.append(benchname)

    if len(failed) != 0:
        print(f"Failed: {len(failed)} out of {len(benchnames)}")
        for bench in failed:
            print(bench)
