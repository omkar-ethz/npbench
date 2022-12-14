import argparse
import os

from multiprocessing import Process
from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

def run_benchmark(benchname, fname, preset, validate, repeat, timeout, num_threads):
    frmwrk = generate_framework(fname)
    numpy = generate_framework("numpy")
    bench = Benchmark(benchname)
    test = Test(bench, frmwrk, numpy)
    test.run(preset, validate, repeat, timeout, num_threads=num_threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-t", "--timeout", type=float, nargs="?", default=200.0)
    parser.add_argument("-d", "--dace", type=util.str2bool, nargs="?", default=True)
    parser.add_argument("-n", "--num-threads", type=int, nargs="?", default=1)
    args = vars(parser.parse_args())

    print("num_threads=", args["num_threads"])
    os.environ['OMP_NUM_THREADS'] = str(args["num_threads"])
    os.environ['MKL_NUM_THREADS'] = str(args["num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(args["num_threads"])
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args["num_threads"])
    os.environ["NUMEXPR_NUM_THREADS"] = str(args["num_threads"])

    benchmarks = [
        'adi', 'arc_distance', 'atax', 'bicg', 'cavity_flow',
        'cholesky2', 'compute', 'doitgen', 'floyd_warshall', 'gemm', 'gemver',
        'gesummv', 'go_fast', 'hdiff', 'jacobi_2d', 'lenet', 'syr2k', 'trmm',
        'vadv'
    ]


    frameworks = ["numpy", "dace_cpu", "dace_cpu_tasking", "dace_cpu_dynamic_schedule"]

    for benchname in benchmarks:
        for fname in frameworks:
            # set environment variable openmp_tasking
            if fname == "dace_cpu_tasking":
                os.environ['DACE_compiler_cpu_openmp_tasking'] = "true"
            elif fname == "dace_cpu":
                os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
            elif fname == "dace_cpu_dynamic_schedule":
                os.environ['DACE_compiler_cpu_openmp_tasking'] = "false"
                os.environ['DACE_compiler_cpu_openmp_dynamic_schedule'] = "true"
            p = Process(
                target=run_benchmark,
                args=(benchname, fname, args["preset"],
                    args["validate"], args["repeat"], args["timeout"], args["num_threads"])
            )
            p.start()
            p.join()

    # numpy = generate_framework("numpy")
    # numba = generate_framework("numba")

    # for benchname in benchmarks:
    #     bench = Benchmark(benchname)
    #     for frmwrk in [numpy, numba]:
    #         lcount = LineCount(bench, frmwrk, numpy)
    #         lcount.count()
    #         test = Test(bench, frmwrk, numpy)
    #         try:
    #             test.run(args["preset"], args["validate"], args["repeat"],
    #                      args["timeout"])
    #         except:
    #             continue
