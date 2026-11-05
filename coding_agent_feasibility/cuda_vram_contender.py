#!/usr/bin/env python3
"""Optional GPU memory contender using NVIDIA CUDA Driver API (no pip packages).

Allocates memory in a separate process, holds, frees, then exits. Never tries
an OOM intentionally. Emits JSON status lines for the experiment controller.
"""
import argparse
import ctypes
import json
import sys
import time


def say(**data):
    print(json.dumps(data), flush=True)


def run(mib: int, hold: float):
    if mib <= 0 or mib > 2048:
        raise ValueError('mib must be in 1..2048; start with 256 or 512')
    cuda = ctypes.CDLL('libcuda.so.1')
    def sym(name, restype=ctypes.c_int, argtypes=None):
        f = getattr(cuda, name)
        f.restype = restype
        if argtypes:
            f.argtypes = argtypes
        return f
    init = sym('cuInit', argtypes=[ctypes.c_uint])
    dev_get = sym('cuDeviceGet', argtypes=[ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    ctx_create = sym('cuCtxCreate_v2', argtypes=[ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int])
    ctx_destroy = sym('cuCtxDestroy_v2', argtypes=[ctypes.c_void_p])
    alloc = sym('cuMemAlloc_v2', argtypes=[ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    mem_set = sym('cuMemsetD8_v2', argtypes=[ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t])
    free = sym('cuMemFree_v2', argtypes=[ctypes.c_uint64])
    def check(code, step):
        if code:
            raise RuntimeError(f'{step} failed with CUDA Driver error code {code}')
    check(init(0), 'cuInit')
    dev = ctypes.c_int()
    check(dev_get(ctypes.byref(dev), 0), 'cuDeviceGet')
    ctx = ctypes.c_void_p()
    check(ctx_create(ctypes.byref(ctx), 0, dev.value), 'cuCtxCreate')
    ptr = ctypes.c_uint64()
    allocated = False
    try:
        n_bytes = mib * 1024 * 1024
        check(alloc(ctypes.byref(ptr), n_bytes), 'cuMemAlloc')
        allocated = True
        check(mem_set(ptr.value, 0, n_bytes), 'cuMemsetD8')
        say(event='allocated', allocated_mib=mib)
        time.sleep(hold)
        check(free(ptr.value), 'cuMemFree')
        allocated = False
        say(event='freed', allocated_mib=mib)
    finally:
        if allocated:
            free(ptr.value)
        ctx_destroy(ctx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mib', type=int, default=512)
    p.add_argument('--hold', type=float, default=5)
    a = p.parse_args()
    try:
        run(a.mib, a.hold)
    except Exception as exc:
        say(event='error', message=str(exc))
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
