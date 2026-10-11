"""GPU-free dry run of the real oracle's ctypes context lifecycle."""
import ctypes as C
import unittest
from unittest.mock import patch

from kv_glue_gpu import Driver


class FakeFunction:
    def __init__(self, library, name):
        self.library, self.name = library, name

    def __call__(self, *args):
        lib, name = self.library, self.name
        lib.calls.append(name)
        if name == "cuInit":
            return lib.init_status
        if name == "cuDeviceGetCount":
            C.cast(args[0], C.POINTER(C.c_int))[0] = lib.devices
        elif name == "cuDeviceGet":
            C.cast(args[0], C.POINTER(C.c_int))[0] = args[1]
            lib.ordinal = args[1]
        elif name == "cuDevicePrimaryCtxRetain":
            lib.retained = 0xcafe + args[1].value
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = lib.retained
        elif name == "cuCtxSetCurrent":
            lib.current = args[0].value
        elif name == "cuCtxGetCurrent":
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = lib.current
        elif name == "cuModuleLoad":
            assert lib.current == lib.retained, "module load has no primary context"
            if lib.load_status:
                return lib.load_status
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = 0x1234
        elif name == "cuModuleGetFunction":
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = 0x5678
        elif name == "cuDevicePrimaryCtxRelease_v2":
            assert args[0].value == lib.ordinal
            lib.released += 1
        return 0


class FakeCuda:
    def __init__(self, devices=2, init_status=0, load_status=0):
        self.devices, self.init_status, self.load_status = devices, init_status, load_status
        self.current, self.retained, self.released = 0xaaaa, None, 0
        self.calls = []

    def __getattr__(self, name):
        return FakeFunction(self, name)


class ContextDryRun(unittest.TestCase):
    def test_primary_context_before_load_and_same_torch_context(self):
        lib = FakeCuda()
        with patch.object(C, "CDLL", return_value=lib):
            driver = Driver("unused.fatbin", device=1)
        self.assertEqual(lib.calls, ["cuInit", "cuDeviceGetCount", "cuDeviceGet",
                                    "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent",
                                    "cuModuleLoad", "cuModuleGetFunction"])
        driver.assert_current()  # Torch using this same primary context leaves comparison intact.
        lib.current = 0xbbbb  # A reference-side switch must be detected before any fused launch.
        with self.assertRaisesRegex(RuntimeError, "Torch reference and fused kernel"):
            driver.launch(None, 1, None, 128)
        self.assertNotIn("cuLaunchKernel", lib.calls)
        driver.close()
        self.assertEqual(lib.calls[-3:], ["cuCtxSetCurrent", "cuModuleUnload", "cuDevicePrimaryCtxRelease_v2"])
        self.assertEqual(lib.current, lib.retained)
        driver.close()
        self.assertEqual(lib.released, 1)

    def test_no_device_is_clear_and_never_loads_module(self):
        for lib in (FakeCuda(devices=0), FakeCuda(init_status=100)):
            with self.subTest(status=lib.init_status), patch.object(C, "CDLL", return_value=lib):
                with self.assertRaisesRegex(RuntimeError, "[Nn]o CUDA device present.*requires a CUDA GPU"):
                    Driver("unused.fatbin")
            self.assertNotIn("cuDevicePrimaryCtxRetain", lib.calls)
            self.assertNotIn("cuModuleLoad", lib.calls)

    def test_failed_module_load_releases_retained_context(self):
        lib = FakeCuda(load_status=201)
        with patch.object(C, "CDLL", return_value=lib):
            with self.assertRaisesRegex(RuntimeError, "cuModuleLoad: CUDA driver error 201"):
                Driver("unused.fatbin")
        self.assertEqual(lib.calls[-1], "cuDevicePrimaryCtxRelease_v2")
        self.assertEqual(lib.released, 1)


if __name__ == "__main__":
    unittest.main()
