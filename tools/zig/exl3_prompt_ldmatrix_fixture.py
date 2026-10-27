import argparse
import array
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys


SHAPES = ((2560, 2560), (2560, 6144), (2560, 10240), (6144, 2560))
GUARD = 256
CHUNK = 256
HANDLE = ct.c_void_p
PTR = ct.c_uint64
UINT = ct.c_uint
INT = ct.c_int


class Matrix(ct.Structure):
    _fields_ = [(name, UINT) for name in ("k", "n", "bits", "tiles_n", "split")]


def emit(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def mapping_test():
    assert ct.sizeof(Matrix) == 20
    words = 0
    negative_control = False
    for tile_row in range(4):
        for tile_k in range(8):
            supplied = [(16 * tile_row + (lane & 15)) * 136 + 16 * tile_k + (lane >> 4) * 8 for lane in range(32)]
            assert all(address % 8 == 0 for address in supplied)
            for lane in range(32):
                base = (16 * tile_row + lane // 4) * 136 + 16 * tile_k + 2 * (lane % 4)
                scalar = [base, base + 8 * 136, base + 8, base + 8 * 136 + 8]
                matrix = [supplied[8 * quadrant + lane // 4] + 2 * (lane % 4) for quadrant in range(4)]
                assert matrix == scalar
                assert all(0 <= address < 64 * 136 - 1 for address in matrix)
                negative_control |= [matrix[0], matrix[2], matrix[1], matrix[3]] != scalar
                words += 4
    assert negative_control
    assert exact_values()[0] != exact_values()[1]
    exact_fixture_test()
    return words


class Driver:
    def __init__(self, device):
        self.lib = ct.CDLL("libcuda.so.1")
        self.buffers = []
        self.modules = []
        self.events = []
        self.context = HANDLE()
        self.stream = HANDLE()
        signatures = {
            "cuInit": [UINT], "cuDriverGetVersion": [ct.POINTER(INT)],
            "cuDeviceGet": [ct.POINTER(INT), INT], "cuDeviceGetName": [HANDLE, INT, INT],
            "cuDeviceGetAttribute": [ct.POINTER(INT), INT, INT],
            "cuCtxCreate_v2": [ct.POINTER(HANDLE), UINT, INT], "cuCtxDestroy_v2": [HANDLE],
            "cuStreamCreate": [ct.POINTER(HANDLE), UINT], "cuStreamDestroy_v2": [HANDLE],
            "cuStreamSynchronize": [HANDLE], "cuModuleLoad": [ct.POINTER(HANDLE), ct.c_char_p],
            "cuModuleUnload": [HANDLE], "cuModuleGetFunction": [ct.POINTER(HANDLE), HANDLE, ct.c_char_p],
            "cuFuncGetAttribute": [ct.POINTER(INT), INT, HANDLE],
            "cuMemAlloc_v2": [ct.POINTER(PTR), ct.c_size_t], "cuMemFree_v2": [PTR],
            "cuMemcpyHtoD_v2": [PTR, HANDLE, ct.c_size_t], "cuMemcpyDtoH_v2": [HANDLE, PTR, ct.c_size_t],
            "cuMemsetD8Async": [PTR, ct.c_ubyte, ct.c_size_t, HANDLE],
            "cuLaunchKernel": [HANDLE, UINT, UINT, UINT, UINT, UINT, UINT, UINT, HANDLE, ct.POINTER(HANDLE), HANDLE],
            "cuEventCreate": [ct.POINTER(HANDLE), UINT], "cuEventDestroy_v2": [HANDLE],
            "cuEventRecord": [HANDLE, HANDLE], "cuEventSynchronize": [HANDLE],
            "cuEventElapsedTime": [ct.POINTER(ct.c_float), HANDLE, HANDLE],
            "cuStreamBeginCapture_v2": [HANDLE, INT], "cuStreamEndCapture": [HANDLE, ct.POINTER(HANDLE)],
            "cuGraphInstantiateWithFlags": [ct.POINTER(HANDLE), HANDLE, ct.c_uint64],
            "cuGraphLaunch": [HANDLE, HANDLE], "cuGraphExecDestroy": [HANDLE], "cuGraphDestroy": [HANDLE],
            "cuGetErrorName": [INT, ct.POINTER(ct.c_char_p)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.restype = INT
            function.argtypes = arguments
        self.call("cuInit", 0)
        ordinal = INT()
        self.call("cuDeviceGet", ct.byref(ordinal), device)
        self.call("cuCtxCreate_v2", ct.byref(self.context), 0, ordinal)
        self.call("cuStreamCreate", ct.byref(self.stream), 1)
        self.device = ordinal.value

    def call(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            message = ct.c_char_p()
            self.lib.cuGetErrorName(result, ct.byref(message))
            raise RuntimeError(f"{name}: {result} {message.value!r}")

    def synchronize(self):
        self.call("cuStreamSynchronize", self.stream)

    def attribute(self, number):
        value = INT()
        self.call("cuDeviceGetAttribute", ct.byref(value), number, self.device)
        return value.value

    def launch(self, function, grid, threads, arguments):
        pointers = (HANDLE * len(arguments))(*(ct.cast(ct.byref(value), HANDLE) for value in arguments))
        self.call("cuLaunchKernel", function, *grid, threads, 1, 1, 0, self.stream, pointers, None)

    def event(self):
        event = HANDLE()
        self.call("cuEventCreate", ct.byref(event), 0)
        self.events.append(event)
        return event

    def close(self):
        try:
            if self.stream:
                self.synchronize()
        finally:
            for event in reversed(self.events):
                self.lib.cuEventDestroy_v2(event)
            for module in reversed(self.modules):
                self.lib.cuModuleUnload(module)
            for pointer in reversed(self.buffers):
                self.lib.cuMemFree_v2(pointer)
            if self.stream:
                self.lib.cuStreamDestroy_v2(self.stream)
            if self.context:
                self.lib.cuCtxDestroy_v2(self.context)


class Buffer:
    def __init__(self, driver, size):
        self.driver, self.size, self.poison = driver, size, 0xA5
        self.base = PTR()
        driver.call("cuMemAlloc_v2", ct.byref(self.base), size + 2 * GUARD)
        driver.buffers.append(self.base)
        self.ptr = self.base.value + GUARD
        self.reset()

    def reset(self, poison=0xA5):
        self.poison = poison
        self.driver.call("cuMemsetD8Async", self.base, poison, self.size + 2 * GUARD, self.driver.stream)

    def upload(self, data):
        assert len(data) == self.size
        self.driver.synchronize()
        host = ct.create_string_buffer(data)
        self.driver.call("cuMemcpyHtoD_v2", self.ptr, host, self.size)
        self.expected_hash = digest(data)

    def read(self, live=None):
        host = ct.create_string_buffer(self.size + 2 * GUARD)
        self.driver.call("cuMemcpyDtoH_v2", host, self.base, len(host))
        data = host.raw
        live = self.size if live is None else live
        assert data[:GUARD] == bytes([self.poison]) * GUARD, "prefix guard"
        assert data[GUARD + live:] == bytes([self.poison]) * (self.size - live + GUARD), "suffix guard"
        return data[GUARD:GUARD + live]


class Image:
    def __init__(self, driver, path, label):
        self.driver, self.label = driver, label
        module = HANDLE()
        driver.call("cuModuleLoad", ct.byref(module), str(path.resolve()).encode())
        driver.modules.append(module)
        self.functions = {}
        resources = {}
        for name in ("tf_exl3_rotate_h", "tf_exl3_prompt_2"):
            function = HANDLE()
            driver.call("cuModuleGetFunction", ct.byref(function), module, name.encode())
            self.functions[name] = function
            resource = {}
            for key, number in (("max_threads", 0), ("shared_bytes_driver", 1), ("local_bytes", 3), ("registers", 4), ("binary_version", 6)):
                value = INT()
                driver.call("cuFuncGetAttribute", ct.byref(value), number, function)
                resource[key] = value.value
            assert resource["max_threads"] >= (256 if name.endswith("_2") else 32)
            resources[name] = resource
        emit(image=label, sha256=digest(path.read_bytes()), resources=resources)

    def rotate(self, case, count, offset=0):
        self.driver.launch(self.functions["tf_exl3_rotate_h"], (case.k // 128, count, 1), 32,
                           [PTR(case.input.ptr + offset * case.stride * 2), UINT(case.stride), PTR(case.suh.ptr), PTR(case.rotated[self.label].ptr), UINT(case.k)])

    def prompt(self, case, count, f32, bank=0, offset=0):
        self.driver.launch(self.functions["tf_exl3_prompt_2"], (case.n // 128, (count + 63) // 64, 1), 256,
                           [PTR(case.rotated[self.label].ptr), UINT(count), PTR(case.banks[bank].ptr), PTR(case.svh.ptr),
                            Matrix(case.k, case.n, case.bits, case.n // 16, 1), PTR(case.output[self.label].ptr + offset * case.n * (4 if f32 else 2)), UINT(f32)])


def half_bytes(values):
    return b"".join(struct.pack("<e", value) for value in values)


def rounded_bf16(data):
    words = array.array("I", data)
    assert all((word & 0x7F800000) != 0x7F800000 for word in words), "nonfinite random output"
    return array.array("H", (((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF for word in words)).tobytes()


def same(actual, expected, label):
    assert len(actual) == len(expected), f"{label}: length mismatch"
    if actual != expected:
        first = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        raise AssertionError(f"{label}: first differing byte {first}")


def exact_values():
    scale = struct.unpack("<e", bytes.fromhex("ee1e"))[0]
    bias = struct.unpack("<e", bytes.fromhex("31c9"))[0]
    values = []
    for state in (0, 65535):
        product = (state * 0x83DCD12D) & 0xFFFFFFFF
        total = 1024 + sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
        values.append(struct.unpack("<e", struct.pack("<e", total * scale + bias))[0])
    return values


def exact_fixture_test():
    def transform(values):
        values = list(values)
        for step in (1, 2, 4, 8, 16, 32, 64):
            for base in range(0, 128, 2 * step):
                for offset in range(step):
                    lower, upper = base + offset, base + offset + step
                    left, right = values[lower], values[upper]
                    values[lower], values[upper] = left + right, left - right
                    for index in (lower, upper):
                        assert values[index] == struct.unpack("<f", struct.pack("<f", values[index]))[0]
        return values
    weights = exact_values()
    for amplitude in (1, 2, 4):
        result = [value / 8 for value in transform(amplitude * weights[(column // 16) & 1] for column in range(128))]
        expected = [0.0] * 128
        expected[0], expected[16] = 8 * amplitude * (weights[0] + weights[1]), 8 * amplitude * (weights[0] - weights[1])
        assert result == expected
        for frequency in range(128):
            source = [(-1 if (column & frequency).bit_count() & 1 else 1) * amplitude / 8 for column in range(128)]
            rotated = [value / 16 for value in transform(source)]
            assert rotated[frequency] == amplitude and sum(value != 0 for value in rotated) == 1


class Case:
    def __init__(self, driver, images, shape, options, l2):
        self.driver, self.images = driver, images
        self.k, self.n = shape[:2]
        self.bits = shape[2] if len(shape) == 3 else options.bits
        self.stride, self.rows = self.k + 128, max(options.rows)
        self.weight_bytes = self.k * self.n * self.bits // 8
        self.bank_count = 1 if options.profile_only else math.ceil(2 * l2 / self.weight_bytes) + 1
        self.bank_hash = {}
        assert self.bank_count * self.weight_bytes <= options.bank_budget_mib * 1024**2, "weight banks exceed explicit budget"
        self.random = random.Random(options.seed + self.k * 17 + self.n * 31)
        self.banks = []
        for bank in range(self.bank_count):
            buffer = Buffer(driver, self.weight_bytes)
            data = self.random.randbytes(self.weight_bytes)
            buffer.upload(data)
            self.banks.append(buffer)
            if bank == 0:
                self.original_weight = data
        input_words = array.array("H", (0x3F00 | self.random.getrandbits(7) | (self.random.getrandbits(1) << 15) for _ in range(self.rows * self.stride)))
        self.original_input = input_words.tobytes()
        self.original_suh = half_bytes(0.5 + self.random.random() for _ in range(self.k))
        self.original_svh = half_bytes(0.002 + self.random.random() * 0.02 for _ in range(self.n))
        self.input, self.suh, self.svh = Buffer(driver, len(self.original_input)), Buffer(driver, self.k * 2), Buffer(driver, self.n * 2)
        self.rotated = {image.label: Buffer(driver, CHUNK * self.k * 2) for image in images}
        self.output = {image.label: Buffer(driver, self.rows * self.n * 4) for image in images}
        self.restore()
        emit(shape=[self.k, self.n], bits=self.bits, matrix_bytes=self.weight_bytes, banks=self.bank_count,
             rotating_trellis_bytes=self.bank_count * self.weight_bytes, l2_bytes=l2, bank_budget_mib=options.bank_budget_mib)

    def restore(self):
        self.banks[0].upload(self.original_weight)
        self.input.upload(self.original_input)
        self.suh.upload(self.original_suh)
        self.svh.upload(self.original_svh)

    def project(self, image, count, f32, bank, poison):
        self.output[image.label].reset(poison)
        self.rotated[image.label].reset(poison)
        for offset in range(0, count, CHUNK):
            piece = min(CHUNK, count - offset)
            image.rotate(self, piece, offset)
            image.prompt(self, piece, f32, bank, offset)
        self.driver.synchronize()
        self.rotated[image.label].read()
        return self.output[image.label].read(count * self.n * (4 if f32 else 2))

    def compare(self, count, f32, bank, repetition):
        order = self.images if repetition % 2 == 0 else self.images[::-1]
        outputs = {image.label: self.project(image, count, f32, bank, 0xA5 ^ repetition) for image in order}
        same(outputs["candidate"], outputs["baseline"], f"shape {self.k},{self.n} rows {count} bank {bank} f32 {f32}")
        if count == CHUNK:
            self.bank_hash[bank, f32] = digest(outputs["baseline"])
        return outputs["baseline"]

    def checks(self, options):
        rounding = {}
        for f32 in (True, False):
            prefix = None
            for count in options.rows:
                first_hash = None
                for repetition in range(options.check_reps):
                    data = self.compare(count, f32, 0, repetition)
                    current_hash = digest(data)
                    assert first_hash in (None, current_hash), "repeat bytes changed"
                    first_hash = current_hash
                    if repetition == 0:
                        if f32:
                            rounding[count] = digest(rounded_bf16(data))
                        else:
                            assert current_hash == rounding[count], "bf16 is not fp32 rounded"
                        if count == CHUNK:
                            prefix = data
                        if count > CHUNK and prefix is not None:
                            same(data[:len(prefix)], prefix, "256-row seam")
        for bank in range(1, self.bank_count):
            for f32 in (True, False):
                self.compare(CHUNK, f32, bank, bank % 2)
        random_hashes = dict(self.bank_hash)
        self.exact_check()
        self.nonfinite_check()
        self.bank_hash = random_hashes
        self.restore()
        emit(shape=[self.k, self.n], byte_guard_repeat_checks="PASS", banks_checked=self.bank_count, fp64_exact_fixture="PASS", nonfinite_classes="PASS")

    def nonfinite_check(self):
        def transform(values):
            values = list(values)
            for step in (1, 2, 4, 8, 16, 32, 64):
                for base in range(0, 128, 2 * step):
                    for offset in range(step):
                        lower, upper = base + offset, base + offset + step
                        left, right = values[lower], values[upper]
                        values[lower], values[upper] = left + right, left - right
            return values
        def category(value):
            return "nan" if math.isnan(value) else "positive_inf" if value == math.inf else "negative_inf" if value == -math.inf else "finite"
        tile_bytes = 32 * self.bits
        self.banks[0].upload((bytes(tile_bytes) + bytes([255]) * tile_bytes) * (self.n // 32) * (self.k // 16))
        self.suh.upload(half_bytes([1.0] * self.k))
        self.svh.upload(half_bytes([1.0] * self.n))
        source = array.array("H", [0]) * (self.rows * self.stride)
        families = ((0x7FC0,), (0x7F80,), (0xFF80,), (0x7F80, 0xFF80))
        expected = []
        weights = exact_values()
        for row, words in enumerate(families):
            for column, word in enumerate(words):
                source[row * self.stride + column] = word
            values = [struct.unpack("<f", struct.pack("<I", source[row * self.stride + column] << 16))[0] for column in range(128)]
            rotated = [value / 16 for value in transform(values)]
            dots = [sum(value * weights[(column // 16) & 1] for value in rotated) for column in range(128)]
            classes = [category(value / 8) for value in transform(dots)]
            expected.extend(classes * (self.n // 128))
        self.input.upload(source.tobytes())
        for f32 in (True, False):
            data = self.compare(len(families), f32, 0, 0)
            actual = array.array("f", data) if f32 else [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in array.array("H", data)]
            assert [category(value) for value in actual] == expected, "nonfinite FP64 classes differ"

    def exact_check(self):
        tile_bytes = 32 * self.bits
        tile_row = (bytes(tile_bytes) + bytes([255]) * tile_bytes) * (self.n // 32)
        self.banks[0].upload(tile_row * (self.k // 16))
        self.suh.upload(half_bytes([1.0] * self.k))
        self.svh.upload(half_bytes([1.0] * self.n))
        source = array.array("H", [0]) * (self.rows * self.stride)
        weights = exact_values()
        expected = array.array("f", [0]) * (CHUNK * self.n)
        for row in range(CHUNK):
            amplitude = 1 << (row % 3)
            block = (row * 7) % (self.k // 128)
            frequency = row % 128
            magnitude = (0x3E00, 0x3E80, 0x3F00)[row % 3]
            for column in range(128):
                source[row * self.stride + block * 128 + column] = magnitude | ((((column & frequency).bit_count() & 1)) << 15)
            for column in range(0, self.n, 128):
                expected[row * self.n + column] = 8 * amplitude * (weights[0] + weights[1])
                expected[row * self.n + column + 16] = 8 * amplitude * (weights[0] - weights[1])
        self.input.upload(source.tobytes())
        wanted = expected.tobytes()
        for f32 in (True, False):
            data = self.compare(CHUNK, f32, 0, 0)
            same(data, wanted if f32 else rounded_bf16(wanted), "independent dyadic FP64 fixture")

    def prepare(self):
        for image in self.images:
            self.rotated[image.label].reset()
            self.output[image.label].reset()
            image.rotate(self, CHUNK)
        self.driver.synchronize()
        same(self.rotated["candidate"].read(), self.rotated["baseline"].read(), "input rotation")

    def timing(self, options):
        self.prepare()
        start, end = self.driver.event(), self.driver.event()
        submissions = ("eager", "graph") if options.submission == "both" else (options.submission,)
        for submission in submissions:
            for cache in ("warm_weight", "rotating_banks"):
                banks = 1 if cache == "warm_weight" else self.bank_count
                for f32 in (False, True):
                    measurements = {image.label: [] for image in self.images}
                    for image in self.images:
                        self.output[image.label].reset()
                        for iteration in range(options.warmup):
                            image.prompt(self, CHUNK, f32, iteration % banks)
                    self.driver.synchronize()
                    for sample in range(options.samples):
                        sequence = []
                        first = sample * options.iterations % banks
                        for image in (self.images[0], self.images[1], self.images[1], self.images[0]):
                            per_launch = self.window(image, f32, banks, first, options.iterations, submission, start, end)
                            measurements[image.label].append(per_launch)
                            sequence.append({"image": image.label, "ms_per_launch": per_launch})
                        emit(shape=[self.k, self.n], cache=cache, submission=submission, f32=f32, sample=sample, abba=sequence)
                    emit(shape=[self.k, self.n], cache=cache, submission=submission, f32=f32, rows=CHUNK,
                         grid=[self.n // 128, 4, 1], block=[256, 1, 1], timing="CUDA_event_span",
                         eager_submission_gaps_included=submission == "eager",
                         baseline_median_ms=statistics.median(measurements["baseline"]),
                         candidate_median_ms=statistics.median(measurements["candidate"]), iterations=options.iterations,
                         samples=options.samples, warmup=options.warmup, conditioning_launches_per_window=banks,
                         working_set_trellis_bytes=banks * self.weight_bytes)
        self.driver.synchronize()
        for image in self.images:
            self.output[image.label].read(CHUNK * self.n * 4)
            self.rotated[image.label].read()

    def window(self, image, f32, banks, first, iterations, submission, start, end):
        graph, executable = HANDLE(), HANDLE()
        def launches():
            for iteration in range(iterations):
                image.prompt(self, CHUNK, f32, (first + iteration) % banks)
        try:
            if submission == "graph":
                self.driver.call("cuStreamBeginCapture_v2", self.driver.stream, 1)
                launches()
                self.driver.call("cuStreamEndCapture", self.driver.stream, ct.byref(graph))
                self.driver.call("cuGraphInstantiateWithFlags", ct.byref(executable), graph, 0)
            def submit():
                if executable:
                    self.driver.call("cuGraphLaunch", executable, self.driver.stream)
                else:
                    launches()
            submit()
            self.driver.synchronize()
            payload = self.output[image.label].read(CHUNK * self.n * (4 if f32 else 2))
            assert digest(payload) == self.bank_hash[(first + iterations - 1) % banks, f32], "timing preflight bytes"
            for bank in range(banks):
                image.prompt(self, CHUNK, f32, (first + bank) % banks)
            self.driver.call("cuEventRecord", start, self.driver.stream)
            submit()
            self.driver.call("cuEventRecord", end, self.driver.stream)
            self.driver.call("cuEventSynchronize", end)
            elapsed = ct.c_float()
            self.driver.call("cuEventElapsedTime", ct.byref(elapsed), start, end)
            return elapsed.value / iterations
        finally:
            if executable:
                self.driver.lib.cuGraphExecDestroy(executable)
            if graph:
                self.driver.lib.cuGraphDestroy(graph)

    def immutable(self):
        for buffer in [self.input, self.suh, self.svh, *self.banks]:
            assert digest(buffer.read()) == buffer.expected_hash, "read-only input changed"


def main():
    parser = argparse.ArgumentParser(description="Separate-fatbin EXL3 prompt driver fixture; emits JSON lines.")
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bits", type=int, choices=range(1, 9), default=5)
    parser.add_argument("--shape", action="append", help="K,N or K,N,BITS; default is all four traced classes")
    parser.add_argument("--rows", default="1,63,64,65,127,128,129,255,256,257")
    parser.add_argument("--check-reps", type=int, default=3)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--submission", choices=("both", "eager", "graph"), default="both")
    parser.add_argument("--profile-only", choices=("baseline", "candidate"),
                        help="Requires one --shape K,N[,BITS]; launches exactly one 256-row prompt, no timing or byte qualification.")
    parser.add_argument("--profile-f32", action="store_true")
    parser.add_argument("--bank-budget-mib", type=int, default=128)
    options = parser.parse_args()
    assert sys.byteorder == "little"
    emit(host_mapping_words=mapping_test(), mapping_negative_control="PASS", fp64_dyadic_fixture_host_proof="PASS")
    if options.self_test:
        return
    assert options.baseline and options.candidate, "two fatbin paths required"
    assert digest(options.baseline.read_bytes()) != digest(options.candidate.read_bytes()), "identical fatbin images"
    options.rows = sorted(set(map(int, options.rows.split(","))) | {CHUNK, CHUNK + 1})
    assert 1 <= min(options.rows) and max(options.rows) <= 513
    assert options.check_reps >= 2 and options.samples >= 1 and options.iterations >= 1 and options.warmup >= 0
    shapes = [tuple(map(int, value.split(","))) for value in options.shape] if options.shape else SHAPES
    assert not options.profile_only or options.shape and len(shapes) == 1, "profile-only requires exactly one --shape"
    assert all(len(shape) in (2, 3) and shape[:2] in SHAPES and (len(shape) == 2 or 1 <= shape[2] <= 8) for shape in shapes)
    driver = Driver(options.device)
    try:
        name = ct.create_string_buffer(256)
        version = INT()
        driver.call("cuDeviceGetName", name, len(name), driver.device)
        driver.call("cuDriverGetVersion", ct.byref(version))
        l2 = driver.attribute(38)
        assert l2 > 0
        emit(device=options.device, gpu=name.value.decode(), cuda_driver=version.value, l2_bytes=l2,
             python=sys.version, seed=20261022, submission=options.submission, profile_only=options.profile_only,
             runtime_flag_match="must be established by external build receipts")
        options.seed = 20261022
        images = [Image(driver, options.baseline, "baseline"), Image(driver, options.candidate, "candidate")]
        for shape in shapes:
            first_buffer = len(driver.buffers)
            case = Case(driver, images, shape, options, l2)
            if options.profile_only:
                case.prepare()
                image = images[0 if options.profile_only == "baseline" else 1]
                image.prompt(case, CHUNK, options.profile_f32)
                driver.synchronize()
                case.output[image.label].read(CHUNK * case.n * (4 if options.profile_f32 else 2))
                emit(profile_only=image.label, rows=CHUNK, shape=[case.k, case.n], bits=case.bits,
                     cache="uncontrolled_after_upload", byte_comparison_run=False, prompt_launches=1)
            else:
                case.checks(options)
                if not options.checks_only:
                    case.timing(options)
            case.immutable()
            driver.synchronize()
            for pointer in reversed(driver.buffers[first_buffer:]):
                driver.call("cuMemFree_v2", pointer)
            del driver.buffers[first_buffer:]
        emit(result="PROFILE_ONLY" if options.profile_only else "PASS", shapes=len(shapes),
             qualification="microfixture only; serving performance unqualified")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
