# Native weight-read checks

The probe calls `Checkpoint.addFile`, `Run.load`, and `Run.group` on Apple Silicon with Metal.
It uses synthetic 2 MiB safetensors files and requires no model weights.
Python 3.9 or later supplies the test driver.

1. Build the probe with the Zig version in `.zig-version`.

   ```sh
   zig build tf-weight-read-check -Dcpu=apple_m1
   ```

2. Execute the test driver with a new output directory.

   ```sh
   python3 tools/zig/check_weight_reads.py --output /tmp/tensorfold-weight-results
   ```

Use `--storage-directory /path/to/storage` to select a filesystem for the temporary fixtures.
The driver removes its fixtures after the tests.
It retains the logs and `result.json` in the output directory.
The probe overwrites only test files with the filename prefix `tensorfold-loader-probe-`.

The default run has 54 cases across three read paths and tensor offsets 0 and 64.
Normal reads and delayed partial reads each repeat three times.
The partial-read mode limits each `pread` call to 32 KiB and adds a 5 ms delay.
After each successful load, the probe overwrites the source file and verifies all 524,288 values through a Metal copy.
The probe also checks for subsequent `pread` calls.

Each read path and offset has one premature EOF, one `EIO`, and one `EINTR` case.
Each case requires `ShortRead` and unchanged Metal `currentAllocatedSize` after the error.
These cases do not require a new error policy or an `EINTR` retry.
Use `--faults-only` for these 18 cases.

The page-cache state is uncontrolled.
These small tests do not establish cold NAS reliability, model exactness, or decode speed.
The regular `zig build test` target does not execute this GPU probe.
