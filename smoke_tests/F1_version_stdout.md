# F1: `tensorfold --version` prints to stderr
cuda_main.zig uses std.debug.print (stderr). Breaks `$(tensorfold --version)`.
Run: BIN=../zig-out/bin/tensorfold ./version_stdout.sh  (FAILs today)
Fix: write to init.io/stdout (like checkpoint_cli.main does) instead of std.debug.print.
