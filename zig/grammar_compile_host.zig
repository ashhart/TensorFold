//! The grammar compile runner's deadlines, slots, shared jobs and ownership (tf_compile_test.cc), without xgrammar.
const std = @import("std");
extern "c" fn tf_compile_host_test(c_int) c_int;

fn run(which: c_int) !void {
    try std.testing.expectEqual(@as(c_int, 0), tf_compile_host_test(which));
}

test "grammar compile: a cached grammar keeps its bytes, keyed by format and text" {
    try run(0);
}
test "grammar compile: a caller stops waiting at its deadline and a late result is not cached" {
    try run(1);
}
test "grammar compile: a small compile passes a large one still running" {
    try run(2);
}
test "grammar compile: two callers of one input share one compile" {
    try run(3);
}
test "grammar compile: a full runner waits for a slot until the deadline, cached grammars still served" {
    try run(4);
}
test "grammar compile: an abandoned compile keeps the state it owns past close" {
    try run(5);
}
test "grammar compile: a failure is not cached and is reported without waiting for cleanup" {
    try run(6);
}
test "grammar compile: compile threads have the request threads' 8 MiB stack" {
    try run(7);
}
