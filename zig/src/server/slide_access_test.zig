//! Automatic learning takes a chat turn only from a key allowed to change shared memory, or by the server's switch.
const std = @import("std");
const scope = @import("auth_scope.zig");
const t = std.testing;

test "slide access: inference-only and read keys never feed automatic learning; admin keys and keyless servers do" {
    try t.expect(!scope.learnsFrom(.{ .inference = true }));
    try t.expect(!scope.learnsFrom(.{ .read = true }));
    try t.expect(!scope.learnsFrom(.{ .inference = true, .read = true }));
    try t.expect(scope.learnsFrom(.{ .admin = true }));
    try t.expect(scope.learnsFrom(scope.full)); // a server without keys, or a key without a scope list
}

test "slide access: TF_SLIDE_AUTO_ALL=1 is the explicit switch that lets every key's turns be learned" {
    const env = @import("sys").env;
    try t.expectEqual(@as(c_int, 0), env.setenv("TF_SLIDE_AUTO_ALL", "1", 1));
    defer _ = env.unsetenv("TF_SLIDE_AUTO_ALL");
    try t.expect(scope.learnsFrom(.{ .inference = true }));
}

test "slide access: a chat turn is offered to learning only when its route says the key may teach" {
    const Input = @import("chat.zig").Input;
    const input: Input = .{ .fields = .null };
    try t.expect(!input.learns); // routes that do not ask feed nothing
}
