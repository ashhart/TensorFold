//! The native CUDA host's close and trainer-destruction tests for each learning family, on its actual Host.
const Host = @import("cuda.zig").Host;

test "nemotron learning safety: actual native close drains Nemotron before any source, trainer or parent free" {
    try @import("nemotron_trainer_test.zig").run(Host, Host.serve, Host.close, false);
}
test "nemotron learning safety: direct Nemotron trainer destruction drains before its first owner free" {
    try @import("nemotron_trainer_test.zig").run(Host, Host.serve, Host.close, true);
}
test "trainer close: real native close drains the accepted trainer before any child or parent is released" {
    try @import("trainer_close_test.zig").run(Host, Host.serve, Host.close);
}
test "trainer close: real native close drains the dense trainer before any child or parent is released" {
    try @import("dense_trainer_close_test.zig").run(Host, Host.serve, Host.close);
}
