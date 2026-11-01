//! Compile the complete CUDA component with inert images; this object is never executed.
const suite = @import("tests/cuda/qwen_moe_back_tests.zig");
export fn qwen_moe_contract_compile(gpu: *const @import("tests/cuda/check.zig").Gpu) void {
    suite.run(gpu.*) catch {};
    suite.control(gpu.*) catch {};
}
