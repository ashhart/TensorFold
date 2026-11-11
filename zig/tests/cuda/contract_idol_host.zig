//! Idol's kernel-contract references on the host: their ranges hold the float64 values, no GPU work.
test {
    _ = @import("contract_idol_ref.zig");
    _ = @import("contract_idol_norms.zig");
    _ = @import("contract_idol_steps.zig");
    _ = @import("contract_idol_rotary.zig");
    _ = @import("contract_idol_case.zig");
}
