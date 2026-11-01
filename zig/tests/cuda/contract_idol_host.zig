//! Idol's kernel-contract references on the host (denoiser and image encoder): ranges hold float64, no GPU work.
test {
    _ = @import("contract_idol_ref.zig");
    _ = @import("contract_idol_norms.zig");
    _ = @import("contract_idol_steps.zig");
    _ = @import("contract_idol_rotary.zig");
    _ = @import("contract_idol_case.zig");
    _ = @import("contract_idol_vae_convs.zig");
    _ = @import("contract_idol_vae_rows.zig");
}
