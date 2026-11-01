//! Idol's kernel-contract references on the host (every CUDA component): ranges hold float64, no GPU work.
test {
    _ = @import("contract_idol_ref.zig");
    _ = @import("contract_idol_norms.zig");
    _ = @import("contract_idol_steps.zig");
    _ = @import("contract_idol_rotary.zig");
    _ = @import("contract_idol_case.zig");
    _ = @import("contract_idol_dino_ops.zig");
    _ = @import("contract_idol_dino_norms.zig");
    _ = @import("contract_idol_dino_emulate.zig");
    _ = @import("contract_idol_vae_convs.zig");
    _ = @import("contract_idol_vae_rows.zig");
    _ = @import("contract_idol_dec_steps.zig");
    _ = @import("contract_idol_dec_attn.zig");
}
