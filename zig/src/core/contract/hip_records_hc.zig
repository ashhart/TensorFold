//! Gate 5's records of Flash Next's hyper-connection launches: the dense mix's gate table, the mixes and the norms.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The chips with bf16 activations (all but RDNA2), the only ones that launch the 16-byte bf16 kernels.
const bf16_chips = [limits.rdna.len]bool{ false, true, true, true };

pub const records = [_]Record{
    .{
        .launch = .{ .name = "Flash Next HC gate table tf_hc_gate_fill", .symbol = "tf_hc_gate_fill", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 4, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next HC mix tf_fn_hc_mix8", .symbol = "tf_fn_hc_mix8", .image = "flashnext", .threads = 256, .served = bf16_chips },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 26, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 26, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next HC norm tf_fn_hc_norm_v", .symbol = "tf_fn_hc_norm_v", .image = "flashnext", .threads = 128, .served = bf16_chips },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 22, .lds = 128, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 22, .lds = 128, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 23, .lds = 128, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next HC norm with the MoE combine tf_fn_hc_norm_moe", .symbol = "tf_fn_hc_norm_moe", .image = "flashnext", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 19, .sgprs = 40, .lds = 128, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 40, .lds = 128, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 40, .lds = 128, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 42, .lds = 128, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next HC norm with the MoE combine, a block a row tf_fn_hc_norm_moe_v", .symbol = "tf_fn_hc_norm_moe_v", .image = "flashnext", .threads = 512, .served = bf16_chips },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 51, .sgprs = 30, .lds = 10752, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 59, .sgprs = 27, .lds = 10752, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 59, .sgprs = 30, .lds = 10752, .private = 0, .bound = 512, .wave = 32 },
        },
    },
};
