//! Gate 5's records of the gfx11 matrix-core prompt tiles and Flash Next's attention launches.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "prompt expert gate+up tile 4-bit", .symbol = "_ZN2tf4rocm18affine_wmma_moe_guILi4EEEvNS0_6AffineEf", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 226, .sgprs = 34, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 36, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert down tile 4-bit", .symbol = "_ZN2tf4rocm20affine_wmma_moe_downILi4EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 33, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 34, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi320EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320 with the HC mix", .symbol = "tf_affine_dense_hcmix", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 2560", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi2560EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 6144", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi6144EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 10240", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi10240EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next QSA decode prep", .symbol = "tf_fn_qsa_prep", .image = "flashnext", .threads = 256 },
        .usage = .{
            .{ .vgprs = 36, .sgprs = 50, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 50, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 34, .sgprs = 52, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 34, .sgprs = 51, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense inexact tile 4-bit K 2560", .symbol = "_ZN2tf4rocm25affine_wmma_dense_inexactILi2560EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense inexact tile 4-bit K 6144", .symbol = "_ZN2tf4rocm25affine_wmma_dense_inexactILi6144EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert gate+up inexact tile 4-bit", .symbol = "_ZN2tf4rocm26affine_wmma_moe_gu_inexactILi4EEEvNS0_6AffineEf", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 219, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 218, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert down inexact tile 4-bit", .symbol = "_ZN2tf4rocm28affine_wmma_moe_down_inexactILi4EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile", .symbol = "tf_fn_sparse_prompt", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 31, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile, p in bf16 once", .symbol = "tf_fn_sparse_prompt_hi", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 31, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile, split", .symbol = "tf_fn_sparse_prompt_split", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 33, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next decode sparse attention tile parts", .symbol = "tf_fn_sparse_tile_parts", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 247, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 247, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next decode sparse attention tile parts, split", .symbol = "tf_fn_sparse_tile_parts_split", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
