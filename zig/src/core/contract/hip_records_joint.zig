//! Gate 5's records of the joint schema head's kernels (joint_schema/joint_schema.hip): 256 threads a launch.
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "joint schema layer norm tf_js_layer_norm", .symbol = "tf_js_layer_norm", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 20, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema GELU tf_js_gelu", .symbol = "tf_js_gelu", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 7, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema row softmax tf_js_softmax_rows", .symbol = "tf_js_softmax_rows", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 19, .sgprs = 15, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 18, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 15, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 14, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema widening tf_js_widen", .symbol = "tf_js_widen", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 8, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema rounding tf_js_round", .symbol = "tf_js_round", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 5, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 8, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema span sums tf_js_span_sums", .symbol = "tf_js_span_sums", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 7, .sgprs = 22, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 22, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 22, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 22, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema span means tf_js_span_means", .symbol = "tf_js_span_means", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema option summary tf_js_option_summary", .symbol = "tf_js_option_summary", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 34, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 34, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 34, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 34, .lds = 72, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema logits tf_js_logits", .symbol = "tf_js_logits", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 39, .sgprs = 33, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 33, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 34, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 41, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "joint schema gated logits tf_js_combine", .symbol = "tf_js_combine", .image = "joint_schema", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};
