//! MOSS-Image's prompt encoder launches (moss_image_encoder.cu): geometry recorded, five-SM usage receipts owed.
const CudaRecord = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none;
pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "moss image encoder tf_me_norm", .symbol = "tf_me_norm", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_head", .symbol = "tf_me_head", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_attention", .symbol = "tf_me_attention", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_gate", .symbol = "tf_me_gate", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_silu_mul", .symbol = "tf_me_silu_mul", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_conv", .symbol = "tf_me_conv", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_gdn_prep", .symbol = "tf_me_gdn_prep", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_gdn", .symbol = "tf_me_gdn", .image = "moss_image_encoder", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "moss image encoder tf_me_gated_norm", .symbol = "tf_me_gated_norm", .image = "moss_image_encoder", .threads = 256 }, .usage = none },
};
