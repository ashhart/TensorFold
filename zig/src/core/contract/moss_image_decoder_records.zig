//! MOSS-Image's image decoder launches (moss_image_decoder.cu): geometry recorded, five-SM usage receipts owed.
const CudaRecord = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none;
pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "moss image decoder tf_md_unpack", .symbol = "tf_md_unpack", .image = "moss_image_decoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_gemm", .symbol = "tf_md_gemm", .image = "moss_image_decoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_conv3x3", .symbol = "tf_md_conv3x3", .image = "moss_image_decoder", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_group_part 32 channels", .symbol = "tf_md_group_part", .image = "moss_image_decoder", .threads = 32 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_group_part 96 channels", .symbol = "tf_md_group_part", .image = "moss_image_decoder", .threads = 96 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_group_part 192 channels", .symbol = "tf_md_group_part", .image = "moss_image_decoder", .threads = 192 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_group_part 384 channels", .symbol = "tf_md_group_part", .image = "moss_image_decoder", .threads = 384 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_group_finish", .symbol = "tf_md_group_finish", .image = "moss_image_decoder", .threads = 1024 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_norm 32 channels", .symbol = "tf_md_norm", .image = "moss_image_decoder", .threads = 32 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_norm 96 channels", .symbol = "tf_md_norm", .image = "moss_image_decoder", .threads = 96 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_norm 192 channels", .symbol = "tf_md_norm", .image = "moss_image_decoder", .threads = 192 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_norm 384 channels", .symbol = "tf_md_norm", .image = "moss_image_decoder", .threads = 384 }, .usage = none },
    .{ .launch = .{ .name = "moss image decoder tf_md_softmax", .symbol = "tf_md_softmax", .image = "moss_image_decoder", .threads = 256 }, .usage = none },
};
