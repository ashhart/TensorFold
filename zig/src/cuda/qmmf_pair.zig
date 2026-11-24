//! One launch for the two scalar FP8 Qwen Gated DeltaNet projections.
const qmmf = @import("qmmf.zig");
const launch = @import("launch.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const Pair = struct {
    function: Function,

    pub const symbol = "tf_qmmf_fp8_qkv_z";
    const width = 5120;
    const qkv_columns = 10240;
    const z_columns = 6144;
    const tile_rows = 16;
    const split = 2;
    const shared = qmmf.sharedBytes(.fp8, tile_rows);

    pub fn load(lane: *const qmmf.Lane) !Pair {
        const function = try lane.mod.function(symbol);
        try function.allowDynamicShared(shared);
        return .{ .function = function };
    }

    /// Both plans must match the original clustered BF16-output lane launches.
    fn compatible(lane: *const qmmf.Lane, w: qmmf.Weight, columns: u32) bool {
        if (w.mode != .fp8 or w.scales != 0 or w.k != width or w.n != columns or w.npad != columns) return false;
        const plan = qmmf.planFor(tile_rows, w, lane.clusters);
        return plan.cluster and plan.sk == split;
    }

    pub fn project(pair: Pair, lane: *const qmmf.Lane, s: Stream, x: u64, rows: usize, qkv: qmmf.Weight, z: qmmf.Weight, yq: u64, yz: u64) !bool {
        if (rows != tile_rows or !compatible(lane, qkv, qkv_columns) or !compatible(lane, z, z_columns)) return false;
        if (yq % 16 != 0 or yz % 16 != 0) return false;
        const overlaps = if (yq <= yz) yz - yq < tile_rows * qkv_columns * 2 else yq - yz < tile_rows * z_columns * 2;
        if (overlaps) return false;
        var args: launch.Args = .{};
        args.add(x);
        args.add(qkv.codes);
        args.add(z.codes);
        args.add(yq);
        args.add(yz);
        args.add(qkv.scale);
        args.add(z.scale);
        try launch.launch(pair.function, .{
            .grid = .{ .x = (qkv_columns + z_columns) / 64, .z = split },
            .block = .{ .x = 128 },
            .shared = shared,
            .cluster = .{ .x = 1, .y = 1, .z = split },
            .pdl = lane.pdl,
        }, s, &args);
        return true;
    }
};

test "FP8 QKV/Z pair leaves unsupported decode shapes and overlapping outputs on the original path" {
    const testing = @import("std").testing;
    var lane: qmmf.Lane = undefined;
    lane.clusters = true;
    const pair: Pair = .{ .function = undefined };
    const s: Stream = undefined;
    const qkv: qmmf.Weight = .{ .mode = .fp8, .codes = 16, .scales = 0, .n = 10240, .k = 5120, .npad = 10240 };
    const z: qmmf.Weight = .{ .mode = .fp8, .codes = 32, .scales = 0, .n = 6144, .k = 5120, .npad = 6144 };
    const yq = 64;
    const yz = yq + 16 * 10240 * 2;
    try testing.expect(!try pair.project(&lane, s, 0, 15, qkv, z, yq, yz));
    try testing.expect(!try pair.project(&lane, s, 0, 17, qkv, z, yq, yz));
    try testing.expect(!try pair.project(&lane, s, 0, 16, z, qkv, yq, yz));
    try testing.expect(!try pair.project(&lane, s, 0, 16, qkv, z, yq, yz - 16));
    try testing.expect(!try pair.project(&lane, s, 0, 16, qkv, try qmmf.pinned(z, 4), yq, yz));
    lane.clusters = false;
    try testing.expect(!try pair.project(&lane, s, 0, 16, qkv, z, yq, yz));
}
