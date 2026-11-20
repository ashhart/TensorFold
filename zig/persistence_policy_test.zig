//! A fixed probe cannot stand in for resolved policies or the actual loaded weight pack.
const std = @import("std");
const fixture = @import("src/native/persistence_fixture.zig");
const qwen = @import("src/native/qwen27_disk.zig");
const glm = @import("src/native/glm_disk.zig");
const flash = @import("src/native/flashnext_disk.zig");
const nemo = @import("src/native/metal_identity.zig");
const t = std.testing;

// These expose only the resolved fields the production key producers consume, without a GPU owner.
test "persistence identity: Qwen Metal loaded ANE columns miss with the same probe" {
    const Share = struct { columns: u32 };
    var share: Share = .{ .columns = 1024 };
    var h: struct { model: struct { ane: ?*Share, frame: struct { capacity: u32 } }, back: struct { chunk: usize } } = .{ .model = .{ .ane = @as(?*Share, &share), .frame = .{ .capacity = @as(u32, 128) } }, .back = .{ .chunk = @as(usize, 128) } };
    const before = qwen.withPolicy(7, &h);
    const matching = qwen.withPolicy(7, &h);
    share.columns = 2048;
    const other_share = qwen.withPolicy(7, &h);
    share.columns = 1024;
    h.model.frame.capacity = 256; // buffer rows only: the same states
    try t.expectEqual(before, qwen.withPolicy(7, &h));
    h.back.chunk = 256; // another chunk height starts cold
    try fixture.misses(before, matching, &.{ other_share, qwen.withPolicy(7, &h) });
}

test "persistence identity: Qwen Metal control vector and strength miss with the same probe" {
    var file_sha: [32]u8 = @splat(0x13);
    const before = qwen.withSteering(7, &file_sha, 0.75);
    const matching = qwen.withSteering(7, &file_sha, 0.75);
    const stronger = qwen.withSteering(7, &file_sha, 1.5);
    file_sha[31] ^= 1;
    try fixture.misses(before, matching, &.{ qwen.withSteering(7, &file_sha, 0.75), stronger, 7 });
}

test "persistence identity: GLM Metal resolved policy misses with the same probe" {
    const Config = @import("src/families/glm/config.zig").Config;
    const Prompt = struct { rows: u32 = 4096, sparse_nax: bool = true };
    const Head = struct { eh_bf16: ?u8 = 1 };
    const Wide = struct { bf16_nax: ?u8 = 1 };
    var prompt: Prompt = .{};
    var weights: struct { mtp: ?Head = .{} } = .{};
    var kernels: struct { wide: ?Wide = .{} } = .{};
    var e: struct { c: Config, pr: ?*Prompt, fused_route: bool, w: @TypeOf(&weights), k: @TypeOf(&kernels) } = .{ .c = Config{}, .pr = @as(?*Prompt, &prompt), .fused_route = true, .w = &weights, .k = &kernels };
    try t.expect(glm.ehTile(&e));
    const before = glm.withPolicy(7, &e);
    var changed: std.ArrayList(u64) = .empty;
    defer changed.deinit(t.allocator);
    e.fused_route = false;
    try changed.append(t.allocator, glm.withPolicy(7, &e));
    e.fused_route = true;
    inline for (.{ "run", "tp_rank", "dense_inter", "kda_heads", "mla_heads" }) |field| {
        @field(e.c, field) += 1;
        try changed.append(t.allocator, glm.withPolicy(7, &e));
        @field(e.c, field) -= 1;
    }
    inline for (.{ "own", "inter" }) |field| {
        @field(e.c, field)[0] += 1;
        try changed.append(t.allocator, glm.withPolicy(7, &e));
        @field(e.c, field)[0] -= 1;
    }
    prompt.sparse_nax = false;
    try changed.append(t.allocator, glm.withPolicy(7, &e));
    prompt.sparse_nax = true;
    prompt.rows = 8192; // another chunk height starts cold
    try changed.append(t.allocator, glm.withPolicy(7, &e));
    prompt.rows = 4096;
    kernels.wide.?.bf16_nax = null; // GLM_EHPROJ_NAX=0 or a failed tile build: the prompt's eh_proj is scalar
    try t.expect(!glm.ehTile(&e));
    const scalar = glm.withPolicy(7, &e);
    try changed.append(t.allocator, scalar);
    kernels.wide = null;
    try t.expectEqual(scalar, glm.withPolicy(7, &e));
    kernels.wide = .{};
    weights.mtp.?.eh_bf16 = null; // a quantized head runs neither bf16 kernel
    try t.expectEqual(scalar, glm.withPolicy(7, &e));
    weights.mtp = null;
    try t.expectEqual(scalar, glm.withPolicy(7, &e));
    weights.mtp = .{};
    e.pr = null;
    try t.expect(!glm.ehTile(&e));
    try changed.append(t.allocator, glm.withPolicy(7, &e));
    e.pr = &prompt;
    try fixture.misses(before, glm.withPolicy(7, &e), changed.items);
}

const flags = .{ "fused_xsum", "dense", "dense_target", "split", "gdn_step", "hc_mma", "grouped", "xpack", "xfused", "xsx", "copy", "prefetch", "lane_new", "rotary_enabled" };
const Gx = struct { min_rows: usize };
const Tp = struct { rank: u32 };
const Run = struct {
    fused_xsum: bool = false,
    serial: bool = false,
    dense: bool = false,
    dense_target: bool = false,
    split: bool = false,
    gdn_step: bool = false,
    hc_mma: bool = false,
    grouped: bool = false,
    xpack: bool = false,
    xfused: bool = false,
    xsx: bool = false,
    copy: bool = false,
    prefetch: bool = false,
    lane_new: bool = false,
    rotary_enabled: bool = false,
    skip: u32 = 0,
    hcskip: u32 = 0,
    gskip: u32 = 0,
    lane_pf: usize = 1,
    gdn_pipe: ?u8 = null,
    gdn_kept: ?u8 = null,
    gx: ?*const Gx = null,
    tp: ?*const Tp = null,
};

test "persistence identity: Flash Next Metal resolved branches miss with the same probe" {
    var r: Run = .{};
    const before = flash.withPolicy(7, &r);
    r.serial = true; // a serial encoder in place of barriers: the same launches in the same order
    try t.expectEqual(before, flash.withPolicy(7, &r));
    r.serial = false;
    var changed: std.ArrayList(u64) = .empty;
    defer changed.deinit(t.allocator);
    inline for (flags) |field| {
        @field(r, field) = true;
        try changed.append(t.allocator, flash.withPolicy(7, &r));
        @field(r, field) = false;
    }
    inline for (.{ "skip", "hcskip", "gskip", "lane_pf" }) |field| {
        @field(r, field) += 1;
        try changed.append(t.allocator, flash.withPolicy(7, &r));
        @field(r, field) -= 1;
    }
    r.gdn_pipe = 1;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    r.gdn_pipe = null;
    r.gdn_kept = 1;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    r.gdn_kept = null;
    var gx: Gx = .{ .min_rows = 4 };
    r.gx = &gx;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    gx.min_rows = 8;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    r.gx = null;
    var tp: Tp = .{ .rank = 0 };
    r.tp = &tp;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    tp.rank = 1;
    try changed.append(t.allocator, flash.withPolicy(7, &r));
    r.tp = null;
    try fixture.misses(before, flash.withPolicy(7, &r), changed.items);
}

test "persistence identity: Nemotron Metal coop K-slice split misses with the same probe" {
    var b: struct { coop_split: bool, scratch: struct { rows: usize } } = .{ .coop_split = true, .scratch = .{ .rows = 64 } };
    const before = nemo.withPolicy(7, &b);
    const matching = nemo.withPolicy(7, &b);
    b.scratch.rows = 128; // a chunk of 65 to 128 rows splits only when the scratch holds its partials
    const wider = nemo.withPolicy(7, &b);
    b.coop_split = false; // TF_COOP_SPLIT=0
    const off = nemo.withPolicy(7, &b);
    b.scratch.rows = 64; // with no split, the scratch rows decide nothing
    try t.expectEqual(off, nemo.withPolicy(7, &b));
    try fixture.misses(before, matching, &.{ wider, off });
}

test "persistence identity: Flash Next Metal actual pack rewrite misses with the same source and probe" {
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"unprobed.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var pack_bytes: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, pack_bytes[0..8], header.len, .little);
    @memcpy(pack_bytes[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pack.safetensors", .data = &pack_bytes });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "other.safetensors", .data = &pack_bytes });
    const file = try tmp.dir.openFile(t.io, "pack.safetensors", .{});
    defer file.close(t.io);
    const other = try tmp.dir.openFile(t.io, "other.safetensors", .{});
    defer other.close(t.io);
    const files = [_]c_int{ file.handle, other.handle };
    const before = try flash.withFiles(7, a, t.io, &files);
    const matching = try flash.withFiles(7, a, t.io, &.{ other.handle, file.handle });
    pack_bytes[pack_bytes.len - 1] = 1;
    try fixture.rewrite(tmp.dir, "pack.safetensors", &pack_bytes);
    try fixture.misses(before, matching, &.{try flash.withFiles(7, a, t.io, &files)});
    try t.expectError(error.IdentityFileUnavailable, flash.withFiles(7, a, t.io, &.{-1}));
}

test "persistence identity: Flash Next Metal mapped reference bytes miss with the same probe" {
    var first = [_]u8{ 1, 2, 3, 4 };
    const second = [_]u8{ 5, 6, 7, 8 };
    const Map = struct { bytes: []const u8, size: usize };
    const maps = [_]Map{ .{ .bytes = &first, .size = 4 }, .{ .bytes = &second, .size = 4 } };
    var run = .{ .files = .{ .items = @as([]const c_int, &.{}) }, .maps = .{ .items = @as([]const Map, &maps) } };
    const before = try flash.withInputs(7, t.allocator, t.io, &run);
    const matching = try flash.withInputs(7, t.allocator, t.io, &run);
    first[3] ^= 1;
    try fixture.misses(before, matching, &.{try flash.withInputs(7, t.allocator, t.io, &run)});
    first[3] ^= 1;
    const reversed = [_]Map{ maps[1], maps[0] };
    run.maps.items = &reversed;
    try t.expectEqual(before, try flash.withInputs(7, t.allocator, t.io, &run));
}
