//! A fixed probe cannot stand in for resolved policies or the actual loaded weight pack.
const std = @import("std");
const fixture = @import("src/native/persistence_fixture.zig");
const qwen = @import("src/native/qwen27_disk.zig");
const glm = @import("src/native/glm_disk.zig");
const flash = @import("src/native/flashnext_disk.zig");
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
    h.model.frame.capacity = 256;
    const other_frame = qwen.withPolicy(7, &h);
    h.model.frame.capacity = 128;
    h.back.chunk = 256;
    try fixture.misses(before, matching, &.{ other_share, other_frame, qwen.withPolicy(7, &h) });
}

test "persistence identity: GLM Metal resolved policy misses with the same probe" {
    const Config = @import("src/families/glm/config.zig").Config;
    const Prompt = struct { rows: u32 = 4096, sparse_nax: bool = true };
    var prompt: Prompt = .{};
    var e: struct { c: Config, pr: ?*Prompt, fused_route: bool } = .{ .c = Config{}, .pr = @as(?*Prompt, &prompt), .fused_route = true };
    const before = glm.withPolicy(7, &e, true);
    var changed: std.ArrayList(u64) = .empty;
    defer changed.deinit(t.allocator);
    e.fused_route = false;
    try changed.append(t.allocator, glm.withPolicy(7, &e, true));
    e.fused_route = true;
    inline for (.{ "run", "tp_rank", "dense_inter", "kda_heads", "mla_heads" }) |field| {
        @field(e.c, field) += 1;
        try changed.append(t.allocator, glm.withPolicy(7, &e, true));
        @field(e.c, field) -= 1;
    }
    inline for (.{ "own", "inter" }) |field| {
        @field(e.c, field)[0] += 1;
        try changed.append(t.allocator, glm.withPolicy(7, &e, true));
        @field(e.c, field)[0] -= 1;
    }
    prompt.sparse_nax = false;
    try changed.append(t.allocator, glm.withPolicy(7, &e, true));
    prompt.sparse_nax = true;
    prompt.rows = 8192;
    try changed.append(t.allocator, glm.withPolicy(7, &e, true));
    prompt.rows = 4096;
    try changed.append(t.allocator, glm.withPolicy(7, &e, false));
    e.pr = null;
    try changed.append(t.allocator, glm.withPolicy(7, &e, true));
    e.pr = &prompt;
    try fixture.misses(before, glm.withPolicy(7, &e, true), changed.items);
}

const flags = .{ "fused_xsum", "serial", "dense", "dense_target", "split", "gdn_step", "hc_mma", "grouped", "xpack", "xfused", "xsx", "copy", "prefetch", "lane_new", "rotary_enabled" };
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
    try std.Io.sleep(t.io, .fromMilliseconds(5), .awake);
    pack_bytes[pack_bytes.len - 1] = 1;
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pack.safetensors", .data = &pack_bytes });
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
