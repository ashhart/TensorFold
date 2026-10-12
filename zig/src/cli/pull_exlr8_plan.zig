//! What one EXLR8 node fetches by range: its halves of every expert at the manifest's widths, as checked runs.
const std = @import("std");
const Allocator = std.mem.Allocator;
const exlr8 = @import("exlr8");
const single = @import("pull_file.zig");
const runs = @import("pull_runs.zig");

const Proj = exlr8.experts.Proj;
/// The header every width file starts with; trellis data begins here.
pub const header_bytes: u64 = 65536;
/// A run's most bytes (a node of one: whole experts back to back).
const max_run: u64 = 16 << 20;

/// experts/exlr8-layer-LLL.kK.safetensors: a node holds only some of its bytes.
pub fn isWidthFile(path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, "experts/exlr8-layer-")) return false;
    for ([_][]const u8{ ".k2.safetensors", ".k3.safetensors", ".k4.safetensors" }) |s| if (std.mem.endsWith(u8, path, s)) return true;
    return false;
}

/// A profile names a manifest file: letters, digits, '-' and '_' only.
pub fn profileOk(p: []const u8) bool {
    if (p.len == 0 or p.len > 64) return false;
    for (p) |c| if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_')) return false;
    return true;
}

/// The weight files' published hashes equal the hub's sha256 and sizes, and give the manifest's set_id; returns it.
pub fn checkPublished(a: Allocator, io: std.Io, dir: []const u8, entries: []const single.Entry, manifest_rel: []const u8) ![]const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    var buf: [512]u8 = undefined;
    for (entries) |e| {
        const weight = (std.mem.startsWith(u8, e.path, "experts/") or std.mem.startsWith(u8, e.path, "dense/")) and std.mem.endsWith(u8, e.path, ".safetensors");
        if (!weight) continue;
        const hp = try std.fs.path.join(a, &.{ dir, try exlr8.release.hashPath(&buf, e.path) });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, hp, a, .limited(4096));
        const want = try exlr8.release.parsePublished(text);
        const sha = e.sha256 orelse return error.WeightNotLfs;
        if (want.size != e.size or !std.mem.eql(u8, &want.sha256, &sha)) return error.PublishedHashDiffers;
        try lines.append(a, try std.fmt.allocPrint(a, "{s} {d} {s}", .{ e.path, e.size, std.fmt.bytesToHex(sha, .lower) }));
    }
    std.mem.sort([]const u8, lines.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    const id = std.fmt.bytesToHex(exlr8.node.setId(lines.items), .lower);
    var m = try exlr8.manifest.Manifest.load(a, io, try std.fs.path.join(a, &.{ dir, manifest_rel }));
    defer m.deinit();
    if (!std.mem.eql(u8, m.set_id, &id)) return error.SetIdDiffers;
    return a.dupe(u8, &id);
}

/// One width file the node fills in part.
pub const Target = struct { rel: []const u8, size: u64, layer: u16, k: u8 };

pub const Plan = struct {
    targets: std.ArrayList(Target) = .empty,
    units: std.ArrayList(exlr8.experts.Entry) = .empty,
    runs: std.ArrayList(runs.Run) = .empty,
    layers: std.ArrayList(exlr8.experts.Layer) = .empty,
    expert_bytes: u64 = 0,

    /// The node's halves of every MoE layer the manifest names, at its widths, in runs of adjacent halves.
    pub fn build(a: Allocator, io: std.Io, dir: []const u8, manifest_rel: []const u8, node: exlr8.node.Node, widths: std.StringHashMapUnmanaged(u64)) !Plan {
        var p: Plan = .{};
        errdefer p.deinit(io);
        var m = try exlr8.manifest.Manifest.load(a, io, try std.fs.path.join(a, &.{ dir, manifest_rel }));
        defer m.deinit();
        for (0..exlr8.manifest.layers) |li| {
            if (m.layers[li] == null) continue;
            const number: u16 = @intCast(li);
            try p.layers.append(a, try exlr8.experts.Layer.openIndex(a, io, dir, number));
            const layer = &p.layers.items[p.layers.items.len - 1];
            var per: [5]std.ArrayList(exlr8.experts.Entry) = @splat(.empty);
            for (0..exlr8.experts.shared + 1) |ei| for ([_]Proj{ .gate, .up, .down }) |proj| for (node.first()..node.end()) |h| {
                const e: u16 = @intCast(ei);
                const k = m.width(li, e, switch (proj) {
                    .gate => .gate,
                    .up => .up,
                    .down => .down,
                }, h) orelse return error.BadExlr8Manifest;
                try per[k].append(a, try layer.entry(e, proj, h, k));
            };
            for (2..5) |ki| if (per[ki].items.len > 0) {
                const k: u8 = @intCast(ki);
                const rel = try std.fmt.allocPrint(a, "experts/exlr8-layer-{d:0>3}.k{d}.safetensors", .{ number, k });
                const size = widths.get(rel) orelse return error.WidthFileNotListed;
                try p.add(a, .{ .rel = rel, .size = size, .layer = number, .k = k }, per[ki].items);
            };
        }
        return p;
    }

    fn add(p: *Plan, a: Allocator, t: Target, units: []exlr8.experts.Entry) !void {
        std.mem.sort(exlr8.experts.Entry, units, {}, struct {
            fn lt(_: void, x: exlr8.experts.Entry, y: exlr8.experts.Entry) bool {
                return x.off < y.off;
            }
        }.lt);
        const file: u32 = @intCast(p.targets.items.len);
        try p.targets.append(a, t);
        const unit0: u32 = @intCast(p.units.items.len);
        try p.runs.append(a, .{ .file = file, .first = 0, .end = header_bytes, .units = .{ unit0, unit0 } });
        const offs = try a.alloc([2]u64, units.len);
        for (units, offs, 0..) |u, *o, i| {
            if (u.off < header_bytes or u.off + u.len > t.size) return error.BadExlr8Offset;
            if (i > 0 and u.off < units[i - 1].off + units[i - 1].len) return error.BadExlr8Offset;
            o.* = .{ u.off, u.len };
            p.expert_bytes += u.len;
        }
        try p.units.appendSlice(a, units);
        try runs.coalesce(a, file, offs, unit0, max_run, &p.runs);
    }

    pub fn deinit(p: *Plan, io: std.Io) void {
        for (p.layers.items) |*l| l.close(io);
        p.* = undefined;
    }

    pub fn check(p: *const Plan) runs.Check {
        return .{ .ctx = p, .run = checkRun };
    }

    /// Runs whose bytes on disk fail their checks (all of a target's when its file is absent or the wrong size).
    pub fn missing(p: *const Plan, a: Allocator, io: std.Io, targets: []const runs.Target) ![]runs.Run {
        const bad = try a.alloc(bool, p.runs.items.len);
        @memset(bad, false);
        var s: Scan = .{ .p = p, .io = io, .targets = targets, .bad = bad };
        var threads: [8]std.Thread = undefined;
        for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Scan.work, .{&s});
        for (threads) |t| t.join();
        if (s.failed.load(.acquire)) return error.ScanFailed;
        var out: std.ArrayList(runs.Run) = .empty;
        for (p.runs.items, bad) |r, b| if (b) try out.append(a, r);
        return out.items;
    }
};

fn checkRun(ctx: *const anyopaque, r: runs.Run, bytes: []const u8) bool {
    const p: *const Plan = @ptrCast(@alignCast(ctx));
    if (r.units[0] == r.units[1]) return headerOk(p.targets.items[r.file], bytes);
    for (p.units.items[r.units[0]..r.units[1]]) |u| if (!u.verify(bytes[@intCast(u.off - r.first)..][0..u.len])) return false;
    return true;
}

/// A width file's header: a safetensors index of this layer's trellis at this width, data from byte 65536.
fn headerOk(t: Target, bytes: []const u8) bool {
    if (bytes.len != header_bytes) return false;
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    if (n + 8 != header_bytes) return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes[8..], .{}) catch return false;
    if (v != .object) return false;
    const meta = v.object.get("__metadata__") orelse return false;
    if (meta != .object) return false;
    var buf: [8]u8 = undefined;
    const want = [_][2][]const u8{
        .{ "format", "exlr8-v2" },
        .{ "contents", "trellis" },
        .{ "layer", std.fmt.bufPrint(buf[0..4], "{d}", .{t.layer}) catch return false },
        .{ "k", std.fmt.bufPrint(buf[4..], "{d}", .{t.k}) catch return false },
    };
    for (want) |kv| {
        const got = meta.object.get(kv[0]) orelse return false;
        if (got != .string or !std.mem.eql(u8, got.string, kv[1])) return false;
    }
    return true;
}

const Scan = struct {
    p: *const Plan,
    io: std.Io,
    targets: []const runs.Target,
    bad: []bool,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn work(s: *Scan) void {
        const buf = std.heap.page_allocator.alloc(u8, max_run) catch {
            s.failed.store(true, .release);
            return;
        };
        defer std.heap.page_allocator.free(buf);
        var open_file: ?u32 = null;
        var handle: ?std.Io.File = null;
        defer if (handle) |h| h.close(s.io);
        while (true) {
            const i = s.next.fetchAdd(1, .monotonic);
            if (i >= s.p.runs.items.len) return;
            const r = s.p.runs.items[i];
            if (open_file != r.file) {
                if (handle) |h| h.close(s.io);
                handle = null;
                open_file = r.file;
                const t = s.targets[r.file];
                const f = std.Io.Dir.cwd().openFile(s.io, t.path, .{}) catch null;
                if (f) |file| {
                    if ((file.length(s.io) catch 0) == t.size) handle = file else file.close(s.io);
                }
            }
            const h = handle orelse {
                s.bad[i] = true;
                continue;
            };
            const bytes = buf[0..@intCast(r.end - r.first)];
            const n = h.readPositionalAll(s.io, bytes, r.first) catch 0;
            s.bad[i] = n != bytes.len or !checkRun(s.p, r, bytes);
        }
    }
};

test "width files and profiles are told apart from the rest of the release" {
    try std.testing.expect(isWidthFile("experts/exlr8-layer-040.k3.safetensors"));
    try std.testing.expect(!isWidthFile("experts/exlr8-layer-040.index.safetensors"));
    try std.testing.expect(!isWidthFile("dense/exlr8-dense-040.safetensors"));
    try std.testing.expect(profileOk("home") and profileOk("bit325") and !profileOk("../x") and !profileOk(""));
}

test "a width file's header names its layer and width" {
    var bytes: [header_bytes]u8 = @splat(' ');
    const json = "{\"__metadata__\":{\"format\":\"exlr8-v2\",\"contents\":\"trellis\",\"layer\":\"40\",\"k\":\"3\"}}";
    std.mem.writeInt(u64, bytes[0..8], header_bytes - 8, .little);
    @memcpy(bytes[8..][0..json.len], json);
    try std.testing.expect(headerOk(.{ .rel = "", .size = 0, .layer = 40, .k = 3 }, &bytes));
    try std.testing.expect(!headerOk(.{ .rel = "", .size = 0, .layer = 40, .k = 4 }, &bytes));
    try std.testing.expect(!headerOk(.{ .rel = "", .size = 0, .layer = 3, .k = 3 }, &bytes));
}
