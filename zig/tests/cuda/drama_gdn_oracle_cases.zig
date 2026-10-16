//! tf-drama-gdn-canonical-oracle's cases.json: geometries, input streams, initial states and ops, strictly checked.
const std = @import("std");
const drama = @import("drama");
const gdn = drama.backbone.gdn;
const Allocator = std.mem.Allocator;
const chunk = gdn.chunk_rows;
const gdn_oracle = @import("drama_gdn_canonical_oracle.zig");
const cases_schema = gdn_oracle.cases_schema;
const dv = gdn_oracle.dv;
const layer_count = gdn_oracle.layer_count;
const max_file = gdn_oracle.max_file;
const sha256Hex = gdn_oracle.sha256Hex;

const BadCases = error{BadCases};

fn bad(comptime f: []const u8, args: anytype) BadCases {
    std.debug.print("cases manifest: " ++ f ++ "\n", args);
    return error.BadCases;
}

const File = struct { name: []const u8, sha: []const u8, bytes: []align(16) const u8 };

pub const Geometry = struct {
    name: []const u8,
    nk: usize,
    nv: usize,
    a_log: File,
    dt_bias: File,
    max_rows: usize = 0, // largest case max_rows: output rows allocated
    ws_bytes: usize = 0, // largest Workspace.layout size over this geometry's cases
    used: bool = false,

    pub fn width(g: Geometry) usize {
        return (2 * g.nk + g.nv) * dv;
    }
    pub fn stateBytes(g: Geometry) usize {
        return g.nv * dv * dv * 4;
    }
    pub fn rowBytes(g: Geometry) usize {
        return g.nv * dv * 2;
    }
    pub fn qkvCacheBytes(g: Geometry) usize {
        return chunk * g.width() * 2;
    }
    pub fn baCacheBytes(g: Geometry) usize {
        return chunk * 2 * g.nv * 2;
    }
};

const Stream = struct { name: []const u8, geometry: usize, rows: usize, ld_qkv: usize, ld_ba: usize, qkv: File, ba: File };
const Init = struct { name: []const u8, geometry: usize, file: File };
const OpKind = enum { prefill, decode, reset };
pub const Op = struct {
    kind: OpKind,
    layer: usize,
    stream: usize = 0,
    start: usize = 0,
    rows: usize = 0,
    probe: bool = false,
    init: ?usize = null, // reset: null = zero state
    poison_cache: bool = false,
};
pub const Case = struct {
    name: []const u8,
    geometry: usize,
    max_rows: usize,
    init: [layer_count]?usize,
    ops: []const Op,
    save_old_states: bool,
    last_op: [layer_count]?usize, // the layer's last prefill/decode op
};

pub const Cases = struct {
    sha: [64]u8,
    geometries: []Geometry,
    streams: []Stream,
    inits: []Init,
    cases: []Case,
};

fn get(o: std.json.ObjectMap, key: []const u8) BadCases!std.json.Value {
    return o.get(key) orelse bad("missing key {s}", .{key});
}

fn uintOf(o: std.json.ObjectMap, key: []const u8) BadCases!usize {
    const v = try get(o, key);
    if (v != .integer or v.integer < 0) return bad("{s} is not a non-negative integer", .{key});
    return @intCast(v.integer);
}

fn textOf(o: std.json.ObjectMap, key: []const u8) BadCases![]const u8 {
    const v = try get(o, key);
    if (v != .string) return bad("{s} is not a string", .{key});
    return v.string;
}

fn boolOr(o: std.json.ObjectMap, key: []const u8, default: bool) BadCases!bool {
    const v = o.get(key) orelse return default;
    if (v != .bool) return bad("{s} is not a bool", .{key});
    return v.bool;
}

fn objectOf(v: std.json.Value, what: []const u8) BadCases!std.json.ObjectMap {
    if (v != .object) return bad("{s} is not an object", .{what});
    return v.object;
}

fn arrayOf(o: std.json.ObjectMap, key: []const u8) BadCases![]std.json.Value {
    const v = try get(o, key);
    if (v != .array) return bad("{s} is not an array", .{key});
    return v.array.items;
}

/// Names become file names: lowercase letters, digits, '_', '-', '.', not leading '.', at most 96 bytes.
fn safeName(s: []const u8) BadCases![]const u8 {
    if (s.len == 0 or s.len > 96 or s[0] == '.') return bad("unsafe name '{s}'", .{s});
    for (s) |c| switch (c) {
        'a'...'z', '0'...'9', '_', '-', '.' => {},
        else => return bad("unsafe name '{s}'", .{s}),
    };
    return s;
}

fn isSha256Hex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| switch (c) {
        '0'...'9', 'a'...'f' => {},
        else => return false,
    };
    return true;
}

fn loadFile(arena: Allocator, io: std.Io, dir: std.Io.Dir, v: std.json.Value, what: []const u8, size: usize) !File {
    const o = try objectOf(v, what);
    const name = try safeName(try textOf(o, "file"));
    const sha = try textOf(o, "sha256");
    if (!isSha256Hex(sha)) return bad("{s}: sha256 is not 64 lowercase hex digits", .{what});
    const bytes = dir.readFileAllocOptions(io, name, arena, .limited(max_file), .@"16", null) catch |err|
        return bad("{s}: {s} unreadable: {s}", .{ what, name, @errorName(err) });
    if (bytes.len != size) return bad("{s}: {s} has {d} bytes, expected {d}", .{ what, name, bytes.len, size });
    const got = sha256Hex(bytes);
    if (!std.mem.eql(u8, &got, sha)) return bad("{s}: {s} sha256 {s} is not the manifest's {s}", .{ what, name, got[0..], sha });
    return .{ .name = name, .sha = sha, .bytes = bytes };
}

fn findName(comptime T: type, items: []const T, name: []const u8, what: []const u8) BadCases!usize {
    for (items, 0..) |item, i| {
        if (std.mem.eql(u8, item.name, name)) return i;
    }
    return bad("unknown {s} '{s}'", .{ what, name });
}

pub fn workspaceCapacity(max_rows: usize) usize {
    return @max(chunk, max_rows / chunk * chunk);
}

pub fn parseCases(arena: Allocator, io: std.Io, dir: std.Io.Dir) !Cases {
    const text = dir.readFileAlloc(io, "cases.json", arena, .limited(1 << 24)) catch |err| return bad("cases.json unreadable: {s}", .{@errorName(err)});
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| return bad("cases.json is not JSON: {s}", .{@errorName(err)});
    const root = try objectOf(doc, "cases.json");
    if (!std.mem.eql(u8, try textOf(root, "schema"), cases_schema)) return bad("schema is not {s}", .{cases_schema});

    const gs = try arrayOf(root, "geometries");
    const geometries = try arena.alloc(Geometry, gs.len);
    for (gs, geometries) |v, *g| {
        const o = try objectOf(v, "geometry");
        const nk = try uintOf(o, "nk");
        const nv = try uintOf(o, "nv");
        if (nk == 0 or nv == 0 or nv % nk != 0) return bad("geometry nk {d} nv {d}: nv must be a positive multiple of nk", .{ nk, nv });
        if ((try uintOf(o, "dv")) != dv) return bad("geometry dv must be {d}", .{dv});
        g.* = .{
            .name = try safeName(try textOf(o, "name")),
            .nk = nk,
            .nv = nv,
            .a_log = try loadFile(arena, io, dir, try get(o, "a_log"), "a_log", nv * 4),
            .dt_bias = try loadFile(arena, io, dir, try get(o, "dt_bias"), "dt_bias", nv * 4),
        };
    }
    for (geometries, 0..) |g, i| for (geometries[0..i]) |h| if (std.mem.eql(u8, g.name, h.name)) return bad("duplicate geometry {s}", .{g.name});

    const ss = try arrayOf(root, "streams");
    const streams = try arena.alloc(Stream, ss.len);
    for (ss, streams) |v, *s| {
        const o = try objectOf(v, "stream");
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        const g = geometries[gi];
        const rows = try uintOf(o, "rows");
        const ld_qkv = try uintOf(o, "ld_qkv");
        const ld_ba = try uintOf(o, "ld_ba");
        if (rows == 0) return bad("stream with zero rows", .{});
        if (ld_qkv < g.width() or ld_ba < 2 * g.nv) return bad("stream strides {d}/{d} below the row widths {d}/{d}", .{ ld_qkv, ld_ba, g.width(), 2 * g.nv });
        if (ld_qkv % 8 != 0 or ld_ba % 8 != 0) return bad("stream strides must keep 16-byte rows", .{});
        s.* = .{
            .name = try safeName(try textOf(o, "name")),
            .geometry = gi,
            .rows = rows,
            .ld_qkv = ld_qkv,
            .ld_ba = ld_ba,
            .qkv = try loadFile(arena, io, dir, try get(o, "qkv"), "qkv", rows * ld_qkv * 2),
            .ba = try loadFile(arena, io, dir, try get(o, "ba"), "ba", rows * ld_ba * 2),
        };
    }

    const is = try arrayOf(root, "states");
    const inits = try arena.alloc(Init, is.len);
    for (is, inits) |v, *s| {
        const o = try objectOf(v, "state");
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        s.* = .{
            .name = try safeName(try textOf(o, "name")),
            .geometry = gi,
            .file = try loadFile(arena, io, dir, try get(o, "data"), "state", geometries[gi].stateBytes()),
        };
    }

    const cs = try arrayOf(root, "cases");
    const cases = try arena.alloc(Case, cs.len);
    for (cs, cases) |v, *c| {
        const o = try objectOf(v, "case");
        const name = try safeName(try textOf(o, "name"));
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        const max_rows = try uintOf(o, "max_rows");
        if (max_rows == 0) return bad("{s}: max_rows 0", .{name});
        const init_list = try arrayOf(o, "init");
        if (init_list.len != layer_count) return bad("{s}: init must name {d} layer states", .{ name, layer_count });
        var init: [layer_count]?usize = undefined;
        for (init_list, &init) |iv, *slot| slot.* = try initRef(inits, iv, gi, name);
        const ov = try arrayOf(o, "ops");
        if (ov.len == 0) return bad("{s}: no ops", .{name});
        const ops = try arena.alloc(Op, ov.len);
        var last: [layer_count]?usize = @splat(null);
        for (ov, ops, 0..) |item, *op, oi| {
            const p = try objectOf(item, "op");
            const kind = std.meta.stringToEnum(OpKind, try textOf(p, "op")) orelse return bad("{s} op {d}: unknown op", .{ name, oi });
            const layer = try uintOf(p, "layer");
            if (layer >= layer_count) return bad("{s} op {d}: layer {d}", .{ name, oi, layer });
            op.* = .{ .kind = kind, .layer = layer };
            switch (kind) {
                .reset => {
                    op.init = try initRef(inits, try get(p, "state"), gi, name);
                    op.poison_cache = try boolOr(p, "poison_cache", false);
                },
                .prefill, .decode => {
                    op.stream = try findName(Stream, streams, try textOf(p, "stream"), "stream");
                    op.start = try uintOf(p, "start");
                    op.rows = try uintOf(p, "rows");
                    op.probe = try boolOr(p, "probe", false);
                    const s = streams[op.stream];
                    if (s.geometry != gi) return bad("{s} op {d}: stream {s} has another geometry", .{ name, oi, s.name });
                    if (op.rows == 0 or op.rows > max_rows) return bad("{s} op {d}: rows {d} outside 1..max_rows {d}", .{ name, oi, op.rows, max_rows });
                    if (op.start + op.rows > s.rows) return bad("{s} op {d}: rows {d}+{d} beyond stream {s}", .{ name, oi, op.start, op.rows, s.name });
                    if (op.probe and kind != .prefill) return bad("{s} op {d}: probe on a decode", .{ name, oi });
                    last[layer] = oi;
                },
            }
        }
        c.* = .{
            .name = name,
            .geometry = gi,
            .max_rows = max_rows,
            .init = init,
            .ops = ops,
            .save_old_states = try boolOr(o, "save_old_states", false),
            .last_op = last,
        };
        const g = &geometries[gi];
        g.used = true;
        g.max_rows = @max(g.max_rows, max_rows);
        _, const ws_bytes = gdn.Workspace.layout(max_rows, g.nk, g.nv);
        g.ws_bytes = @max(g.ws_bytes, ws_bytes);
    }
    for (cases, 0..) |c, i| for (cases[0..i]) |d| if (std.mem.eql(u8, c.name, d.name)) return bad("duplicate case {s}", .{c.name});
    return .{ .sha = sha256Hex(text), .geometries = geometries, .streams = streams, .inits = inits, .cases = cases };
}

/// A state reference: the string "zero" or the name of a state of geometry `gi`.
fn initRef(inits: []const Init, v: std.json.Value, gi: usize, case_name: []const u8) BadCases!?usize {
    if (v != .string) return bad("{s}: state reference is not a string", .{case_name});
    if (std.mem.eql(u8, v.string, "zero")) return null;
    const i = try findName(Init, inits, v.string, "state");
    if (inits[i].geometry != gi) return bad("{s}: state {s} has another geometry", .{ case_name, v.string });
    return i;
}

// recorded failures, hashing, JSON
