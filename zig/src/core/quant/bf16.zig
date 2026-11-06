//! Weights kept as stored: BF16 [N, K] projections of an unquantized checkpoint, multiplied by rocBLAS as torch does.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;

pub const id: types.Format = .bf16;
pub const decoder: types.Decoder = .bf16;

/// The format has no settings: the checkpoint declares it by declaring no quantization at all.
pub const Config = void;

/// A config with neither `quantization` nor `quantization_config` (root or text tower): its weights are as trained.
pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    if (src.quantization.count() > 0 or declares(src.root)) return null;
    if (src.root.get("text_config")) |t| if (t == .object and declares(t.object)) return null;
    return {};
}

fn declares(o: std.json.ObjectMap) bool {
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (o.get(key)) |v| if (v != .null) return true;
    }
    return false;
}

pub const Host = struct { weight: Tensor };

/// Whether `key` holds a BF16 matrix with no table of scales beside it.
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    const w = t.get(join(&buf, &.{ key, ".weight" })) catch return false;
    return w.dtype == .bf16 and w.rank == 2 and !t.has(join(&buf, &.{ key, ".scales" }));
}

/// Why a stored weight cannot be a bf16 projection, or null when it can.
pub fn unfit(w: Tensor) ?[]const u8 {
    if (w.dtype != .bf16) return "a bf16 checkpoint's projection must be BF16";
    if (w.rank != 2) return "a bf16 projection must be [N, K]";
    return null;
}

/// The BF16 weight as mapped; any other dtype or rank is refused by name.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    _ = a;
    var buf: [256]u8 = undefined;
    const name = join(&buf, &.{ key, ".weight" });
    const w = try t.get(name);
    if (unfit(w)) |why| return types.refuse(name, why);
    return .{ .weight = w };
}

pub fn rows(h: Host) usize {
    return h.weight.shape[0];
}

pub fn bytes(h: Host) usize {
    return h.weight.bytes.len;
}

/// Both halves along K, copied.
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const half = p.weight.shape[1] / 2;
    return .{
        .{ .weight = try slice.takeCols(a, p.weight, 0, half) },
        .{ .weight = try slice.takeCols(a, p.weight, half, p.weight.shape[1]) },
    };
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    return .{ .weight = try slice.takeRows(a, p.weight, spans) };
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const width = try slice.even(p.weight.shape[1], r.world, what);
    return .{ .weight = try slice.takeCols(a, p.weight, r.rank * width, (r.rank + 1) * width) };
}

pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{ .weight = try slice.takeExperts(a, p.weight, r, part) };
}

pub const Device = struct { weight: types.Buf };

pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{ .weight = try u.tensor(h.weight) };
}

/// What the products read: the BF16 weight's address.
pub const Matrix = struct { weight: u64 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    if (stacked or d.weight.dtype != .bf16) return error.UnsupportedTables;
    return .{ .n = @intCast(d.weight.dim(0)), .k = @intCast(d.weight.dim(1)), .matrix = .{ .weight = d.weight.ptr } };
}

pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// The weight as fp64, row by row (BF16 widens exactly).
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    if (out.len != h.weight.numel()) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = convert.load(.bf16, h.weight.bytes, i);
}

/// A config document for the tests, its root object.
fn parsed(a: Allocator, text: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

fn sources(root: std.json.ObjectMap) types.Sources {
    const q = if (root.get("quantization")) |v| (if (v == .object) v.object else std.json.ObjectMap.empty) else std.json.ObjectMap.empty;
    return .{ .root = root, .quantization = q };
}

test "bf16 is detected only when the config declares no quantization anywhere" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try parsed(a, "{\"model_type\": \"qwen3_5\", \"text_config\": {\"hidden_size\": 8}}");
    try std.testing.expect(try detect(a, sources(plain)) != null);
    const nulls = try parsed(a, "{\"quantization_config\": null, \"text_config\": {\"quantization\": null}}");
    try std.testing.expect(try detect(a, sources(nulls)) != null);
    const refused = [_][]const u8{
        "{\"quantization\": {\"bits\": 4, \"group_size\": 64}}",
        "{\"quantization_config\": {\"quant_method\": \"fp8\"}}",
        "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"awq\"}}}",
        "{\"text_config\": {\"quantization\": {\"bits\": 8}}}",
    };
    for (refused) |text| try std.testing.expect(try detect(a, sources(try parsed(a, text))) == null);
}

const Fake = struct {
    w: Tensor,
    scales: bool = false,

    pub fn get(f: *const Fake, key: []const u8) !Tensor {
        if (!std.mem.endsWith(u8, key, ".weight")) return error.MissingTensor;
        return f.w;
    }

    pub fn has(f: *const Fake, key: []const u8) bool {
        return f.scales and std.mem.endsWith(u8, key, ".scales");
    }
};

test "a bf16 projection is read as stored; another dtype or rank is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var data = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0 };
    const w: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 3, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&data) };
    const ok: Fake = .{ .w = w };
    try std.testing.expect(matches(&ok, "x"));
    const h = try read(a, &ok, "x");
    try std.testing.expectEqual(@intFromPtr(w.bytes.ptr), @intFromPtr(h.weight.bytes.ptr));
    var out: [6]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 5, 6 }, &out);
    const low = try sliceRows(a, h, &.{.{ .from = 1, .to = 2 }});
    try std.testing.expectEqualSlices(u16, data[3..], @alignCast(std.mem.bytesAsSlice(u16, low.weight.bytes)));
    try std.testing.expect(unfit(w) == null);
    var cube = w;
    cube.rank = 3;
    try std.testing.expect(!matches(&Fake{ .w = cube }, "x"));
    try std.testing.expect(unfit(cube) != null);
    var wide = w;
    wide.dtype = .f32;
    wide.shape = .{ 1, 3, 1, 1, 1 };
    try std.testing.expect(!matches(&Fake{ .w = wide }, "x"));
    try std.testing.expect(unfit(wide) != null);
    try std.testing.expect(!matches(&Fake{ .w = w, .scales = true }, "x"));
}

test "a view is the weight's address and shape; a stack is refused" {
    const d: Device = .{ .weight = .{ .ptr = 4096, .len = 12, .dtype = .bf16, .rank = 2, .shape = .{ 2, 3, 1, 1, 1 } } };
    const v = try view(d, false);
    try std.testing.expectEqual(@as(u32, 2), v.n);
    try std.testing.expectEqual(@as(u32, 3), v.k);
    try std.testing.expectEqual(@as(u64, 4096), v.matrix.weight);
    try std.testing.expectError(error.UnsupportedTables, view(d, true));
}
