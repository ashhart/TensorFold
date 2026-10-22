//! A JointHead beside the 27B's trunk: option-final hidden rows in, one score an option out.
const std = @import("std");
const tf = @import("tensorfold");
const st = tf.safetensors;
const Allocator = std.mem.Allocator;

/// `joint_head_config.json`: the head's shape and its weight file's format.
pub const Config = struct {
    dim: usize, // the trunk's hidden width
    width: usize, // the head's
    layers: usize, // Linear→GELU residual blocks
};

/// The head: `proj` (dim→width), Linear→GELU residual layers, `out` (width→1); bf16 words read from the file.
pub const Head = struct {
    gpa: Allocator,
    io: std.Io,
    file: st.File,
    cfg: Config,
    data: []u16, // every weight, one owned row: proj, each block, out
    proj_w: []const u16, // [width, dim]
    proj_b: []const u16, // [width]
    block_w: [][]const u16, // a [width, width] a block
    block_b: [][]const u16, // a [width] a block
    out_w: []const u16, // [width]
    out_b: u16,
    x: []f32, // a row after proj and each residual
    y: []f32, // a block's Linear→GELU half
    h: []f32, // a hidden row as f32

    /// `path`'s head: `joint_head_config.json` plus its one safetensors (a directory), or the file beside the config.
    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Head {
        var dir_path = path;
        var name: ?[]const u8 = null;
        if (std.mem.endsWith(u8, path, ".safetensors")) {
            name = std.fs.path.basename(path);
            dir_path = std.fs.path.dirname(path) orelse ".";
        }
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        const text = try dir.readFileAlloc(io, "joint_head_config.json", gpa, .limited(1 << 20));
        defer gpa.free(text);
        const cfg = try parseConfig(gpa, text);
        const file_name = name orelse try oneShard(gpa, io, dir_path);
        defer if (name == null) gpa.free(file_name);
        const file_path = try std.fs.path.join(gpa, &.{ dir_path, file_name });
        defer gpa.free(file_path);
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .io = io, .file = try st.File.open(gpa, io, file_path), .cfg = cfg, .data = &.{}, .proj_w = &.{}, .proj_b = &.{}, .block_w = &.{}, .block_b = &.{}, .out_w = &.{}, .out_b = 0, .x = &.{}, .y = &.{}, .h = &.{} };
        errdefer h.file.close(io);
        h.h = try gpa.alloc(f32, cfg.dim);
        errdefer gpa.free(h.h);
        h.x = try gpa.alloc(f32, cfg.width);
        errdefer gpa.free(h.x);
        h.y = try gpa.alloc(f32, cfg.width);
        errdefer gpa.free(h.y);
        h.data = try gpa.alloc(u16, cfg.width * cfg.dim + cfg.width + cfg.layers * cfg.width * (cfg.width + 1) + cfg.width + 1);
        errdefer gpa.free(h.data);
        h.block_w = try gpa.alloc([]const u16, cfg.layers);
        errdefer gpa.free(h.block_w);
        h.block_b = try gpa.alloc([]const u16, cfg.layers);
        errdefer gpa.free(h.block_b);
        var off: usize = 0;
        h.proj_w = try h.part(&off, "proj.weight", &.{ cfg.width, cfg.dim });
        h.proj_b = try h.part(&off, "proj.bias", &.{cfg.width});
        for (0..cfg.layers) |i| {
            var buf: [64]u8 = undefined;
            h.block_w[i] = try h.part(&off, try std.fmt.bufPrint(&buf, "blocks.{d}.0.weight", .{i}), &.{ cfg.width, cfg.width });
            h.block_b[i] = try h.part(&off, try std.fmt.bufPrint(&buf, "blocks.{d}.0.bias", .{i}), &.{cfg.width});
        }
        h.out_w = try h.part(&off, "out.weight", &.{ 1, cfg.width });
        h.out_b = (try h.part(&off, "out.bias", &.{1}))[0];
        return h;
    }

    pub fn close(h: *Head) void {
        h.gpa.free(h.block_b);
        h.gpa.free(h.block_w);
        h.gpa.free(h.data);
        h.gpa.free(h.y);
        h.gpa.free(h.x);
        h.gpa.free(h.h);
        h.file.close(h.io);
        h.gpa.destroy(h);
    }

    /// One option's score: `hidden` a post-final-norm row of the trunk (`dim` bf16 words).
    pub fn score(h: *Head, hidden: []const u16) error{Failed}!f64 {
        if (hidden.len != h.cfg.dim) return error.Failed;
        for (h.h, hidden) |*d, bits| {
            const v: f32 = @bitCast(@as(u32, bits) << 16);
            if (!std.math.isFinite(v)) return error.Failed;
            d.* = v;
        }
        matvec(h.proj_w, h.cfg.dim, h.h, h.proj_b, h.x);
        for (h.block_w, h.block_b) |w, bias| {
            matvec(w, h.cfg.width, h.x, bias, h.y);
            for (h.x, h.y) |*a, b| a.* += gelu(b);
        }
        var s: f64 = f32of(h.out_b);
        for (h.x, h.out_w) |a, bits| s += @as(f64, f32of(bits)) * a;
        return if (std.math.isFinite(s)) s else error.Failed;
    }

    /// `data`'s next `shape` elements taken as the bf16 tensor `name`, checked.
    fn part(h: *Head, off: *usize, name: []const u8, shape: []const usize) ![]const u16 {
        const t = h.file.get(name) orelse return error.MissingTensor;
        if (!t.is(.bf16, shape)) return error.UnexpectedTensor;
        const dest = h.data[off.*..][0..t.numel()];
        @memcpy(std.mem.sliceAsBytes(dest), t.bytes);
        off.* += t.numel();
        return dest;
    }
};

fn f32of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// y[j] = bias[j] + row j of `w` ([out, in] bf16) times `x`, f64 accumulation.
fn matvec(w: []const u16, in: usize, x: []const f32, bias: []const u16, y: []f32) void {
    for (y, bias, 0..) |*o, b, j| {
        var s: f64 = f32of(b);
        for (w[j * in ..][0..in], x) |wb, xv| s += @as(f64, f32of(wb)) * xv;
        o.* = @floatCast(s);
    }
}

/// nn.GELU: x·Φ(x); Φ's tail by Abramowitz–Stegun 26.2.17 (|ε| ≤ 7.5e-8), 1/√(2π) = 0.39894228040143268.
fn gelu(x: f32) f32 {
    const t = 1.0 / (1.0 + 0.2316419 * @abs(x));
    const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    const tail = poly * @exp(-x * x / 2) * 0.39894228040143268;
    return x * (if (x < 0) tail else 1.0 - tail);
}

/// The config's dim, width and layers; format must be bf16-safetensors-v1.
fn parseConfig(a: Allocator, text: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadJointHeadConfig;
    const o = parsed.value.object;
    const format = o.get("format") orelse return error.BadJointHeadConfig;
    if (format != .string or !std.mem.eql(u8, format.string, "bf16-safetensors-v1")) return error.BadJointHeadConfig;
    return .{ .dim = try field(o, "dim"), .width = try field(o, "width"), .layers = try field(o, "layers") };
}

fn field(o: std.json.ObjectMap, name: []const u8) !usize {
    const v = o.get(name) orelse return error.BadJointHeadConfig;
    if (v != .integer or v.integer <= 0) return error.BadJointHeadConfig;
    return @intCast(v.integer);
}

/// The directory's one safetensors file's name; zero or several are refused.
fn oneShard(gpa: Allocator, io: std.Io, dir_path: []const u8) ![]const u8 {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var found: ?[]const u8 = null;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        if (found != null) return error.JointHeadShardCount;
        found = try gpa.dupe(u8, e.name);
    }
    return found orelse error.JointHeadShardCount;
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// A head fixture in `tmp` (dim 4, width 4, 2 blocks; every weight exactly bf16): its directory's path.
fn fixture(gpa: Allocator, tmp: *const std.testing.TmpDir) ![]u8 {
    const header =
        \\{"proj.weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]},
        \\ "proj.bias":{"dtype":"BF16","shape":[4],"data_offsets":[32,40]},
        \\ "blocks.0.0.weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[40,72]},
        \\ "blocks.0.0.bias":{"dtype":"BF16","shape":[4],"data_offsets":[72,80]},
        \\ "blocks.1.0.weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[80,112]},
        \\ "blocks.1.0.bias":{"dtype":"BF16","shape":[4],"data_offsets":[112,120]},
        \\ "out.weight":{"dtype":"BF16","shape":[1,4],"data_offsets":[120,128]},
        \\ "out.bias":{"dtype":"BF16","shape":[1],"data_offsets":[128,130]}}
    ;
    const weights = [_][]const f32{
        &.{ 0.5, -0.25, 0.125, 0 }, &.{ 0, 0.5, -0.25, 0.125 }, &.{ 0.125, 0, 0.5, -0.25 }, &.{ -0.25, 0.125, 0, 0.5 }, // proj
        &.{ 0.125, -0.25, 0.375, -0.5 }, // proj bias
        &.{ 0.25, 0.125, 0, 0 }, &.{ 0, 0.25, 0.125, 0 }, &.{ 0, 0, 0.25, 0.125 }, &.{ 0.125, 0, 0, 0.25 }, // block 0
        &.{ 0.125, -0.125, 0.125, -0.125 }, // block 0 bias
        &.{ 0.5, 0.25, 0, 0 }, &.{ 0, 0.5, 0.25, 0 }, &.{ 0, 0, 0.5, 0.25 }, &.{ 0.25, 0, 0, 0.5 }, // block 1
        &.{ -0.25, 0.25, -0.25, 0.25 }, // block 1 bias
        &.{ 0.5, -0.5, 0.25, -0.25 }, // out
        &.{0.125}, // out bias
    };
    var data: [8 + header.len + 130]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    var at: usize = 8 + header.len;
    for (weights) |w| for (w) |x| {
        std.mem.writeInt(u16, data[at..][0..2], bf16(x), .little);
        at += 2;
    };
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "head.safetensors", .data = &data });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "joint_head_config.json", .data = "{\"dim\":4,\"width\":4,\"layers\":2,\"format\":\"bf16-safetensors-v1\"}" });
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", gpa);
    defer gpa.free(root);
    return std.fs.path.join(gpa, &.{root});
}

fn row(gpa: Allocator, xs: []const f32) ![]u16 {
    const out = try gpa.alloc(u16, xs.len);
    for (xs, out) |x, *b| b.* = bf16(x);
    return out;
}

test "a JointHead loads its config and tensors and scores a hidden row" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try fixture(a, &tmp);
    defer a.free(dir);
    const h = try Head.open(a, std.testing.io, dir);
    defer h.close();
    try std.testing.expectEqual(@as(usize, 4), h.cfg.dim);
    try std.testing.expectEqual(@as(usize, 4), h.cfg.width);
    try std.testing.expectEqual(@as(usize, 2), h.cfg.layers);
    const hidden = try row(a, &.{ 0.5, -0.25, 0.125, -0.5 });
    defer a.free(hidden);
    try std.testing.expectApproxEqAbs(1.035910918342501, try h.score(hidden), 1e-4);
    const second = try row(a, &.{ 0.25, 0.5, -0.125, 0.75 });
    defer a.free(second);
    try std.testing.expectApproxEqAbs(0.09842275576323394, try h.score(second), 1e-4);
    const third = try row(a, &.{ -1.0, 0.0, 1.0, 0.5 });
    defer a.free(third);
    try std.testing.expectApproxEqAbs(0.3339105568875796, try h.score(third), 1e-4);
    const file = try std.fs.path.join(a, &.{ dir, "head.safetensors" });
    defer a.free(file);
    const other = try Head.open(a, std.testing.io, file);
    defer other.close();
    try std.testing.expectApproxEqAbs(1.035910918342501, try other.score(hidden), 1e-4); // the file path itself also opens
}

test "a JointHead refuses a wrong-width row and a nonfinite one" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try fixture(a, &tmp);
    defer a.free(dir);
    const h = try Head.open(a, std.testing.io, dir);
    defer h.close();
    const hidden = try row(a, &.{ 0.5, -0.25, 0.125 });
    defer a.free(hidden);
    try std.testing.expectError(error.Failed, h.score(hidden));
    const bad = try row(a, &.{ 0.5, -0.25, 0.125, -0.5 });
    defer a.free(bad);
    bad[2] = 0x7FC0; // a NaN hidden word ends the pass
    try std.testing.expectError(error.Failed, h.score(bad));
}

test "a JointHead refuses a wrong-format config and a missing tensor" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try fixture(a, &tmp);
    defer a.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = "{\"dim\":4,\"width\":4,\"layers\":2,\"format\":\"fp32\"}" });
    try std.testing.expectError(error.BadJointHeadConfig, Head.open(a, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = "{\"dim\":8,\"width\":4,\"layers\":2,\"format\":\"bf16-safetensors-v1\"}" });
    try std.testing.expectError(error.UnexpectedTensor, Head.open(a, io, dir)); // proj.weight is [4, 4], not [4, 8]
}
