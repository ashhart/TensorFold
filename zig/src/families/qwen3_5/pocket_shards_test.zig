//! The pocket in a Flash Next-shaped folder: the first write widens every expert, a later one lands in place.
const std = @import("std");
const living = @import("living");

const core = struct {
    pub const Checkpoint = @import("../../core/checkpoint.zig").Checkpoint;
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
};
const Shards = living.shards.Io(core);
const st = core.safetensors;
const affine = living.affine;
const f = living.families.qwen3_5;
const q = f.quant;
const bits: usize = q.bits;

const d = 64; // hidden
const own = 128; // two group-64 chunks
const neurons = f.neurons;
const wide = own + neurons;
const layers = 3;
const mtp = "language_model.mtp.layers.0.mlp.shared_expert.";

/// One shared expert's tensors as a checkpoint holds them, random codes: [gate, up, down] x [weight, scales, biases].
const Expert = struct {
    w: [3][]u32,
    s: [3][]u16,
    b: [3][]u16,

    fn init(a: std.mem.Allocator, r: std.Random) !Expert {
        var e: Expert = undefined;
        for (0..3) |p| {
            const values: usize = own * d; // gate and up: [own, d]; down: [d, own]
            e.w[p] = try a.alloc(u32, values * bits / 32);
            e.s[p] = try a.alloc(u16, values / q.group);
            e.b[p] = try a.alloc(u16, values / q.group);
            for (e.w[p]) |*x| x.* = r.int(u32);
            for (e.s[p], e.b[p]) |*s, *b| {
                s.* = affine.bf16of(0.01 + r.float(f32) * 0.01);
                b.* = affine.bf16of(-0.08);
            }
        }
        return e;
    }

    fn tensors(e: *const Expert, a: std.mem.Allocator, base: []const u8, out: *std.ArrayList(core.shard_edit.New)) !void {
        const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
        for (projs, 0..) |proj, p| {
            const rows: usize = if (p < 2) own else d;
            const cols: usize = if (p < 2) d else own;
            try out.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.weight", .{ base, proj }), .dtype = .u32, .shape = try a.dupe(usize, &.{ rows, cols * bits / 32 }), .bytes = std.mem.sliceAsBytes(e.w[p]) });
            try out.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.scales", .{ base, proj }), .dtype = .bf16, .shape = try a.dupe(usize, &.{ rows, cols / q.group }), .bytes = std.mem.sliceAsBytes(e.s[p]) });
            try out.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.biases", .{ base, proj }), .dtype = .bf16, .shape = try a.dupe(usize, &.{ rows, cols / q.group }), .bytes = std.mem.sliceAsBytes(e.b[p]) });
        }
    }
};

/// Row `r` of a [rows, cols] affine tensor `t`, dequantized into `out` (cols values).
fn rowOf(file: *const st.File, name: []const u8, r: usize, cols: usize, out: []f32) !void {
    var buf: [192]u8 = undefined;
    const w = file.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name})).?;
    const s = file.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{name})).?;
    const b = file.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{name})).?;
    const nw = cols * bits / 32;
    const ng = cols / q.group;
    var wa: [wide * bits / 32]u32 = undefined;
    var sa: [wide / q.group]u16 = undefined;
    var ba: [wide / q.group]u16 = undefined;
    for (wa[0..nw], std.mem.bytesAsSlice(u32, w.bytes[r * nw * 4 ..][0 .. nw * 4])) |*x, y| x.* = y;
    for (sa[0..ng], std.mem.bytesAsSlice(u16, s.bytes[r * ng * 2 ..][0 .. ng * 2])) |*x, y| x.* = y;
    for (ba[0..ng], std.mem.bytesAsSlice(u16, b.bytes[r * ng * 2 ..][0 .. ng * 2])) |*x, y| x.* = y;
    affine.dequantize(q, wa[0..nw], sa[0..ng], ba[0..ng], out);
}

test "Trainer admission: dense group-64 save reopen preserves base bytes and exactly the judged gate up down rows" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    // Layers 0 and 1 in the first shard, layer 2 in the second, plus unrelated tensors.
    var experts: [layers]Expert = undefined;
    for (&experts) |*e| e.* = try .init(a, r);
    var gate_w: [d * bits / 32]u32 = undefined;
    for (&gate_w, 0..) |*x, i| x.* = 0x12345678 +% @as(u32, @intCast(i)) *% 0x9e3779b9;
    const one: [d / q.group]u16 = @splat(affine.bf16of(0.5));
    var lists: [2]std.ArrayList(core.shard_edit.New) = .{ .empty, .empty };
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"metadata\":{\"total_size\":1},\"weight_map\":{");
    for (0..layers) |l| {
        const shard: usize = if (l < 2) 0 else 1;
        const base = try f.expertBase(try a.alloc(u8, 128), l);
        try experts[l].tensors(a, base, &lists[shard]);
        const gate_name = try std.fmt.allocPrint(a, "{s}shared_expert_gate", .{base[0 .. base.len - "mlp.".len]});
        try lists[shard].append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{gate_name}), .dtype = .u32, .shape = &.{ 1, d * bits / 32 }, .bytes = std.mem.sliceAsBytes(&gate_w) });
        try lists[shard].append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{gate_name}), .dtype = .bf16, .shape = &.{ 1, d / q.group }, .bytes = std.mem.sliceAsBytes(&one) });
    }
    const names = [2][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    for (names, lists) |name, list| {
        try core.shard_edit.create(gpa, try std.fs.path.join(a, &.{ dir, name }), &.{.{ "format", "mlx" }}, list.items);
        for (list.items) |t| try index.print(a, "{s}\"{s}\":\"{s}\"", .{ if (index.items[index.items.len - 1] == '{') "" else ",", t.name, name });
    }
    try index.appendSlice(a, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"qwen3_5","quantization":{"group_size":64,"bits":4,"mode":"affine"},
        \\"text_config":{"hidden_size":64,"num_hidden_layers":3,"intermediate_size":128}}
    });
    const text = (try std.json.parseFromSliceLeaky(std.json.Value, a, try tmp.dir.readFileAlloc(io, "config.json", a, .limited(1 << 16)), .{})).object.get("text_config").?.object;
    _ = text;
    var all: living.Layers = .empty;
    for (0..layers) |l| all.set(l);
    const lay = living.Layout.fromExperts(d, own, layers, all);
    try std.testing.expectEqual(@as(usize, own), lay.width);
    var at: [layers]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, lay.list(lay.learn, &at));
    // a neuron at each learning layer: gate rows sharp u, up rows u / sharp, outputs
    const n = neurons * d;
    const gate = try a.alloc(f32, 2 * n);
    const up = try a.alloc(f32, 2 * n);
    const down = try a.alloc(f32, 2 * n);
    @memset(gate, 0);
    @memset(up, 0);
    @memset(down, 0);
    for (0..2) |j| {
        var u: [d]f32 = undefined;
        for (&u) |*x| x.* = r.floatNorm(f32) * 0.05;
        living.neuron.gateRow(f.neuron, &u, gate[j * n ..][0..d]);
        living.neuron.upRow(f.neuron, &u, up[j * n ..][0..d]);
        for (down[j * n ..][0..d]) |*v| v.* = r.floatNorm(f32) * 0.2;
    }
    const asked_up = try a.dupe(f32, up);
    for (0..2) |j| try living.pocket_round.layer(f, gpa, d, neurons, gate[j * n ..][0..n], up[j * n ..][0..n], down[j * n ..][0..n]);
    const judged_gate = try a.dupe(f32, gate);
    const judged_up = try a.dupe(f32, up);
    const judged_down = try a.dupe(f32, down);
    _ = try Shards.write(gpa, io, dir, f, lay, &.{ 0, 1 }, neurons, &.{ gate, up }, down);
    try std.testing.expectEqualSlices(f32, judged_gate, gate);
    try std.testing.expectEqualSlices(f32, judged_up, up);
    try std.testing.expectEqualSlices(f32, judged_down, down);
    // text_config holds the wider expert and the pocket's width; the top level gains neither
    const se = core.shard_edit;
    try std.testing.expectEqual(@as(?i64, wide), try se.configIntIn(gpa, io, dir, "text_config", "intermediate_size"));
    try std.testing.expectEqual(@as(?i64, neurons), try se.configIntIn(gpa, io, dir, "text_config", "tensorfold_learned_neurons"));

    try std.testing.expectEqual(@as(?i64, null), try se.configInt(gpa, io, dir, "tensorfold_learned_neurons"));
    try std.testing.expectEqual(@as(?i64, null), try se.configInt(gpa, io, dir, "intermediate_size"));
    // handed back as written: within half a step of what was asked
    for (up, asked_up) |got, want| try std.testing.expectApproxEqAbs(want, got, 0.01);
    var row: [wide]f32 = undefined;
    var sizes: [2]u64 = undefined;
    for (names, 0..) |name, i| {
        var file = try st.File.open(gpa, io, try std.fs.path.join(a, &.{ dir, name }));
        defer file.close(io);
        sizes[i] = (try tmp.dir.statFile(io, name, .{})).size;
        for (0..layers) |l| {
            if ((l < 2) != (i == 0)) continue;
            const base = try f.expertBase(try a.alloc(u8, 128), l);
            const e = experts[l];
            for ([_][]const u8{ "gate_proj", "up_proj" }, 0..) |proj, p| {
                const name_p = try std.fmt.allocPrint(a, "{s}{s}", .{ base, proj });
                const w = file.get(try std.fmt.allocPrint(a, "{s}.weight", .{name_p})).?;
                try std.testing.expect(w.is(.u32, &.{ wide, d * bits / 32 }));
                try std.testing.expect(file.get(try std.fmt.allocPrint(a, "{s}.scales", .{name_p})).?.is(.bf16, &.{ wide, d / q.group }));
                // own rows byte for byte
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(e.w[p]), w.bytes[0 .. own * d * bits / 8]);
                // learned rows as handed back; a layer that learns nothing (the last) all zero
                for (0..neurons) |k| {
                    try rowOf(&file, name_p, own + k, d, row[0..d]);
                    const want: []const f32 = if (l < 2) (if (p == 0) gate else up)[l * n + k * d ..][0..d] else &(@as([d]f32, @splat(0)));
                    try std.testing.expectEqualSlices(f32, want, row[0..d]);
                }
            }
            const name_d = try std.fmt.allocPrint(a, "{s}down_proj", .{base});
            const dw = file.get(try std.fmt.allocPrint(a, "{s}.weight", .{name_d})).?;
            try std.testing.expect(dw.is(.u32, &.{ d, wide * bits / 32 }));
            for (0..d) |o| {
                // own columns byte for byte, learned ones as the down array now holds them
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(e.w[2][o * own * bits / 32 ..][0 .. own * bits / 32]), dw.bytes[o * wide * bits / 8 ..][0 .. own * bits / 8]);
                try rowOf(&file, name_d, o, wide, &row);
                for (0..neurons) |k| try std.testing.expectEqual(if (l < 2) down[l * n + k * d + o] else 0, row[own + k]);
            }
            // the expert's gate beside it is untouched
            const g = file.get(try std.fmt.allocPrint(a, "{s}shared_expert_gate.weight", .{base[0 .. base.len - "mlp.".len]})).?;
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&gate_w), g.bytes);
        }
    }
    // The owned loader reads the saved gate/up rows and transposed down columns exactly as judged.
    {
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        var restored: [neurons * d]f32 = undefined;
        const reader = @import("cuda_wide.zig");
        for (0..2) |layer| {
            const base = try f.expertBase(try a.alloc(u8, 128), layer);
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, projection| {
                try reader.read(&ck, base, name, if (projection < 2) own else d, if (projection < 2) d else own, &restored);
                const expected = (if (projection == 0) judged_gate else if (projection == 1) judged_up else judged_down)[layer * n ..][0..n];
                try std.testing.expectEqualSlices(f32, expected, &restored);
            }
        }
    }
    // a second neuron at layer 1: written in place, nothing grows, the first neurons stay
    var u_next: [d]f32 = undefined;
    for (&u_next) |*x| x.* = r.floatNorm(f32) * 0.05;
    living.neuron.gateRow(f.neuron, &u_next, gate[n + d ..][0..d]);
    living.neuron.upRow(f.neuron, &u_next, up[n + d ..][0..d]);
    for (down[n + d ..][0..d]) |*v| v.* = r.floatNorm(f32) * 0.2;
    const first = try a.dupe(f32, gate[n..][0..d]);
    _ = try Shards.write(gpa, io, dir, f, lay, &.{ 0, 1 }, neurons, &.{ gate, up }, down);
    for (names, sizes) |name, size| try std.testing.expectEqual(size, (try tmp.dir.statFile(io, name, .{})).size);
    try std.testing.expectEqualSlices(f32, first, gate[n..][0..d]);
    var file = try st.File.open(gpa, io, try std.fs.path.join(a, &.{ dir, names[0] }));
    defer file.close(io);
    try rowOf(&file, "language_model.model.layers.1.mlp.up_proj", own + 1, d, row[0..d]);
    try std.testing.expectEqualSlices(f32, up[n + d ..][0..d], row[0..d]);
    // a pocket of another width is refused
    try std.testing.expectError(error.PocketOfAnotherSize, Shards.write(gpa, io, dir, f, lay, &.{ 0, 1 }, 2 * neurons, &.{ gate, up }, down));
    // An actual in-place shard write succeeds before the writer throws; a new load must remain closed.
    const Partial = living.shards.Io(struct {
        pub const safetensors = core.safetensors;
        pub const shard_edit = PartialEdit;
    });
    const before_partial = try tmp.dir.readFileAlloc(io, names[0], a, .limited(1 << 24));
    gate[0] += 1;
    try std.testing.expectError(error.InjectedPartialWrite, Partial.write(gpa, io, dir, f, lay, &.{ 0, 1 }, neurons, &.{ gate, up }, down));
    const after_partial = try tmp.dir.readFileAlloc(io, names[0], a, .limited(1 << 24));
    try std.testing.expect(!std.mem.eql(u8, before_partial, after_partial));
    _ = try tmp.dir.statFile(io, ".tensorfold-save-pending", .{});
    const CK = @import("../../core/checkpoint.zig").Checkpoint;
    try std.testing.expectError(error.CheckpointSaveUncertain, CK.openModel(gpa, io, dir));
    _ = try living.shards.Io(core).writePending(gpa, io, dir, f, lay, &.{ 0, 1 }, neurons, &.{ gate, up }, down);
    try std.testing.expectError(error.CheckpointSaveUncertain, CK.openModel(gpa, io, dir));
    try core.shard_edit.save_guard.finish(io, dir);
    var reopened = try CK.openModel(gpa, io, dir);
    reopened.close();
}

const PartialEdit = struct {
    const base = core.shard_edit;
    pub const Replacement = base.Replacement;
    pub const configIntIn = base.configIntIn;
    pub const setConfigIn = base.setConfigIn;
    pub const rewrite = base.rewrite;
    pub const save_guard = base.save_guard;
    pub fn overwrite(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, edits: []const Replacement) !void {
        try base.overwrite(gpa, io, dir, edits[0..3]);
        return error.InjectedPartialWrite;
    }
    pub fn grow(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, edits: []const Replacement) !void {
        try base.grow(gpa, io, dir, edits);
        return error.InjectedPartialWrite;
    }
};
