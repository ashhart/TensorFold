//! Model-free CUDA-shard to Metal-pocket check against an independent float64 formula.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const living = @import("living");
const pk = @import("pocket");
const cfg = pk.config;
const d = 64;
const own = 128;
const cap = 64;
const n = d * cap;
const bf = core.affine4_host;
const Shards = living.shards.Io(core);

const Fixture = struct {
    gate: []f32,
    up: []f32,
    down: []f32,
    config: cfg.Config,
};

fn fixture(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Fixture {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp = arena.allocator();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(tmp, "{\"weight_map\":{");
    for (0..2) |layer| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, p| {
        const rows: usize = if (p == 2) d else own;
        const cols: usize = if (p == 2) own else d;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, kind| {
            const width = if (kind == 0) cols / 8 else cols / 64;
            const bytes = try tmp.alloc(u8, rows * width * (if (kind == 0) @as(usize, 4) else 2));
            @memset(bytes, 0);
            if (kind == 1) for (0..rows * width) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bf.bf16of(1), .little);
            const name = try std.fmt.allocPrint(tmp, "language_model.model.layers.{d}.mlp.{s}.{s}", .{ layer, proj, leaf });
            try tensors.append(tmp, .{ .name = name, .dtype = if (kind == 0) .u32 else .bf16, .shape = try tmp.dupe(usize, &.{ rows, width }), .bytes = bytes });
            try index.print(tmp, "{s}\"{s}\":\"model.safetensors\"", .{ if (tensors.items.len == 1) "" else ",", name });
        }
    };
    try index.appendSlice(tmp, "}}");
    try core.shard_edit.create(a, try std.fs.path.join(tmp, &.{ dir, "model.safetensors" }), &.{.{ "format", "mlx" }}, tensors.items);
    var folder = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer folder.close(io);
    try folder.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try folder.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"qwen3_5","quantization":{"bits":4,"group_size":64},"text_config":{
        \\"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
        \\"num_key_value_heads":2,"head_dim":16,"vocab_size":97,"linear_num_key_heads":2,
        \\"linear_num_value_heads":4,"linear_key_head_dim":16,"linear_value_head_dim":16,
        \\"linear_conv_kernel_dim":4,"eos_token_id":2}}
    });
    const gate = try a.alloc(f32, n);
    errdefer a.free(gate);
    const up = try a.alloc(f32, n);
    errdefer a.free(up);
    const down = try a.alloc(f32, n);
    errdefer a.free(down);
    @memset(gate, 0);
    @memset(up, 0);
    @memset(down, 0);
    gate[0] = 16;
    up[0] = 0.0625;
    down[0] = 0.125;
    down[1] = -0.125;
    var all: living.Layers = .empty;
    all.set(0);
    all.set(1);
    const layout = living.Layout.fromExperts(d, own, 2, all);
    _ = try Shards.write(a, io, dir, living.families.qwen3_5, layout, &.{0}, cap, &.{ gate, up }, down);
    const text = try folder.readFileAlloc(io, "config.json", tmp, .limited(1 << 20));
    return .{ .gate = gate, .up = up, .down = down, .config = try cfg.parse(a, text, null) };
}

fn reference(f: Fixture, x: []const u16, out: u16, col: usize) u16 {
    var g: f64 = 0;
    var u: f64 = 0;
    for (0..d) |k| {
        g += @as(f64, bf.f32of(x[k])) * f.gate[k];
        u += @as(f64, bf.f32of(x[k])) * f.up[k];
    }
    const activation = (g / (1 + @exp(-g))) * u;
    return bf.bf16of(@floatCast(@as(f64, bf.f32of(out)) + activation * f.down[col]));
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) return error.NeedFixtureDirectory;
    const f = try fixture(a, io, args[1]);
    if (f.config.intermediate != own or f.config.pocket != cap) return error.PocketWidthNotSplit;
    if (args.len == 3 and std.mem.eql(u8, args[2], "--fixture-only")) {
        std.debug.print("Q27_POCKET_FIXTURE saved_by_common_CUDA_writer=true original_width={d} pocket={d}\n", .{ f.config.intermediate, f.config.pocket });
        return;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pockets = try pk.Pockets.init(a, device, d, 2);
    defer pockets.deinit();
    try pockets.load(io, args[1], f.config);
    const read = try a.alloc(f32, 3 * n);
    const used = try pockets.read(0, read[0..n], read[n..][0..n], read[2 * n ..]);
    if (used != 1 or !std.mem.eql(f32, f.gate, read[0..n]) or !std.mem.eql(f32, f.up, read[n..][0..n]) or !std.mem.eql(f32, f.down, read[2 * n ..])) return error.CudaPocketReadDiffers;
    if (pockets.layers[1] != null) return error.EmptyLayerAllocated;
    var compared: usize = 0;
    var untouched: usize = 0;
    for ([_]usize{ 1, 3, 128, 2048 }) |rows| {
        const count = rows * d;
        const input = try device.buffer((count + 16) * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer input.deinit();
        const output = try device.buffer((count + 16) * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer output.deinit();
        const x = input.slice(u16, count + 16);
        @memset(x, 0);
        for (0..rows) |r| x[r * d] = bf.bf16of(([_]f32{ 0, 1, -1, 2 })[r % 4]);
        @memset(x[count..], 0x5a3c);
        const before = try a.dupe(u16, x);
        const initial = try a.alloc(u16, count + 16);
        @memset(initial, bf.bf16of(0.25));
        @memset(initial[count..], 0x5a3c);
        const y = output.slice(u16, count + 16);
        for ([_]bool{ false, true }) |untaught| {
            pockets.selected = .{ .all = !untaught };
            var first: ?[]u16 = null;
            for (0..5) |_| {
                @memcpy(y, initial);
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                try pockets.add(enc, 0, .{ .buffer = input }, .{ .buffer = output }, rows);
                try pockets.add(enc, 1, .{ .buffer = input }, .{ .buffer = output }, rows);
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure() != null) return error.PocketGpuFailure;
                for (0..rows) |r| for (0..d) |col| {
                    const expected = if (untaught) initial[r * d + col] else reference(f, x[r * d ..][0..d], initial[r * d + col], col);
                    if (y[r * d + col] != expected) {
                        std.debug.print("POCKET_DIFF rows={d} row={d} col={d} got={x} reference={x}\n", .{ rows, r, col, y[r * d + col], expected });
                        return error.PocketFloat64Mismatch;
                    }
                    if (untaught) untouched += 1 else compared += 1;
                };
                if (!std.mem.eql(u16, before, x) or !std.mem.eql(u16, initial[count..], y[count..])) return error.PocketInputOrGuardChanged;
                if (first) |prior| {
                    if (!std.mem.eql(u16, prior, y)) return error.PocketRepeatChanged;
                } else first = try a.dupe(u16, y);
            }
        }
    }
    std.debug.print("Q27_POCKET compared={d} untaught_equal={d} repeats=5 rows=1,3,128,2048 CUDA_shards_equal=true empty_layer_no_buffer=true\n", .{ compared, untouched });
}
