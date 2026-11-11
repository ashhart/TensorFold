//! --vision on a small checkpoint booted as the server boots one: the tower, the image biases, an image prompt.
const std = @import("std");
const pkg = @import("dsv41_metal");
const fv = @import("fixture_vis.zig");
const boot_test = @import("boot_test.zig");
const vt = @import("vision_test.zig");

const testing = std.testing;
const host = @import("dsv41");
const api = host.api;
const vis = pkg.vision;
const prep = pkg.vision_prep;
const vrows = pkg.vision_rows;
const say = vt.say;

/// A booted small checkpoint's folder with the tiny tower's tensors as a fourth shard named in the index.
fn writeWithVision(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, cfg: vis.Config, seed: u64) !void {
    try boot_test.write(a, io, dir, seed);
    const c = boot_test.config();
    const img = try fv.image(a, cfg, c.layers, c.experts, seed + 1);
    defer a.free(img);
    try dir.writeFile(io, .{ .sub_path = "model-vision.safetensors", .data = img });
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const hlen: usize = @intCast(std.mem.readInt(u64, img[0..8], .little));
    const head = try std.json.parseFromSliceLeaky(std.json.Value, al, img[8..][0..hlen], .{});
    var index: []const u8 = try dir.readFileAlloc(io, "model.safetensors.index.json", al, .limited(1 << 24));
    index = std.mem.trimEnd(u8, index, "}");
    var out: std.Io.Writer.Allocating = .init(al);
    try out.writer.writeAll(index);
    var it = head.object.iterator();
    while (it.next()) |e| if (!std.mem.eql(u8, e.key_ptr.*, "__metadata__")) try out.writer.print(",\"{s}\":\"model-vision.safetensors\"", .{e.key_ptr.*});
    try out.writer.writeAll("}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = out.written() });
    // the config names the tower and the image token
    var json: []const u8 = try dir.readFileAlloc(io, "config.json", al, .limited(1 << 24));
    json = std.mem.trimEnd(u8, json, " \n");
    json = json[0 .. json.len - 1];
    const extra = try std.fmt.allocPrint(al, ",\"image_token_id\":{d},\"vision_config\":{{\"num_hidden_layers\":{d},\"hidden_size\":{d},\"num_attention_heads\":{d},\"intermediate_size\":{d},\"patch_size\":{d},\"rope_theta\":10000,\"downsample_ratio\":{d},\"max_image_tokens\":{d},\"min_pixels\":{d}}}}}", .{ cfg.image_token_id, cfg.layers, cfg.dim, cfg.heads, cfg.inter, cfg.patch, cfg.ratio, cfg.max_tokens, cfg.min_pixels });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.mem.concat(al, u8, &.{ json, extra }) });
}

test "dsv41 metal vision: --vision at boot loads the tower and the image biases; an image prompt replies, then resumes from the cache" {
    const a = testing.allocator;
    const io = testing.io;
    var cfg = fv.config();
    cfg.image_token_id = 300;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeWithVision(a, io, tmp.dir, cfg, 85);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const m = try pkg.boot.Boot.open(a, io, dir, .{ .slots = 2, .context = 1024, .small = true, .cache_bytes = 1 << 26, .vision = true });
    defer {
        const md = m.model();
        md.close(md.ptr);
    }
    const layers = m.parts.env.model.layers;
    var biased: usize = 0;
    for (layers) |l| biased += @intFromBool(l.gate_bvl != null);
    try testing.expect(m.t.tower != null);
    try testing.expectEqual(layers.len, biased);
    // 40 text tokens, an image of 10 span tokens, 30 more
    const px = try fv.patches(a, cfg, 5, 7, 86);
    defer a.free(px);
    const lg = prep.llmGrid(5, 7, cfg.ratio);
    const image: vrows.Image = .{ .pixels = px, .gh = 5, .gw = 7, .at = 40, .tokens = prep.spanTokens(lg[0], lg[1]), .hash = 0x1234567 };
    var ids: [80]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = if (i >= 40 and i < 50) cfg.image_token_id else @intCast(70 + i % 50);
    try m.t.setImages(0, &.{image});
    const first = try m.t.serial(0, &ids, 6, null);
    defer a.free(first);
    const hits = m.t.stats.hits;
    try m.t.setImages(1, &.{image});
    const again = try m.t.serial(1, &ids, 6, null);
    defer a.free(again);
    say(m.t.stats.hits == hits + 1 and std.mem.eql(u32, first, again), "(boot) {d} layers with image biases, a 10-token image prompt replied {any}, again from the cache: {any}", .{ biased, first, again });
    try testing.expectEqual(hits + 1, m.t.stats.hits);
    try testing.expectEqualSlices(u32, first, again);
}

/// One reply of the served engine: its tokens and the prompt rows it took from kept state.
const Got = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    cached: u32 = 0,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.mutex.lockUncancelable(testing.io);
        defer g.mutex.unlock(testing.io);
        switch (e.*) {
            .tokens => |ts| g.out.appendSlice(testing.allocator, ts) catch {},
            .prefilled => |n| g.cached = n,
            .finished => g.done = true,
            else => {},
        }
    }

    fn wait(g: *Got) !void {
        for (0..600_000) |_| {
            g.mutex.lockUncancelable(testing.io);
            const done = g.done;
            g.mutex.unlock(testing.io);
            if (done) return;
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

var request_id: api.Id = 500;

/// One request to the engine, waited for: its reply (testing.allocator's) and the rows it resumed.
fn ask(e: api.Engine, ids: []const u32, images: []const api.Image, n: u32) !struct { []u32, u32 } {
    var g: Got = .{};
    defer g.out.deinit(testing.allocator);
    const r: api.Request = .{ .prompt = ids, .max_tokens = n, .drafts = false, .images = images };
    request_id += 1;
    try e.submit(request_id, &r, .{ .ctx = &g, .event = Got.event });
    try g.wait();
    return .{ try g.out.toOwnedSlice(testing.allocator), g.cached };
}

test "dsv41 metal vision: a PNG through the engine's image input and an image request == the target's reply, again from the cache" {
    const a = testing.allocator;
    const io = testing.io;
    var cfg = fv.config();
    cfg.image_token_id = 300;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeWithVision(a, io, tmp.dir, cfg, 87);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const o: pkg.boot.Options = .{ .slots = 2, .context = 1024, .small = true, .vision = true };
    var served = o;
    served.cache_bytes = 1 << 26;
    const m = try pkg.boot.Boot.open(a, io, dir, served);
    const s = try host.serve_lanes.Served.open(a, io, m.model(), dir, null);
    defer host.serve_lanes.Served.close(s);
    var plain = o;
    plain.cache_bytes = 0;
    const ref = try pkg.boot.Boot.open(a, io, dir, plain);
    defer ref.model().close(ref.model().ptr);
    const e = s.engine();
    const v = e.info().vision orelse return error.TestNoVision;
    try testing.expectEqual(cfg.image_token_id, v.image_token);
    // the engine's prepare is the package's: the bytes to patches, a grid, span tokens and an identity
    const png = try pkg.vision_prepare.solidPng(a, 60, 50, .{ 200, 100, 50 });
    defer a.free(png);
    const p = try v.prepare(v.ctx, a, png, 64);
    defer a.free(p.pixels);
    const direct = try pkg.vision_prepare.prepare(io, a, png, 64, cfg);
    defer a.free(direct.pixels);
    try testing.expect(p.tokens <= 64 and p.tokens == direct.tokens and p.hash == direct.hash);
    try testing.expectEqualSlices(f32, direct.pixels, p.pixels);
    // 20 text tokens, the image token once a span token, 25 more: the prompt as the server expands it
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    for (0..20) |i| try ids.append(a, @intCast(70 + i % 50));
    for (0..p.tokens) |_| try ids.append(a, cfg.image_token_id);
    for (0..25) |i| try ids.append(a, @intCast(90 + i % 50));
    const image: api.Image = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = 20, .tokens = p.tokens, .hash = p.hash };
    const first = try ask(e, ids.items, &.{image}, 6);
    defer a.free(first[0]);
    const again = try ask(e, ids.items, &.{image}, 6);
    defer a.free(again[0]);
    try ref.t.setImages(0, &.{image});
    const want = try ref.t.serial(0, ids.items, 6, null);
    defer a.free(want);
    const ok = std.mem.eql(u32, want, first[0]) and std.mem.eql(u32, want, again[0]) and first[1] == 0 and again[1] > 0;
    say(ok, "(serve) a {d}-token PNG image request replied {any} == the target's; resumed rows {d} then {d}", .{ p.tokens, first[0], first[1], again[1] });
    try testing.expectEqualSlices(u32, want, first[0]);
    try testing.expectEqualSlices(u32, want, again[0]);
    try testing.expectEqual(@as(u32, 0), first[1]);
    try testing.expect(again[1] > 0 and again[1] <= ids.items.len);
}
