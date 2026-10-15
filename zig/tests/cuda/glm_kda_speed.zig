//! glm-kda-speed: 64 heads x 128, a 2048-row prompt (row path vs chunked), a decode row, a 16-row window; median ms.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("glm_kda_rig.zig");
const Rig = rig.Rig;
const kda = rig.kda;
const dk = rig.dk;

const heads = 64;
const n = heads * dk;
const prompt = 2048;

const What = enum { path, replay, prep_out };

fn time(r: *Rig, m: *const rig.Mods, c: kda.Chain, rows: usize, o: rig.Out, path: rig.Path, what: What) !f64 {
    var start = try cuda.Event.init(r.gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(r.gpu.d, true);
    defer end.deinit();
    var ms: [9]f64 = undefined;
    for (0..ms.len + 1) |i| {
        try start.record(r.s);
        switch (what) {
            .path => try rig.run(m, r.s, c, rows, o, path),
            .replay => if (path == .chunked) try kda.chunkReplay(m.ck, r.s, c.state_in, o.k, o.v, o.g, o.b, rows, heads, o.state) else try kda.replay(&m.ks, r.s, c.state_in, o.k, o.v, o.g, o.b, rows, heads, o.state),
            .prep_out => {
                try kda.prep(&m.ks, r.s, c, rows, o.saves(), .{ .q = o.q, .y = o.y });
                try kda.out(&m.ks, r.s, c, rows, o.saves(), .{ .q = o.q, .y = o.y });
            },
        }
        try end.record(r.s);
        try r.s.synchronize();
        if (i > 0) ms[i - 1] = try start.elapsedMs(end);
    }
    std.mem.sort(f64, &ms, {}, std.sort.asc(f64));
    return ms[ms.len / 2];
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(64);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var m = try rig.Mods.load(gpu.d);
    defer m.unload();
    const p_stride = 3 * n + heads;
    const state = try r.a.alloc(f32, n * dk);
    for (state) |*v| v.* = (r.rng.float(f32) * 2 - 1) * 0.05;
    const cw = try r.a.alloc(f32, 3 * n * 4);
    for (cw) |*v| v.* = r.rng.float(f32) - 0.5;
    const small = try r.a.alloc(f32, n);
    for (small) |*v| v.* = r.rng.float(f32) - 0.5;
    const B = std.mem.sliceAsBytes;
    const c: kda.Chain = .{ .p = try r.dev(B(try r.bfs(prompt * p_stride, 1.0))), .p_stride = p_stride, .b_off = 3 * n, .a = try r.dev(B(try r.bfs(prompt * n, 2.0))), .a_stride = n, .g = try r.dev(B(try r.bfs(prompt * n, 2.0))), .g_stride = n, .conv_state = try r.dev(B(try r.bfs(9 * n, 1.0))), .conv_w = try r.dev(B(cw)), .state_in = try r.dev(B(state)), .a_log = try r.dev(B(small[0..heads])), .dt_bias = try r.dev(B(small)), .norm_w = try r.dev(B(try r.bfs(dk, 1.0))), .eps = 1e-6, .lower = -5, .heads = heads };
    const o = try rig.Out.alloc(&r, heads, prompt);
    const wide = try time(&r, &m, c, prompt, o, .wide, .path);
    const chunk = try time(&r, &m, c, prompt, o, .chunked, .path);
    const rw = try time(&r, &m, c, prompt, o, .wide, .replay);
    const rc = try time(&r, &m, c, prompt, o, .chunked, .replay);
    const po = try time(&r, &m, c, prompt, o, .wide, .prep_out);
    const row = try time(&r, &m, c, 1, o, .fused, .path);
    const win = try time(&r, &m, c, 16, o, .fused, .path);
    const win_wide = try time(&r, &m, c, 16, o, .wide, .path);
    const win_chunk = try time(&r, &m, c, 16, o, .chunked, .path);
    std.debug.print("RESULT glm-kda-speed {d} heads x 128, {d}-row prompt: wide (prep, serial step, out) {d:.3} ms, chunked (prep, WY chunks, out) {d:.3} ms ({d:.2}x); " ++
        "prep + out alone {d:.3} ms; state-only: serial replay {d:.3} ms, chunked replay {d:.3} ms\n", .{ heads, prompt, wide, chunk, wide / chunk, po, rw, rc });
    std.debug.print("RESULT glm-kda-speed decode row (fused chain) {d:.4} ms; 16-row window: fused chain {d:.4} ms, wide {d:.4} ms, chunked {d:.4} ms\n", .{ row, win, win_wide, win_chunk });
    check.pass("glm-kda-speed: timed", .{});
}
