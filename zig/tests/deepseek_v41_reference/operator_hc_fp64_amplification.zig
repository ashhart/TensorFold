//! Hyperconnection amplification comparisons against the FP64 ideal.
const support = @import("hc_fp64_fixture.zig");
const std = support.std;
const mtl = support.mtl;
const math = support.math;
const mhc = support.mhc;
const ops = support.ops;
const Hc = support.Hc;
const Config = support.Config;
const width = support.width;
const copies = support.copies;
const stream_width = support.stream_width;
const mix_count = support.mix_count;
const released_config = support.released_config;
const fixture_path = support.fixture_path;
const Fixture = support.Fixture;
const decodeU16 = support.decodeU16;
const decodeF32 = support.decodeF32;
const upload = support.upload;
const view = support.view;
const runStage = support.runStage;
const syntheticWeight = support.syntheticWeight;
const fourCopies = support.fourCopies;
const stageF64 = support.stageF64;
const perturbOneUlp = support.perturbOneUlp;

test "mHC 1-ULP input amplification through the hyperconnection mix" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var o = try ops.Pipelines.load(device);
    defer o.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_stream = try decodeU16(gpa, fixture.value.ffn_stream_bf16);
    const ffn_branch = try decodeU16(gpa, fixture.value.ffn_branch_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const weight = try syntheticWeight(gpa);

    const scale = try device.buffer(12, mtl.ResourceOptions.shared);
    defer scale.deinit();
    upload(f32, scale, ffn_scale);
    const base = try device.buffer(mix_count * 4, mtl.ResourceOptions.shared);
    defer base.deinit();
    upload(f32, base, ffn_base);
    const weight_buffer = try device.buffer(mix_count * stream_width * 4, mtl.ResourceOptions.shared);
    defer weight_buffer.deinit();
    upload(f32, weight_buffer, weight);
    const hc = Hc{
        .projection = .{ .buffer = weight_buffer, .offset = 0, .bytes = mix_count * stream_width * 4, .dtype = .f32 },
        .scale = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 },
        .base = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 },
    };

    const columns = 64;
    const column_stride = width / columns;
    const Row = struct { name: []const u8, stream: []u16 };
    var rows: [9]Row = undefined;
    for (0..8) |r| {
        rows[r] = .{ .name = "teacher", .stream = try fourCopies(gpa, eight_rows[r * width ..][0..width]) };
    }
    rows[8] = .{ .name = "ffn_token1", .stream = try gpa.dupe(u16, ffn_stream) };

    const y0 = try gpa.alloc(f64, stream_width);
    const y1 = try gpa.alloc(f64, stream_width);
    var y0_bf: [stream_width]u16 = undefined;
    var y1_bf: [stream_width]u16 = undefined;
    const coll0 = try gpa.alloc(f64, width);
    const coll1 = try gpa.alloc(f64, width);
    var inv0: f64 = undefined;
    var inv1: f64 = undefined;
    var comb0: [16]f64 = undefined;
    var comb1: [16]f64 = undefined;

    for (rows, 0..) |row, row_index| {
        stageF64(row.stream, hc, cfg, ffn_branch, y0, &y0_bf, coll0, &inv0, &comb0);
        var sum_y2: f64 = 0;
        for (y0) |v| sum_y2 += v * v;
        var sum_x2: f64 = 0;
        for (row.stream) |v| sum_x2 += blk: {
            const f: f64 = @floatCast(math.f(v));
            break :blk f * f;
        };
        var sum_c2: f64 = 0;
        for (coll0) |v| sum_c2 += v * v;
        var sum_b2: f64 = 0;
        for (y0_bf) |v| {
            const f: f64 = @floatCast(math.f(v));
            sum_b2 += f * f;
        }

        var amp_sum: f64 = 0;
        var amp_max: f64 = 0;
        var amp_n: usize = 0;
        var amp_c_sum: f64 = 0;
        var amp_c_max: f64 = 0;
        var amp_b_sum: f64 = 0;
        var amp_b_max: f64 = 0;
        var changed_any: usize = 0;
        var changed_cells: usize = 0;
        var inv_rel_sum: f64 = 0;
        var comb_sq: f64 = 0;
        var comb_n: usize = 0;

        for (0..copies) |j| {
            for (0..columns) |k| {
                const col = k * column_stride;
                const idx = j * width + col;
                const orig = row.stream[idx];
                if (math.f(orig) == 0) continue;
                const perturbed = perturbOneUlp(orig);
                const dx = @as(f64, @floatCast(math.f(perturbed))) - @as(f64, @floatCast(math.f(orig)));
                row.stream[idx] = perturbed;
                stageF64(row.stream, hc, cfg, ffn_branch, y1, &y1_bf, coll1, &inv1, &comb1);
                row.stream[idx] = orig;

                var dy2: f64 = 0;
                for (y0, y1) |a, b| {
                    const d = a - b;
                    dy2 += d * d;
                }
                const amp = @sqrt(dy2 / sum_y2) * @sqrt(sum_x2) / @abs(dx);
                amp_sum += amp;
                if (amp > amp_max) amp_max = amp;
                amp_n += 1;

                var dc2: f64 = 0;
                for (coll0, coll1) |a, b| {
                    const d = a - b;
                    dc2 += d * d;
                }
                const amp_c = @sqrt(dc2 / sum_c2) * @sqrt(sum_x2) / @abs(dx);
                amp_c_sum += amp_c;
                if (amp_c > amp_c_max) amp_c_max = amp_c;

                var changed: usize = 0;
                var dyb2: f64 = 0;
                for (y0_bf, y1_bf) |a, b| {
                    if (a != b) changed += 1;
                    const av: f64 = @floatCast(math.f(a));
                    const bv: f64 = @floatCast(math.f(b));
                    const d = av - bv;
                    dyb2 += d * d;
                }
                const amp_b = @sqrt(dyb2 / sum_b2) * @sqrt(sum_x2) / @abs(dx);
                amp_b_sum += amp_b;
                if (amp_b > amp_b_max) amp_b_max = amp_b;
                if (changed > 0) {
                    changed_any += 1;
                    changed_cells += changed;
                }
                inv_rel_sum += @abs(inv1 - inv0) / inv0;
                var comb_d2: f64 = 0;
                var comb_m2: f64 = 0;
                for (comb0, comb1) |a, b| {
                    const d = a - b;
                    comb_d2 += d * d;
                    comb_m2 += b * b;
                }
                comb_sq += @sqrt(comb_d2 / comb_m2);
                comb_n += 1;
            }
        }
        std.debug.print("HC64 {{\"amplification\":{{\"row\":{d},\"row_name\":\"{s}\",\"perturbations\":{d},\"amp_mean\":{d},\"amp_max\":{d},\"amp_collapse_mean\":{d},\"amp_collapse_max\":{d},\"amp_bf16_mean\":{d},\"amp_bf16_max\":{d},\"bf16_changed_fraction\":{d},\"mean_changed_cells\":{d},\"inv_rel_mean\":{d},\"comb_nrms_mean\":{d}}}}}\n", .{ row_index, row.name, amp_n, amp_sum / @as(f64, @floatFromInt(amp_n)), amp_max, amp_c_sum / @as(f64, @floatFromInt(amp_n)), amp_c_max, amp_b_sum / @as(f64, @floatFromInt(amp_n)), amp_b_max, changed_any, @as(f64, @floatFromInt(changed_cells)) / @as(f64, @floatFromInt(amp_n)), inv_rel_sum / @as(f64, @floatFromInt(amp_n)), comb_sq / @as(f64, @floatFromInt(comb_n)) });
        // Bound analog gain below one, BF16-visible growth near one ULP and collapse gain near one half.
        try std.testing.expect(amp_sum / @as(f64, @floatFromInt(amp_n)) < 1.0);
        try std.testing.expect(amp_max < 1.0);
        try std.testing.expect(amp_c_max < 2.5);
        try std.testing.expect(amp_b_sum / @as(f64, @floatFromInt(amp_n)) < 3.0);
        try std.testing.expect(inv_rel_sum / @as(f64, @floatFromInt(amp_n)) < 1e-5);
    }

    // Batch the first teacher row and 16 stream-cell perturbations through Metal, then compare FP64 gain.
    {
        const metal_perturbations = 16;
        const batch = metal_perturbations + 1;
        var batch_stream = try gpa.alloc(u16, batch * stream_width);
        var cells: [metal_perturbations]usize = undefined;
        var dxs: [metal_perturbations]f64 = undefined;
        var filled: usize = 0;
        @memcpy(batch_stream[0..stream_width], rows[0].stream);
        outer: for (0..copies) |j| {
            for (0..columns) |k| {
                const idx = j * width + k * column_stride;
                const orig = rows[0].stream[idx];
                if (math.f(orig) == 0) continue;
                cells[filled] = idx;
                dxs[filled] = @as(f64, @floatCast(math.f(perturbOneUlp(orig)))) - @as(f64, @floatCast(math.f(orig)));
                filled += 1;
                if (filled == metal_perturbations) break :outer;
            }
        }
        for (0..filled) |i| {
            @memcpy(batch_stream[(i + 1) * stream_width ..][0..stream_width], rows[0].stream);
            batch_stream[(i + 1) * stream_width + cells[i]] = perturbOneUlp(rows[0].stream[cells[i]]);
        }
        const stream_buf = try device.buffer(batch * stream_width * 2, mtl.ResourceOptions.shared);
        defer stream_buf.deinit();
        upload(u16, stream_buf, batch_stream);
        const inv_buf = try device.buffer(batch * 4, mtl.ResourceOptions.shared);
        defer inv_buf.deinit();
        const mixes_buf = try device.buffer(batch * mix_count * 4, mtl.ResourceOptions.shared);
        defer mixes_buf.deinit();
        const pre_buf = try device.buffer(batch * 4 * 4, mtl.ResourceOptions.shared);
        defer pre_buf.deinit();
        const post_buf = try device.buffer(batch * 4 * 4, mtl.ResourceOptions.shared);
        defer post_buf.deinit();
        const comb_buf = try device.buffer(batch * 16 * 4, mtl.ResourceOptions.shared);
        defer comb_buf.deinit();
        const collapsed_buf = try device.buffer(batch * width * 2, mtl.ResourceOptions.shared);
        defer collapsed_buf.deinit();
        const branch_buf = try device.buffer(batch * width * 2, mtl.ResourceOptions.shared);
        defer branch_buf.deinit();
        for (0..batch) |r| @memcpy(branch_buf.slice(u16, batch * width)[r * width ..][0..width], ffn_branch);
        const y_buf = try device.buffer(batch * stream_width * 2, mtl.ResourceOptions.shared);
        defer y_buf.deinit();
        try runStage(queue, &p, &o, @intCast(batch), cfg, view(stream_buf), view(weight_buffer), view(inv_buf), view(mixes_buf), view(scale), view(base), view(pre_buf), view(post_buf), view(comb_buf), view(collapsed_buf), view(branch_buf), view(y_buf));

        const y_metal = y_buf.slice(u16, batch * stream_width);
        var nan_count: usize = 0;
        var first_nan: usize = 0;
        for (y_metal[0..stream_width], 0..) |v, i| {
            const f = math.f(v);
            if (std.math.isNan(f)) {
                nan_count += 1;
                if (nan_count == 1) first_nan = i;
            }
        }
        var sum_x2_row0: f64 = 0;
        for (rows[0].stream) |v| {
            const f: f64 = @floatCast(math.f(v));
            sum_x2_row0 += f * f;
        }
        var sum_base2: f64 = 0;
        for (0..stream_width) |i| {
            const f: f64 = @floatCast(math.f(y_metal[i]));
            sum_base2 += f * f;
        }
        var amp_sum: f64 = 0;
        var changed_total: usize = 0;
        for (0..filled) |i| {
            var dy2: f64 = 0;
            var changed: usize = 0;
            for (0..stream_width) |c| {
                const a: f64 = @floatCast(math.f(y_metal[(i + 1) * stream_width + c]));
                const b: f64 = @floatCast(math.f(y_metal[c]));
                const d = a - b;
                if (d != 0) changed += 1;
                dy2 += d * d;
            }
            changed_total += changed;
            amp_sum += @sqrt(dy2 / sum_base2) * @sqrt(sum_x2_row0) / @abs(dxs[i]);
        }
        std.debug.print("HC64 {{\"amplification\":{{\"row\":0,\"metal\":true,\"perturbations\":{d},\"amp_mean\":{d},\"mean_changed_cells\":{d}}}}}\n", .{ filled, amp_sum / @as(f64, @floatFromInt(filled)), @as(f64, @floatFromInt(changed_total)) / @as(f64, @floatFromInt(filled)) });
        const metal_amp = amp_sum / @as(f64, @floatFromInt(filled));
        try std.testing.expect(metal_amp > 0.5 and metal_amp < 3.0);
    }
}
