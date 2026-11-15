//! Places in gpu_rule's order without sorting a row; rule.better is a strict total order, so they are the full sort's.
const std = @import("std");
const rule = @import("gpu_rule.zig");
const Cand = rule.Cand;

/// The `n` best of `logits` * `inv_t` (n at most rule.candidates), sorted: the full sort's first `n`.
pub fn top(logits: []const f32, inv_t: f32, n: usize, out: *[rule.candidates]Cand) []Cand {
    std.debug.assert(n <= rule.candidates and n <= logits.len);
    const heap = out[0..n];
    if (n == 0) return heap;
    for (heap, logits[0..n], 0..) |*c, l, i| c.* = .{ .v = l * inv_t, .i = @intCast(i) };
    var parent = n / 2;
    while (parent > 0) {
        parent -= 1;
        sink(heap, parent);
    }
    for (logits[n..], n..) |l, i| {
        const c: Cand = .{ .v = l * inv_t, .i = @intCast(i) };
        if (rule.better({}, c, heap[0])) {
            heap[0] = c;
            sink(heap, 0);
        }
    }
    std.mem.sort(Cand, heap, {}, rule.better);
    return heap;
}

/// Restores the heap below `start`: every parent is worse than its children, so the root is the worst one kept.
fn sink(heap: []Cand, start: usize) void {
    var at = start;
    while (2 * at + 1 < heap.len) {
        var child = 2 * at + 1;
        if (child + 1 < heap.len and rule.better({}, heap[child], heap[child + 1])) child += 1;
        if (!rule.better({}, heap[at], heap[child])) break;
        std.mem.swap(Cand, &heap[at], &heap[child]);
        at = child;
    }
}

/// The candidate at place `k` (from 0) of the full sort, found by selection; `items` is reordered.
pub fn nth(items: []Cand, k: usize) Cand {
    std.debug.assert(k < items.len);
    var lo: usize = 0;
    var hi: usize = items.len;
    var budget: usize = 2 * std.math.log2_int_ceil(usize, items.len) + 4; // past it, a sort bounds the worst case
    while (hi - lo > 16 and budget > 0) : (budget -= 1) {
        const mid = lo + (hi - lo) / 2;
        std.mem.swap(Cand, &items[median(items, lo, mid, hi - 1)], &items[hi - 1]);
        const pivot = items[hi - 1];
        var store = lo;
        for (lo..hi - 1) |i| if (rule.better({}, items[i], pivot)) {
            std.mem.swap(Cand, &items[i], &items[store]);
            store += 1;
        };
        std.mem.swap(Cand, &items[store], &items[hi - 1]);
        if (k == store) return items[k];
        if (k < store) hi = store else lo = store + 1;
    }
    std.mem.sort(Cand, items[lo..hi], {}, rule.better);
    return items[k];
}

/// The index of the middle candidate of three.
fn median(items: []const Cand, a: usize, b: usize, c: usize) usize {
    const ab = rule.better({}, items[a], items[b]);
    const bc = rule.better({}, items[b], items[c]);
    const ac = rule.better({}, items[a], items[c]);
    if (ab == bc) return b;
    return if (ab == ac) c else a;
}
