// window_rows.cuh's planted faults and a decode row's live chunks; included inside namespace tf_wrows, after CK.
#pragma once

// Planted faults, compiled only into contract_controls.cu's controls; every served kernel has NONE.
enum Fault : int {
    NONE = 0,
    EDGE = 1,    // the window's first key one too early
    LAST = 2,    // a prompt row's last key: its own position, the given one ignored
    RESCALE = 3, // the merge adds the chunks' partials without exp(m_c - M)
    TRUNC = 4,   // the merge's output rounded toward zero
    ALL = 5,     // every chunk, empty or not, as before a row's chunks were bounded: the equality tests' baseline
};

// Chunks a row at position p has partials in: with no window, those up to p's; a window's keep their own count.
__device__ __forceinline__ int liveChunks(int chunks, int win, int p, bool all) {
    return win > 0 || all ? chunks : min(chunks, p / CK + 1);
}
