//! What a family's prompt-cache states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A planned family's own zero-anchored chunk rows, its starts when a request names none (0: none).
    grid: u32 = 0,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep only the history's mark, at a chunk end (`chunk`); a planned family's marks add no call.
    min_prompt: u32 = 4096,
    /// Rows an unplanned pass's chunk runs from its resume point or last cut (1: any row; 0: short prompts keep none).
    chunk: u32 = 0,
    /// Unplanned passes also cut at Request.chunks and decoded-span edges (false: only their own grid and marks).
    chunk_starts: bool = true,
    /// Marks always end chunks; false when a backend may capture them inside a chunk without splitting it.
    chunk_marks: bool = true,
    /// Background reply passes kept at their end: .on drops a history mark near the resume point, .waits keeps it.
    warm: enum { off, on, waits } = .off,
    /// A warm pass keeps its end only where `usable` resumes it; its marks still keep (false: any end, as before).
    warm_usable: bool = false,
    /// A paged store (prompt_radix.zig): the tokens a page holds, where its marks and resumes sit (0: not paged).
    page: u32 = 0,
    /// Keep the prompt's last page for a later turn whose history it did not name (unpaged: last chunk, none named).
    tail: bool = false,
    /// A tail state's position: the pass's last chunk end rounded down to this many rows (1: the exact end).
    /// A resumed pass's chunk starts here, so its rows must sit in the same tile lanes a fresh pass gives them.
    tail_align: u32 = 1,
    /// ROUND-502: also keep a state at the pass's own end, so a later turn whose new rows are a tensor-tile
    /// chunk prefills exactly those rows instead of the tile tail's 128 behind it. Only kept when this pass's
    /// last chunk kept the class a fresh pass gives the same rows; `find` re-checks the next turn's class.
    end_mark: bool = false,
    /// ROUND-504: a pass may SNAPSHOT the tail mark inside the chunk that runs past it -- the backend keeps the
    /// window ring, the taps and the Engram history at the mark -- instead of cutting a second forward there.
    /// Only when the tail piece is a tensor tile (>= 64 rows): the merged chunk is then the same kernel class.
    tail_snap: bool = true,
    /// Repeated raw prompts keep near their end; only families that allow any-row cuts opt in (0: off).
    repeat_min: u32 = 0,
    /// Image and keyed prompts also count as repeats, under the key lookups and saves use (false: raw prompts only).
    repeat_keyed: bool = false,
    /// Keep one before where the prompt parts from the one it resumed (a miss: the longest kept one), however near.
    fork: bool = false,
};
