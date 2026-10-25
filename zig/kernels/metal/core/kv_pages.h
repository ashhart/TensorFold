// The core KV pool's pages (core/metal_kv_pages.zig): a position's page from its stream's table, its row in that page.
// TF_PAGE_TOKENS is metal_kv_pages.page_tokens, which does not compile when they differ.
#define TF_PAGE_TOKENS 256
inline ulong tf_page_offset(device const uint* table, uint pos, ulong page_bytes, ulong row_bytes) {
  return ulong(table[pos / TF_PAGE_TOKENS]) * page_bytes + ulong(pos % TF_PAGE_TOKENS) * row_bytes;
}
