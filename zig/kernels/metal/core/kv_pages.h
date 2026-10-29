// A position's page from its stream's table and its row there; TF_PAGE_TOKENS must equal metal_kv_pages.page_tokens.
#define TF_PAGE_TOKENS 256
inline ulong tf_page_offset(device const uint* table, uint pos, ulong page_bytes, ulong row_bytes) {
  return ulong(table[pos / TF_PAGE_TOKENS]) * page_bytes + ulong(pos % TF_PAGE_TOKENS) * row_bytes;
}
