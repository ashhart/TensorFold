// A paged cache's address: a stream's table row of page ids, elements a page holds across its caches, positions a page.
#pragma once

namespace tf_common {

struct Pg {
    const int* tab;
    long long page;
    int tok;
    int stride; // table entries a slot (decode); a prompt kernel is given its stream's row
};

} // namespace tf_common
