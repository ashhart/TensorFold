#include "../src/video/mem_seek.h"
#include <assert.h>
#include <stdint.h>

int main(void) {
    const size_t len = 100;
    assert(tf_mem_seek_at(len, 10, 40) == 50);
    assert(tf_mem_seek_at(len, -40, 40) == 0);
    assert(tf_mem_seek_at(len, 60, 40) == 100);
    assert(tf_mem_seek_at(len, INT64_MAX, 40) == -1);
    assert(tf_mem_seek_at(len, INT64_MIN, 40) == -1);
    assert(tf_mem_seek_at(len, INT64_MAX, 100) == -1);
    assert(tf_mem_seek_at(len, INT64_MIN, 100) == -1);
    assert(tf_mem_seek_at(len, INT64_MAX, 0) == -1);
    assert(tf_mem_seek_at(len, INT64_MIN, 0) == -1);
    assert(tf_mem_seek_at(len, 101, 0) == -1);
    assert(tf_mem_seek_at(len, -1, 0) == -1);
    assert(tf_mem_seek_at(len, 0, 100) == 100);
    return 0;
}
