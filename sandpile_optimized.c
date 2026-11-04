/* CS598APE HW3: optimized CPU implementation.
 * Algorithm: batched legal topplings, precomputed active row intervals,
 * contiguous grid storage and alternating Gauss-Seidel sweeps.
 * The exact Pacman toppling region from the reference is retained.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int N;
static int *grid;
static int *row_begin, *row_end;

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *alloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) die("malloc failed");
    return p;
}

/* A row's topplable cells form one interval.  Cells outside the interval
 * may receive grains, but are NEVER toppled (exactly as in the handout).
 */
static void initialize_region(void) {
    const int64_t r = N / 2;
    const int64_t r2 = r * r;
    for (int i = 0; i < N; ++i) {
        int max_j = (i > N - 1 - i) ? i : N - 1 - i;
        int64_t di = (int64_t)i - N / 2;
        row_begin[i] = N;
        row_end[i] = -1;
        for (int j = 0; j <= max_j; ++j) {
            int64_t dj = (int64_t)j - N / 2;
            if (di * di + dj * dj <= r2) {
                if (row_begin[i] == N) row_begin[i] = j;
                row_end[i] = j;
            }
        }
    }
}

static void stabilize(void) {
    int changed, reverse = 0;
    do {
        changed = 0;
        if (!reverse) {
            for (int i = 0; i < N; ++i) {
                int *row = grid + (size_t)i * N;
                for (int j = row_begin[i]; j <= row_end[i]; ++j) {
                    int v = row[j];
                    if (v < 4) continue;
                    int q = v >> 2;
                    row[j] = v & 3;
                    if (i) row[j - N] += q;
                    if (j) row[j - 1] += q;
                    if (i + 1 < N) row[j + N] += q;
                    if (j + 1 < N) row[j + 1] += q;
                    changed = 1;
                }
            }
        } else {
            for (int i = N - 1; i >= 0; --i) {
                int *row = grid + (size_t)i * N;
                for (int j = row_end[i]; j >= row_begin[i]; --j) {
                    int v = row[j];
                    if (v < 4) continue;
                    int q = v >> 2;
                    row[j] = v & 3;
                    if (i) row[j - N] += q;
                    if (j) row[j - 1] += q;
                    if (i + 1 < N) row[j + N] += q;
                    if (j + 1 < N) row[j + 1] += q;
                    changed = 1;
                }
            }
        }
        reverse = !reverse;
    } while (changed);
}

static void find_identity(void) {
    const size_t count = (size_t)N * N;
    for (size_t id = 0; id < count; ++id) grid[id] = 6;
    stabilize();
    for (size_t id = 0; id < count; ++id) grid[id] = 6 - grid[id];
    stabilize();
}

static void render(const char *filename) {
    const size_t count = (size_t)N * N;
    if (count > SIZE_MAX / 3) die("image buffer too large");
    unsigned char *pixels = alloc(count * 3);
    for (size_t id = 0; id < count; ++id) {
        unsigned char r = 0, g = 0, b = 0;
        switch (grid[id]) {
        case 0: r = 15;  g = 15;  b = 25;  break;
        case 1: r = 0;   g = 210; b = 255; break;
        case 2: r = 230; g = 30;  b = 130; break;
        case 3: r = 255; g = 200; b = 0;   break;
        default: break;
        }
        pixels[3 * id] = r;
        pixels[3 * id + 1] = g;
        pixels[3 * id + 2] = b;
    }
    if (!stbi_write_png(filename, N, N, 3, pixels, 3 * N))
        die("failed to write PNG");
    printf("Successfully saved sandpile identity to: %s\n", filename);
    free(pixels);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t n = strtoumax(argv[1], &end, 10);
    if (errno || argv[1][0] < '0' || argv[1][0] > '9' || *end ||
        n == 0 || n > INT32_MAX)
        die("grid size must be a positive int");
    N = (int)n;
    if ((size_t)N > SIZE_MAX / (size_t)N ||
        (size_t)N * N > SIZE_MAX / sizeof(int))
        die("grid size overflow");

    grid = alloc((size_t)N * N * sizeof(int));
    row_begin = alloc((size_t)N * sizeof(int));
    row_end = alloc((size_t)N * sizeof(int));
    initialize_region();
    find_identity();
    render("identity.png");
    free(row_end);
    free(row_begin);
    free(grid);
    return EXIT_SUCCESS;
}
