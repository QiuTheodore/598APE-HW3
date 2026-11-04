/* CS598APE HW3: OpenMP CPU implementation.
 * Bulk-synchronous legal topplings with race-free gather updates.
 * Use: gcc -O3 -g -Werror -fopenmp sandpile_openmp.c -lm -o sandpile_omp.exe
 * Set OMP_NUM_THREADS=4 to test on a 4-vCPU machine.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int N, stride;
static int *grid;
static int *topplings;
static int *row_begin, *row_end;

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) die("Out of memory");
    return p;
}

/* Keep exactly the reference's Pacman toppling region. */
static void initialize_region(void) {
    int64_t radius = N / 2;
    int64_t radius_squared = radius * radius;
    for (int i = 0; i < N; ++i) {
        int last = i > N - 1 - i ? i : N - 1 - i;
        int64_t di = (int64_t)i - N / 2;
        row_begin[i] = N;
        row_end[i] = -1;
        for (int j = 0; j <= last; ++j) {
            int64_t dj = (int64_t)j - N / 2;
            if (di * di + dj * dj <= radius_squared) {
                if (row_begin[i] == N) row_begin[i] = j;
                row_end[i] = j;
            }
        }
    }
}

/* Two phases per round:
 * 1) calculate q=floor(height/4) for every topplable cell;
 * 2) gather q from four neighboring cells into each output cell.
 * No threads write the same cell, so no atomics are necessary.
 * Padding is a sink for grains falling off the N x N physical grid.
 * Non-topplable cells have q=0 and can still receive grains.
 */
static void stabilize(void) {
    for (;;) {
        int changed = 0;
        #pragma omp parallel for schedule(static) reduction(|:changed)
        for (int i = 0; i < N; ++i) {
            size_t row = (size_t)(i + 1) * stride + 1;
            for (int j = row_begin[i]; j <= row_end[i]; ++j) {
                int q = grid[row + j] >> 2;
                topplings[row + j] = q;
                changed |= (q != 0);
            }
        }
        if (!changed) break;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            size_t row = (size_t)(i + 1) * stride + 1;
            for (int j = 0; j < N; ++j) {
                size_t id = row + j;
                grid[id] += -4 * topplings[id]
                            + topplings[id - stride]
                            + topplings[id - 1]
                            + topplings[id + stride]
                            + topplings[id + 1];
            }
        }
    }
}

static void find_identity(void) {
    for (int i = 0; i < N; ++i) {
        int *row = grid + (size_t)(i + 1) * stride + 1;
        for (int j = 0; j < N; ++j) row[j] = 6;
    }
    stabilize();
    for (int i = 0; i < N; ++i) {
        int *row = grid + (size_t)(i + 1) * stride + 1;
        for (int j = 0; j < N; ++j) row[j] = 6 - row[j];
    }
    stabilize();
}

static void render(const char *filename) {
    size_t count = (size_t)N * N;
    if (count > SIZE_MAX / 3) die("Image size overflow");
    unsigned char *pixels = allocate(count * 3);
    for (int i = 0; i < N; ++i) {
        const int *row = grid + (size_t)(i + 1) * stride + 1;
        for (int j = 0; j < N; ++j) {
            unsigned char r = 0, g = 0, b = 0;
            switch (row[j]) {
            case 0: r = 15;  g = 15;  b = 25;  break;
            case 1: r = 0;   g = 210; b = 255; break;
            case 2: r = 230; g = 30;  b = 130; break;
            case 3: r = 255; g = 200; b = 0;   break;
            default: break;
            }
            size_t id = ((size_t)i * N + j) * 3;
            pixels[id] = r;
            pixels[id + 1] = g;
            pixels[id + 2] = b;
        }
    }
    if (!stbi_write_png(filename, N, N, 3, pixels, 3 * N))
        die("Failed to write PNG");
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
        n == 0 || n > INT32_MAX - 2)
        die("grid size must be a positive int");
    N = (int)n;
    stride = N + 2;
    if ((size_t)stride > SIZE_MAX / (size_t)stride ||
        (size_t)stride * stride > SIZE_MAX / sizeof(int))
        die("Grid size overflow");
    size_t cells = (size_t)stride * stride;
    grid = calloc(cells, sizeof(int));
    topplings = calloc(cells, sizeof(int));
    row_begin = allocate((size_t)N * sizeof(int));
    row_end = allocate((size_t)N * sizeof(int));
    if (!grid || !topplings) die("Out of memory");
    initialize_region();
    find_identity();
    render("identity.png");
    free(row_end);
    free(row_begin);
    free(topplings);
    free(grid);
    return EXIT_SUCCESS;
}
