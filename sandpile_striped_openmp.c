/* CS598APE HW3: experimental striped parallel Gauss-Seidel implementation.
 *
 * Each OpenMP worker owns a disjoint group of whole rows, so all within-stripe
 * batch topplings modify only thread-owned grid cells. Contributions crossing
 * the top/bottom of the stripe are buffered, exchanged after a barrier, and
 * applied by the receiving stripe's owner. A second barrier ensures buffers
 * can be safely reused. Each batch toppling is legal; no thread writes a cell
 * owned by another thread. The exact reference Pacman region is preserved.
 *
 * Build: gcc -O3 -g -Werror -fopenmp sandpile_striped_openmp.c -lm -o sandpile_striped.exe
 * Run:   OMP_NUM_THREADS=4 ./sandpile_striped.exe 1001
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int N, stride;
static int *grid;
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

static void stabilize(void) {
    int P = omp_get_max_threads();
    if (P > N) P = N;
    if (P < 1) P = 1;
    int *starts = allocate((size_t)(P+1) * sizeof(int));
    int *changed_flags = allocate((size_t)P * sizeof(int));
    int *send_up = calloc((size_t)P*N, sizeof(int));
    int *send_down = calloc((size_t)P*N, sizeof(int));
    if (!send_up || !send_down) die("Out of memory");
    starts[0] = 0;
    starts[P] = N;
    for (int t = 1; t < P; ++t) starts[t] = (int)((int64_t)N*t/P);

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const int begin = starts[tid], end = starts[tid+1];
        int *up = send_up + (size_t)tid * N;
        int *down = send_down + (size_t)tid * N;
        int reverse = 0;
        for (;;) {
            int did_topple = 0;
            if (!reverse) {
                for (int i = begin; i < end; ++i) {
                    size_t row = (size_t)(i+1)*stride + 1;
                    for (int j = row_begin[i]; j <= row_end[i]; ++j) {
                        size_t id = row+j;
                        int v = grid[id];
                        if (v < 4) continue;
                        int q=v>>2;
                        grid[id] = v&3;
                        if (i > 0) { if (i == begin) up[j] += q; else grid[id-stride] += q; }
                        if (i+1 < N) { if (i+1 == end) down[j] += q; else grid[id+stride] += q; }
                        if (j>0) grid[id-1] += q;
                        if (j+1 < N) grid[id+1] += q;
                        did_topple = 1;
                    }
                }
            } else {
                for (int i = end-1; i >= begin; --i) {
                    size_t row = (size_t)(i+1)*stride + 1;
                    for (int j = row_end[i]; j >= row_begin[i]; --j) {
                        size_t id = row+j;
                        int v=grid[id];
                        if(v<4) continue;
                        int q=v>>2;
                        grid[id]=v&3;
                        if(i>0){ if(i==begin) up[j]+=q; else grid[id-stride]+=q; }
                        if(i+1<N){ if(i+1==end) down[j]+=q; else grid[id+stride]+=q; }
                        if(j>0)grid[id-1]+=q;
                        if(j+1<N)grid[id+1]+=q;
                        did_topple=1;
                    }
                }
            }
            changed_flags[tid]=did_topple;
            #pragma omp barrier
            if(tid>0){
                int *receive=send_down+(size_t)(tid-1)*N;
                int *row = grid+(size_t)(begin+1)*stride+1;
                for(int j=0;j<N;++j)row[j]+=receive[j];
            }
            if(tid+1<P){
                int *receive=send_up+(size_t)(tid+1)*N;
                int *row = grid+(size_t)end*stride+1;
                for(int j=0;j<N;++j)row[j]+=receive[j];
            }
            #pragma omp barrier
            int any=0;
            for(int t=0;t<P;++t)any|=changed_flags[t];
            if(!any)break;
            memset(up,0,(size_t)N*sizeof(int));
            memset(down,0,(size_t)N*sizeof(int));
            reverse^=1;
        }
    }
    free(send_down);
    free(send_up);
    free(changed_flags);
    free(starts);
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
    row_begin = allocate((size_t)N * sizeof(int));
    row_end = allocate((size_t)N * sizeof(int));
    if (!grid) die("Out of memory");
    initialize_region();
    find_identity();
    render("identity.png");
    free(row_end);
    free(row_begin);
    free(grid);
    return EXIT_SUCCESS;
}
