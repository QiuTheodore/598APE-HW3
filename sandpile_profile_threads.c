/* CS598APE HW3: multigrid + symmetry + streamlined stabilization (v2).
 * Candidate v2: simplify hot-loop border logic with sink halo, and
 * use eight W-cycles by default with residual correction.
 * Keep all original best-version files untouched.
 *
 * Original explanation: symmetry-aware multigrid-seeded OpenMP experiment.
 *
 * For odd N, the reference Pacman region and all-6 input are invariant
 * under reflection i -> N-1-i.  We therefore store and stabilize only
 * rows 0..N/2; pixels in the bottom half are mirrored at rendering time.
 *
 * Odd center boundary:
 * - Row just above center sends 2*q into the center (it represents both
 *   the row above and its reflected partner below the center).
 * - Center sends q to the row above; its reflection receives the same q.
 *
 * Even N has no such symmetry because the original circle is centered
 * at N/2.  It uses the full grid, avoiding incorrect assumptions.
 *
 * Each OpenMP stripe mutates only its own rows.  Inter-stripe grains
 * are buffered until a barrier.  All updates are legal batched topplings.
 *
 * Compile: gcc -O3 -g -Werror -fopenmp sandpile_multigrid_v2.c -lm
 *          -o sandpile_multigrid_v2.exe
 * Run:     OMP_NUM_THREADS=4 ./sandpile_multigrid_v2.exe 1001
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

static int N, stride, rows;
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
    for (int i = 0; i < rows; ++i) {
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

/* Alternating Gauss-Seidel sweeps with buffered, race-free stripe
 * boundary transfers. Inactive Pacman cells are never toppled, but do
 * receive grains, and participate in the 6-grid subtraction. */
static void stabilize(void) {
    const double profile_start = omp_get_wtime();
    long long profile_sweeps = 0;
    int P = omp_get_max_threads();
    if (P > rows) P = rows;
    if (P < 1) P = 1;
    int *starts = allocate((size_t)(P+1) * sizeof(int));
    int *changed_flags = allocate((size_t)P * sizeof(int));
    int global_any = 0;
    if ((size_t)P > SIZE_MAX / (size_t)N / sizeof(int))
        die("Boundary buffer overflow");
    int *send_up = calloc((size_t)P * N, sizeof(int));
    int *send_down = calloc((size_t)P * N, sizeof(int));
    if (!send_up || !send_down) die("Out of memory");
    starts[0] = 0;
    starts[P] = rows;
    int64_t total_area=0;
    for (int i=0;i<rows;++i) total_area+=row_end[i]-row_begin[i]+1;
    int rr=0;int64_t prefix=0;
    for(int t=1;t<P;++t){
        int64_t target=total_area*t/P;
        int minrow=starts[t-1]+1;
        int maxrow=rows-(P-t);
        while(rr<minrow){
            prefix+=row_end[rr]-row_begin[rr]+1;
            ++rr;
        }
        while(rr<maxrow && prefix<target){
            prefix+=row_end[rr]-row_begin[rr]+1;
            ++rr;
        }
        starts[t]=rr;
    }

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const int begin = starts[tid], end = starts[tid+1];
        int *up = send_up + (size_t)tid * N;
        int *down = send_down + (size_t)tid * N;
        int reverse = 0;
        double profile_scan_s=0,profile_first_wait_s=0,profile_exchange_and_wait_s=0;
        for (;;) {
            double profile_sweep_t0=omp_get_wtime();
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
                        /* Physical grid borders are represented by sink padding.
                         * Skipped values in halo are never toppled or rendered. */
                        if (i == begin) up[j] += q;
                        else grid[id-stride] += q;
                        int send = (N & 1) && (i+1 == rows-1) ? 2*q : q;
                        if (i+1 == end) down[j] += send;
                        else grid[id+stride] += send;
                        grid[id-1] += q;
                        grid[id+1] += q;
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
                        /* Physical grid borders are represented by sink padding.
                         * Skipped values in halo are never toppled or rendered. */
                        if (i == begin) up[j] += q;
                        else grid[id-stride] += q;
                        int send = (N & 1) && (i+1 == rows-1) ? 2*q : q;
                        if (i+1 == end) down[j] += send;
                        else grid[id+stride] += send;
                        grid[id-1] += q;
                        grid[id+1] += q;
                        did_topple=1;
                    }
                }
            }
            changed_flags[tid]=did_topple;
            double profile_scan_t1=omp_get_wtime();
            profile_scan_s+=profile_scan_t1-profile_sweep_t0;
            #pragma omp barrier
            double profile_after_barrier=omp_get_wtime();
            profile_first_wait_s+=profile_after_barrier-profile_scan_t1;
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
            /* The single's implicit barrier both finishes exchanges and
             * publishes one consistent termination decision.  No data race
             * occurs on changed_flags when the next sweep begins. */
            #pragma omp single
            {
                global_any=0;
                for(int t=0;t<P;++t)global_any|=changed_flags[t];
                ++profile_sweeps;
            }
            profile_exchange_and_wait_s+=omp_get_wtime()-profile_after_barrier;
            if(!global_any)break;
            memset(up,0,(size_t)N*sizeof(int));
            memset(down,0,(size_t)N*sizeof(int));
            reverse^=1;
        }
        #pragma omp critical
        fprintf(stderr,"PROFILE thread id=%d stripe=[%d,%d) scan=%.6fs first_barrier=%.6fs exchange_and_wait=%.6fs\n",tid,begin,end,profile_scan_s,profile_first_wait_s,profile_exchange_and_wait_s);
    }
    free(send_down);
    free(send_up);
    free(changed_flags);
    free(starts);
    fprintf(stderr, "PROFILE stabilize: sweeps=%lld, wall=%.6f s, threads=%d\n", profile_sweeps, omp_get_wtime()-profile_start, P);
}

/* Use the discrete maximum principle to conservatively preload a lower
 * bound on legal toppling counts.  The subsequent stabilization corrects it
 * exactly.  The bound works even if the intermediate grid has negatives. */
/* Geometric multigrid (W cycles) for approximate L*u=f on the full domain.
 * We use a posteriori maximum-principle bound to convert the approximation
 * into a guaranteed lower bound for the exact odometer. */
typedef struct {
    int n, stride;
    double *u,*f,*r;
    unsigned char *mask;
} MGLevel;

static MGLevel levels[16];
static int nlevels;

static void mg_smooth(MGLevel *p, int sweeps){
    const int n=p->n, st=p->stride;
    double *u=p->u, *f=p->f;
    const unsigned char *mask=p->mask;
    for(int z=0;z<sweeps;++z){
        if(z&1){
            for(int i=n-1;i>=0;--i){
                int base=(i+1)*st+1;
                for(int j=n-1;j>=0;--j){
                    int id=base+j;
                    if(mask[id])u[id]=(f[id]+u[id-st]+u[id+st]+u[id-1]+u[id+1])*0.25;
                }
            }
        }else{
            for(int i=0;i<n;++i){
                int base=(i+1)*st+1;
                for(int j=0;j<n;++j){
                    int id=base+j;
                    if(mask[id])u[id]=(f[id]+u[id-st]+u[id+st]+u[id-1]+u[id+1])*0.25;
                }
            }
        }
    }
}
static double mg_residual(MGLevel *p){
    const int n=p->n,st=p->stride;
    double rmax=0;
    for(int i=0;i<n;++i){
        int base=(i+1)*st+1;
        for(int j=0;j<n;++j){
            int id=base+j;
            if(!p->mask[id]){p->r[id]=0;continue;}
            double res=p->f[id] - (4*p->u[id]-p->u[id-st]-p->u[id+st]-p->u[id-1]-p->u[id+1]);
            p->r[id]=res;
            double magnitude=res<0?-res:res;
            if(magnitude>rmax)rmax=magnitude;
        }
    }
    return rmax;
}
static void mg_vcycle(int l){
    MGLevel *fine=&levels[l];
    if(l+1==nlevels){mg_smooth(fine,50);return;}
    mg_smooth(fine,4);
    (void)mg_residual(fine);
    MGLevel *coarse=&levels[l+1];
    const int n=coarse->n, cst=coarse->stride, fst=fine->stride;
    for(int i=0;i<n;++i){
        int b=(i+1)*cst+1;
        int fb=(2*i+1)*fst+1;
        for(int j=0;j<n;++j){
            int c=b+j;
            if(!coarse->mask[c])continue;
            int f=fb+2*j;
            double *r=fine->r;
            double val=4*r[f] +2*(r[f-1]+r[f+1]+r[f-fst]+r[f+fst])
                               +(r[f-fst-1]+r[f-fst+1]+r[f+fst-1]+r[f+fst+1]);
            coarse->f[c] = 0.25*val; /* factor 4 for coarse grid spacing */
        }
    }
    memset(coarse->u,0,(size_t)(n+2)*cst*sizeof(double));
    mg_vcycle(l+1);
    if (l<4) mg_vcycle(l+1);
    for(int i=0;i<fine->n;++i){
        int fb=(i+1)*fst+1;
        int ctop=(i/2+1)*cst+1;
        for(int j=0;j<fine->n;++j){
            int fid=fb+j;
            if(!fine->mask[fid])continue;
            int cid=ctop+j/2;
            double d=coarse->u[cid];
            if(i&1){
                if(j&1)d=0.25*(d+coarse->u[cid+1]+coarse->u[cid+cst]+coarse->u[cid+cst+1]);
                else d=0.5*(d+coarse->u[cid+cst]);
            }else if(j&1)d=0.5*(d+coarse->u[cid+1]);
            fine->u[fid]+=d;
        }
    }
    mg_smooth(fine,4);
}

static void mg_init(void){
    nlevels=0;
    int n=N;
    while(nlevels<16){
        MGLevel *l=&levels[nlevels];
        l->n=n;l->stride=n+2;
        size_t cells=(size_t)(n+2)*(n+2);
        l->u=calloc(cells,sizeof(double));l->f=calloc(cells,sizeof(double));
        l->r=calloc(cells,sizeof(double));l->mask=calloc(cells,sizeof(unsigned char));
        if(!l->u||!l->f||!l->r||!l->mask)die("MG allocation failed");
        if(nlevels==0){
            for(int i=0;i<N;++i){
                int src=i<rows?i:N-1-i;
                for(int j=row_begin[src];j<=row_end[src];++j)
                    l->mask[(size_t)(i+1)*l->stride+1+j]=1;
            }
        }else{
            MGLevel *prev=&levels[nlevels-1];
            for(int i=0;i<n;++i){
                int row=(i+1)*l->stride+1;
                int prevrow=(2*i+1)*prev->stride+1;
                for(int j=0;j<n;++j)
                    l->mask[row+j]=prev->mask[prevrow+2*j];
            }
        }
        nlevels++;
        if(n<=7)break;
        n=(n+1)/2;
    }
}
static void mg_destroy(void){
    for(int i=0;i<nlevels;++i){
        free(levels[i].u);free(levels[i].f);free(levels[i].r);free(levels[i].mask);
    }
}

static void seed_odometer(void) {
    if (!(N&1) || N < 9) return;
    mg_init();
    MGLevel *fine=&levels[0];
    for(int i=0;i<N;++i){
        int src=i<rows?i:N-1-i;
        size_t gridrow=(size_t)(src+1)*stride+1;
        int mgrow=(i+1)*fine->stride+1;
        for(int j=row_begin[src];j<=row_end[src];++j){
            int val=grid[gridrow+j]-3;
            if(val<0)die("Invalid negative source for MG lower bound");
            fine->f[mgrow+j]=val;
        }
    }
    int cycles=8; /* screened empirically; override via HW3_MG_CYCLES */
    int done_cycles=0;
    const char *c=getenv("HW3_MG_CYCLES");
    if(c){int v=atoi(c);if(v>=1 && v<=500)cycles=v;}
    double rmax=0;
    for(int i=0;i<cycles;++i){
        mg_vcycle(0);
        done_cycles=i+1;
        rmax=mg_residual(fine);
        if(rmax < 1e-4)break;
    }
    /* If L*w <= f + rmax, w - rmax*psi <= L^-1(f), where
     * psi=(N^2 -(x-c)^2-(y-c)^2)/4 and L*psi >= 1 on the mask.
     * Since the final stable pile is <=3, the true odometer is >=L^-1(f).
     * Floor that bound and leave an extra two topplings margin. */
    const size_t cells=(size_t)(rows+2)*stride;
    int *u=calloc(cells,sizeof(int));
    if(!u)die("MG seed allocation failed");
    long long total=0;
    for(int i=0;i<rows;++i){
        size_t row=(size_t)(i+1)*stride+1;
        int mrow=(i+1)*fine->stride+1;
        double di=(double)i-N/2;
        for(int j=row_begin[i];j<=row_end[i];++j){
            double dj=(double)j-N/2;
            double psi=((double)N*N-di*di-dj*dj)*0.25;
            double lo=fine->u[mrow+j]-(rmax+1e-8)*psi-2.0;
            if(lo>0){
                if(lo>INT32_MAX)die("MG odometer too large");
                u[row+j]=(int)lo;
                total+=u[row+j];
            }
        }
    }
    fprintf(stderr,"MG cycles=%d, max_residual=%.8f, seeded_topplings=%lld\n",
            done_cycles,rmax,total);
    for(int i=0;i<rows;++i){
        size_t row=(size_t)(i+1)*stride+1;
        for(int j=row_begin[i];j<=row_end[i];++j){
            size_t id=row+j;
            int q=u[id];
            if(!q)continue;
            grid[id]-=4*q;
            if(i>0)grid[id-stride]+=q;
            if(i+1<rows)grid[id+stride]+=((N&1)&&i+1==rows-1)?2*q:q;
            if(j>0)grid[id-1]+=q;
            if(j+1<N)grid[id+1]+=q;
        }
    }
    free(u);
    mg_destroy();
}

static void find_identity(void) {
    double t0=omp_get_wtime();
    for (int i = 0; i < rows; ++i) {
        int *row = grid + (size_t)(i + 1) * stride + 1;
        for (int j = 0; j < N; ++j) row[j] = 6;
    }
    fprintf(stderr, "PROFILE initialize: wall=%.6f s\n", omp_get_wtime()-t0);
    double t1=omp_get_wtime();
    seed_odometer();
    fprintf(stderr, "PROFILE seed_phase1: wall=%.6f s\n", omp_get_wtime()-t1);
    t1=omp_get_wtime();
    stabilize();
    fprintf(stderr, "PROFILE stabilize_phase1: wall=%.6f s\n", omp_get_wtime()-t1);
    for (int i = 0; i < rows; ++i) {
        int *row = grid + (size_t)(i + 1) * stride + 1;
        for (int j = 0; j < N; ++j) row[j] = 6 - row[j];
    }
    t1=omp_get_wtime();
    seed_odometer();
    fprintf(stderr, "PROFILE seed_phase2: wall=%.6f s\n", omp_get_wtime()-t1);
    t1=omp_get_wtime();
    stabilize();
    fprintf(stderr, "PROFILE stabilize_phase2: wall=%.6f s\n", omp_get_wtime()-t1);
}

static void render(const char *filename) {
    size_t count = (size_t)N * N;
    if (count > SIZE_MAX / 3) die("Image size overflow");
    unsigned char *pixels = allocate(count * 3);
    for (int i = 0; i < N; ++i) {
        const int src = i < rows ? i : N - 1 - i;
        const int *row = grid + (size_t)(src + 1) * stride + 1;
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
    rows = (N & 1) ? (N + 1) / 2 : N;
    if ((size_t)(rows + 2) > SIZE_MAX / (size_t)stride ||
        (size_t)(rows + 2) * stride > SIZE_MAX / sizeof(int))
        die("Grid size overflow");
    size_t cells = (size_t)(rows + 2) * stride;
    grid = calloc(cells, sizeof(int));
    row_begin = allocate((size_t)rows * sizeof(int));
    row_end = allocate((size_t)rows * sizeof(int));
    if (!grid) die("Out of memory");
    initialize_region();
    double profile_t=omp_get_wtime();
    find_identity();
    fprintf(stderr, "PROFILE find_identity_total: wall=%.6f s\n", omp_get_wtime()-profile_t);
    profile_t=omp_get_wtime();
    render("identity.png");
    fprintf(stderr, "PROFILE render: wall=%.6f s\n", omp_get_wtime()-profile_t);
    free(row_end);
    free(row_begin);
    free(grid);
    return EXIT_SUCCESS;
}
