/**
 * Klein (2021) Q-learning collusion, CUDA implementation.
 *
 * All runs execute in parallel. Within each run, the T-period
 * loop is sequential (each step depends on the last).
 *
 * Memory notes:
 *
 * Build:
 *   nvcc -O3 -o collusion collusion.cu
 *
 * Run:
 *   ./collusion [num_runs] [T] [k]
 *
 * Output: writes results to "results.bin" as raw floats for Python analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>


#define SEED             42
#define DEFAULT_NUM_RUNS 1000
#define DEFAULT_T        500000
#define DEFAULT_K        6
#define ALPHA            0.3f
#define GAMMA            0.95f
#define GAMMA2           (GAMMA * GAMMA)  // delta^2
#define LAST_N           1000
#define MAX_PRICES       49       // goes up to k=48

// Constant memory price grid
__constant__ float c_prices[MAX_PRICES];


/**
 * Demand for firm i (eq. 3).
 * D_i = 1-p_i if p_i < p_j, 0.5*(1-p_i) if equal, 0 if p_i > p_j.
 */
__device__ float demand(float p_i, float p_j) {
    if (p_i < p_j)       return 1.0f - p_i;
    else if (p_i == p_j)  return 0.5f * (1.0f - p_i);
    else                   return 0.0f;
}

/**
 * Profit for firm i: price * demand (eq. 1).
 */
__device__ float profit(float p_i, float p_j) {
    return p_i * demand(p_i, p_j);
}

/**
 * Return index of maximum value in q_row[0..num_prices-1].
 * Ties broken by first occurrence!
 */
__device__ int argmax_q(float* q_row, int num_prices) {
    float best_val = q_row[0];
    int best_idx = 0;
    for (int i = 1; i < num_prices; i++) {
        if (q_row[i] > best_val) {
            best_val = q_row[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/**
 * Return the maximum value in q_row[0..num_prices-1].
 */
__device__ float max_q(float* q_row, int num_prices) {
    float best = q_row[0];
    for (int i = 1; i < num_prices; i++) {
        if (q_row[i] > best) {
            best = q_row[i];
        }
    }
    return best;
}

/**
 * Epsilon-greedy action selection (eq. 6).
 */
__device__ int eps_greedy(float* q_row, int num_prices, float eps,
                          curandState* rng) {
    float r = curand_uniform(rng);  // (0, 1]
    if (r <= eps) {
        // Random action: uniform over [0, num_prices)
        int a = (int)(curand_uniform(rng) * num_prices);
        if (a >= num_prices) a = num_prices - 1;  // edge case
        return a;
    } else {
        return argmax_q(q_row, num_prices);
    }
}


/**
 * Each thread runs one independent simulation of T periods.
 */
__global__ void run_simulation(
    int num_runs,
    int T,
    int num_prices,       // k+1
    float theta,          // epsilon decay: theta = 1 - 0.001^(2/T)
    float* d_final_profit,  // output [num_runs * 2]
    int*   d_final_prices,  // output [num_runs * 2]
    float* d_q_tables       // output [num_runs * 2 * num_prices * num_prices]
) {
    int run_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_id >= num_runs) return;

    curandState rng;
    curand_init(SEED + run_id, 0, 0, &rng);

    float Q[2][MAX_PRICES][MAX_PRICES];
    for (int a = 0; a < 2; a++)
        for (int s = 0; s < num_prices; s++)
            for (int act = 0; act < num_prices; act++)
                Q[a][s][act] = 0.0f;

    // Initial random prices
    int p[2];
    p[0] = (int)(curand_uniform(&rng) * num_prices);
    if (p[0] >= num_prices) p[0] = num_prices - 1;
    p[1] = (int)(curand_uniform(&rng) * num_prices);
    if (p[1] >= num_prices) p[1] = num_prices - 1;

    // ---- Delayed update storage ----
    int   last_action[2] = {0, 0};
    int   last_state[2]  = {0, 0};
    float last_profit[2] = {0.0f, 0.0f};

    // Epsilon per agent
    float eps[2] = {1.0f, 1.0f};

    // Profit accumulation for final LAST_N periods
    float profit_sum[2] = {0.0f, 0.0f};
    int   profit_count = 0;

    // Main loop
    for (int t = 0; t < T; t++) {
        int mover = t % 2;
        int other = 1 - mover;

        // State = opponent's current price index
        int state = p[other];

        // Mover picks action using epsilon-greedy
        int action = eps_greedy(Q[mover][state], num_prices, eps[mover], &rng);

        // Profit for mover
        float pi = profit(c_prices[action], c_prices[p[other]]);

        // Delayed Q-update for the agent who moved LAST period (= other)
        if (t > 0) {
            int prev_mover = other;
            int next_state = action;

            float pi_cont = profit(c_prices[last_action[prev_mover]],
                                   c_prices[action]);

            float target = last_profit[prev_mover]
                         + GAMMA * pi_cont
                         + GAMMA2 * max_q(Q[prev_mover][next_state], num_prices);

            float old_q = Q[prev_mover][last_state[prev_mover]][last_action[prev_mover]];
            Q[prev_mover][last_state[prev_mover]][last_action[prev_mover]] =
                old_q + ALPHA * (target - old_q);
        }

        // Update mover's price
        p[mover] = action;

        // Store for delayed update
        last_action[mover] = action;
        last_profit[mover] = pi;
        last_state[mover]  = state;

        // Decay epsilon for mover
        eps[mover] *= (1.0f - theta);

        // Accumulate profit for final LAST_N periods
        if (t >= T - LAST_N) {
            // Both firms' profit this period
            profit_sum[0] += profit(c_prices[p[0]], c_prices[p[1]]);
            profit_sum[1] += profit(c_prices[p[1]], c_prices[p[0]]);
            profit_count++;
        }
    }

    // Write outputs
    int base = run_id * 2;

    // Average profit over final LAST_N periods
    d_final_profit[base + 0] = profit_sum[0] / (float)profit_count;
    d_final_profit[base + 1] = profit_sum[1] / (float)profit_count;

    // Final prices (as indices)
    d_final_prices[base + 0] = p[0];
    d_final_prices[base + 1] = p[1];

    // Q-tables: flatten Q[2][num_prices][num_prices]
    int q_size = num_prices * num_prices;
    int q_base = run_id * 2 * q_size;
    for (int a = 0; a < 2; a++)
        for (int s = 0; s < num_prices; s++)
            for (int act = 0; act < num_prices; act++)
                d_q_tables[q_base + a * q_size + s * num_prices + act] =
                    Q[a][s][act];
}



// Host code
int main(int argc, char** argv) {
    // Parse CLI args
    int num_runs  = (argc > 1) ? atoi(argv[1]) : DEFAULT_NUM_RUNS;
    int T         = (argc > 2) ? atoi(argv[2]) : DEFAULT_T;
    int k         = (argc > 3) ? atoi(argv[3]) : DEFAULT_K;
    int num_prices = k + 1;

    if (num_prices > MAX_PRICES) {
        fprintf(stderr, "Error: k=%d gives %d prices, max is %d\n",
                k, num_prices, MAX_PRICES);
        return 1;
    }

    // Epsilon decay: theta = 1 - 0.001^(2/T)
    // As described in paper for replicability
    float theta = 1.0f - powf(0.001f, 2.0f / (float)T);

    printf("Klein (2021) Q-learning Collusion — CUDA\n");
    printf("  runs=%d, T=%d, k=%d (num_prices=%d)\n", num_runs, T, k, num_prices);
    printf("  alpha=%.2f, gamma=%.2f, theta=%.8f\n", ALPHA, GAMMA, theta);
    printf("  Collusive profit=0.1250, Competitive~0.0611\n\n");

    float h_prices[MAX_PRICES];
    for (int i = 0; i < num_prices; i++) {
        h_prices[i] = (float)i / (float)k;
    }

    // Copy price grid to constant memory
    cudaMemcpyToSymbol(c_prices, h_prices, num_prices * sizeof(float));

    int q_table_size = num_runs * 2 * num_prices * num_prices;

    float* d_final_profit;
    int*   d_final_prices;
    float* d_q_tables;

    cudaMalloc(&d_final_profit, num_runs * 2 * sizeof(float));
    cudaMalloc(&d_final_prices, num_runs * 2 * sizeof(int));
    cudaMalloc(&d_q_tables,     q_table_size * sizeof(float));

    // Launch kernel
    int threads_per_block = 128;
    int num_blocks = (num_runs + threads_per_block - 1) / threads_per_block;

    printf("Launching %d blocks x %d threads...\n", num_blocks, threads_per_block);

    run_simulation<<<num_blocks, threads_per_block>>>(
        num_runs, T, num_prices, theta,
        d_final_profit, d_final_prices, d_q_tables
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Done.\n\n");

    // Copy results back to host
    float* h_final_profit = (float*)malloc(num_runs * 2 * sizeof(float));
    int*   h_final_prices = (int*)malloc(num_runs * 2 * sizeof(int));
    float* h_q_tables     = (float*)malloc(q_table_size * sizeof(float));

    cudaMemcpy(h_final_profit, d_final_profit, num_runs * 2 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_prices, d_final_prices, num_runs * 2 * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_q_tables, d_q_tables, q_table_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Summary stats
    float total_profit = 0.0f;
    int supra_competitive = 0;  // profit > competitive benchmark (~0.0611)
    float competitive_bench = 0.0611f;

    for (int r = 0; r < num_runs; r++) {
        float avg = (h_final_profit[r*2] + h_final_profit[r*2+1]) / 2.0f;
        total_profit += avg;
        if (avg > competitive_bench) supra_competitive++;
    }

    float mean_profit = total_profit / (float)num_runs;

    // Stdev
    float var = 0.0f;
    for (int r = 0; r < num_runs; r++) {
        float avg = (h_final_profit[r*2] + h_final_profit[r*2+1]) / 2.0f;
        var += (avg - mean_profit) * (avg - mean_profit);
    }
    float std_profit = sqrtf(var / (float)num_runs);

    printf("Results over %d runs:\n", num_runs);
    printf("  Mean profit:        %.4f\n", mean_profit);
    printf("  Std profit:         %.4f\n", std_profit);
    printf("  Supra-competitive:  %d / %d (%.1f%%)\n",
           supra_competitive, num_runs,
           100.0f * supra_competitive / (float)num_runs);
    printf("  Collusive bench:    0.1250\n");
    printf("  Competitive bench: ~0.0611\n");

    // Writing binary output for Python analysis
    const char* outpath = "results.bin";
    FILE* fp = fopen(outpath, "wb");
    if (fp) {
        int header[4] = {num_runs, T, k, num_prices};
        fwrite(header, sizeof(int), 4, fp);
        fwrite(h_final_profit, sizeof(float), num_runs * 2, fp);
        fwrite(h_final_prices, sizeof(int), num_runs * 2, fp);
        fwrite(h_q_tables, sizeof(float), q_table_size, fp);
        fclose(fp);
        printf("\nResults written to %s\n", outpath);
        printf("For full analysis: python analyze.py --summary --figures\n");
    } else {
        fprintf(stderr, "Warning: could not open %s for writing\n", outpath);
    }

    // Cleaning
    free(h_final_profit);
    free(h_final_prices);
    free(h_q_tables);
    cudaFree(d_final_profit);
    cudaFree(d_final_prices);
    cudaFree(d_q_tables);

    return 0;
}
