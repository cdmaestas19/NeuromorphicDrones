/* The escape subcircuit's LIF network, stepped in C: the Shiu et al. 2024 model (model.py of the
 * Drosophila_brain_model repo), the same equations, parameters and order of events as the Brian2 network
 * in circuit.py, without Brian2's per-step Python overhead (250 steps per 50 ms tick at 0.2 ms).
 * fastlif.py builds it with gcc and calls it through ctypes; check_fast.py checks it against Brian2.
 *
 * Per neuron:  dv/dt = (v_0 - v + g) / t_mbr,  dg/dt = -g / tau,  both frozen while refractory.
 * Linear, so each step is the exact solution:  v <- v_0 + (v - v_0) ev + g cvg,  g <- g eg.
 * Spike when v > v_th and not refractory; then v = v_rst, g = 0, refractory for rfc steps.
 * A spike adds the synapse's weight to the target's g `delay` steps later.
 * Input neurons get Poisson kicks: v += kick, with probability p each step.
 *
 * One step, in Brian2's default schedule order:
 *   groups      integrate every neuron that is not refractory
 *   thresholds  detect spikes (and the Poisson inputs fire)
 *   synapses    deliver the effects due now; queue this step's spikes; apply the Poisson kicks
 *   resets      spiking neurons back to v_rst, g = 0 (an input arriving the same step is lost, as in Brian2)
 * Input reaching a neuron while it is refractory is dropped: v and g are "(unless refractory)" variables,
 * and Brian2 guards every synaptic update of such a variable with `if not_refractory` (found by comparing
 * the two step by step: without it this network spiked ~5% more).
 */
#include <stddef.h>
#include <stdint.h>

#define TINY 1e-12  /* mV */

static inline uint64_t next(uint64_t *s) {  /* xorshift64 */
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static inline double uniform(uint64_t *s) { return (double)(next(s) >> 11) * (1.0 / 9007199254740992.0); }

/* Run `steps` steps from global step t0. State arrays are the caller's (numpy), updated in place.
 * ring: D x n pending g increments, the row for step t is t % D (D > delay).
 * in_spikes: steps x n_in input spikes (1/0) to use instead of the Poisson draws, or NULL.
 * counts: spikes per neuron, added to. spk: scratch, n ints. Returns the number of spikes. */
long lif_run(int n, int steps, int64_t t0,
             double *v, double *g, int64_t *last, const int32_t *rfc,
             double ev, double eg, double cvg, double v0, double vrst, double vth,
             const int32_t *row_ptr, const int32_t *post, const double *w,
             double *ring, int D, int delay,
             int n_in, const int32_t *idx_in, const double *p_in, double kick, const uint8_t *in_spikes,
             uint64_t *rng, int32_t *counts, int32_t *spk)
{
    long total = 0;
    for (int s = 0; s < steps; s++) {
        const int64_t t = t0 + s;
        for (int i = 0; i < n; i++) {
            if (t - last[i] >= rfc[i]) {
                const double gi = g[i];
                double u = (v[i] - v0) * ev + gi * cvg, gn = gi * eg;
                /* Flush what has decayed to nothing: a quiet network's g and v - v_0 keep shrinking
                 * geometrically, into subnormal numbers within seconds, which cost ~100x per operation
                 * (16.7 ms per tick in flight instead of 1.4). 1e-12 mV moves nothing across a threshold. */
                if (u < TINY && u > -TINY)
                    u = 0.0;
                if (gn < TINY && gn > -TINY)
                    gn = 0.0;
                v[i] = v0 + u;
                g[i] = gn;
            }
        }
        int ns = 0;
        for (int i = 0; i < n; i++) {
            if (v[i] > vth && t - last[i] >= rfc[i]) {
                spk[ns++] = i;
                last[i] = t;
                counts[i]++;
            }
        }
        double *due = ring + (size_t)(t % D) * (size_t)n;
        for (int i = 0; i < n; i++) {
            if (t - last[i] >= rfc[i])  /* g is "unless refractory": Brian2 drops input arriving meanwhile */
                g[i] += due[i];
            due[i] = 0.0;
        }
        double *later = ring + (size_t)((t + delay) % D) * (size_t)n;
        for (int k = 0; k < ns; k++) {
            const int i = spk[k];
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
                later[post[j]] += w[j];
        }
        for (int k = 0; k < n_in; k++) {
            const int fire = in_spikes ? in_spikes[(size_t)s * (size_t)n_in + (size_t)k] : uniform(rng) < p_in[k];
            if (fire)
                v[idx_in[k]] += kick;
        }
        for (int k = 0; k < ns; k++) {
            v[spk[k]] = vrst;
            g[spk[k]] = 0.0;
        }
        total += ns;
    }
    return total;
}
