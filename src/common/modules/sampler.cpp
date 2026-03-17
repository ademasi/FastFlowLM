/// \file sampler.cpp
/// \brief sampler class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This is a header file for the sampler class
#pragma once

#include "modules/sampler.hpp"

#include <deque>
#include <cstdlib>    // for std::rand, std::srand, RAND_MAX
#include <ctime>      // for std::time
#include <algorithm>  // for std::sort
#include <cmath>      // for std::exp

/// \brief Constructor
/// \param in_features the input features
/// \param config the configuration
Sampler::Sampler(int in_features, sampler_config& config) {
    this->in_features           = in_features;
    this->logits.resize(in_features);
    this->counters.resize(in_features, 0);
    this->token_positions.resize(in_features, -1);
    this->top_k_logits.resize(config.top_k);

    this->temperature           = config.temperature;
    this->top_k                 = config.top_k;
    this->top_p                 = config.top_p;
    this->min_p                 = config.min_p;
    this->total_tokens          = 0;

    this->rep_penalty           = config.rep_penalty;
    this->freq_penalty          = config.freq_penalty;
    this->pre_penalty           = config.pre_penalty;
    this->rep_penalty_window    = config.rep_penalty_window;
    this->freq_penalty_window   = config.freq_penalty_window;
    this->repeat_last_n        = config.repeat_last_n;

    this->token_history.clear();
}

/// \brief Reset the penalties
/// \note The function will reset the penalties
/// \note The function will reset the token positions
/// \note The function will reset the token history
/// \note The function will reset the total tokens
/// \note The function will reset the token positions
void Sampler::reset_penalties() {
    for (int i = 0; i < in_features; i++) {
        this->counters[i]            = 0;
        this->token_positions[i]     = -1;
    }
    this->total_tokens = 0;
    this->token_history.clear();
}

void Sampler::softmax_inplace() {
    if (this->top_k_logits.empty()) 
        return;

    float max_l = this->top_k_logits[0].logits;
    for (int i = 1; i < this->top_k_logits.size(); ++i) {
        if (this->top_k_logits[i].logits > max_l) {
            max_l = this->top_k_logits[i].logits;
        }
    }

    double sum = 0.0;
    for (auto& kv : this->top_k_logits) {
        double x = std::exp(double(kv.logits - max_l));
        kv.prob = float(x); 
        sum += x;
    }

    float inv_sum = 1.0f / float(sum);
    for (auto& kv : this->top_k_logits) {
        kv.prob *= inv_sum;
    }
}

void Sampler::sampler_penalty_apply() {
    if ((this->repeat_last_n == 0) ||
        (this->rep_penalty == 1.0f && this->freq_penalty == 0.0f && this->pre_penalty == 0.0f)) {
        return;
    }

    // Apply frequency and presence penalties
    for (int token_id = 0; token_id < in_features; token_id++) {
        int count = this->counters[token_id];
        if (count <= 0) {
            continue; 
        }
        assert(count > 0 && count <= this->repeat_last_n);

        float logit = this->logits[token_id];

        if (this->logits[token_id] <= 0.0f) {
            this->logits[token_id] *= this->rep_penalty;
        }
        else {
            this->logits[token_id] /= this->rep_penalty;
        }

        this->logits[token_id] -= (float(count) * this->freq_penalty + float(count > 0) * this->pre_penalty);
    }
}

void Sampler::sampler_topk_apply(int k) {
    if (k == 0) {
        return;
    }

    logits_list_t pairs;
    pairs.resize(in_features);
    for (int i = 0; i < in_features; i++) {
        pairs[i].logits = this->logits[i];
        pairs[i].token_id = i;
        pairs[i].prob = 0.0f;
    }

    std::partial_sort(
        pairs.begin(),
        pairs.begin() + k,
        pairs.end(),
        [](const logits_t& a, const logits_t& b) {
            return a.logits > b.logits;
        }
    );

    this->top_k_logits.assign(pairs.begin(), pairs.begin() + k);
}

void Sampler::sampler_topp_apply(float p) {
    if (p >= 1.0f) {
        return;
    }

    float cum = 0.0f;
    int last_idx = top_k_logits.size();
    for (int i = 0; i < top_k_logits.size(); ++i) {
        cum += top_k_logits[i].prob;
        if (cum >= p) {
            last_idx = i + 1;
            break;
        }
    }
    if (last_idx < top_k_logits.size()) {
        top_k_logits.resize(last_idx);
    }
}

void Sampler::sampler_minp_apply(float p) {
    if (p <= 0.0f || p > 1) return;

    float max_logit = this->top_k_logits[0].logits;

    // logit >= max_logit + log(p)
    float min_logit = max_logit + std::log(p);
    for (int i = 0; i < this->top_k_logits.size(); i++) {
        if (this->top_k_logits[i].logits < min_logit) {
            this->top_k_logits.resize(i);
            break;
        }
    }
}

void Sampler::sampler_temp_apply(float temp) {
    if (temp == 0) { // Greedy sampling
        for (int i = 1; i < this->top_k_logits.size(); i++) {
            this->top_k_logits[i].logits = -std::numeric_limits<float>::infinity();
        }
        return;
    }

    for (int i = 0; i < this->top_k_logits.size(); i++) {
        this->top_k_logits[i].logits /= temp;
    }
}

int Sampler::sample_from_probs() {
    float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float cdf = 0.0f;
    int   sampled_index = 0;
    for (int i = 0; i <= top_k_logits.size(); i++) {
        cdf += this->top_k_logits[i].prob;
        if (u <= cdf) {
            sampled_index = this->top_k_logits[i].token_id;
            break;
        }
    }
    return sampled_index;
}

void Sampler::ring_buffer_update(int sampled_index) {
    // UPDATE RING BUFFER (token_history), COUNTERS, POSITIONS, total_tokens
    if (this->repeat_last_n > 0) {
        // Push new token and update its counter
        this->token_history.push_back(sampled_index);
        this->counters[sampled_index]++;

        // If buffer exceeds window, pop oldest and decrement its counter
        if (this->token_history.size() > this->repeat_last_n) {
            int oldest = this->token_history.front();
            this->token_history.pop_front();
            this->counters[oldest]--;
        }
    }

    // Update last‐seen position for repetition penalty
    this->token_positions[sampled_index] = this->total_tokens;

    // Advance global token count
    this->total_tokens++;
}

/// \brief Sample the token
/// \param x the input buffer
/// \return the sampled token
int Sampler::sample(buffer<bf16>& x) {
    // Re‐seed the PRNG each call:
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // COPY FROM `x` → `this->logits[]`
    #if USEAVX2
    const int simd_width = 8;
    int i = 0;
    for (; i <= in_features - simd_width; i += simd_width) {
        __m128i bf16_vals_x = _mm_loadu_si128(
            (__m128i*)&x[i]
        );
        __m256 fp32_vals_x = bf16o_fp32(bf16_vals_x);
        _mm256_storeu_ps(&this->logits[i], fp32_vals_x);
    }
    for (; i < in_features; i++) {
        this->logits[i] = x[i];
    }
    #else
    for (int i = 0; i < in_features; i++) {
        this->logits[i] = x[i];
    }
    #endif

    // PENALTIES -> TOP‐K -> softmax -> TOP-P -> softmax -> MIN-P -> TEMP -> softmax -> distribution
    sampler_penalty_apply();
    sampler_topk_apply(this->top_k);
    softmax_inplace();
    sampler_topp_apply(this->top_p);
    softmax_inplace();
    sampler_minp_apply(this->min_p);
    sampler_temp_apply(this->temperature);
    softmax_inplace();
    int sampled_index = sample_from_probs();
    ring_buffer_update(sampled_index);

    return sampled_index;
}