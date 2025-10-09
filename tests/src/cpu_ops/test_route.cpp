#include "cpu_ops/test_route.hpp"
#include "minfer/ops/kernels.hpp"

TestRoute::TestRoute(const std::string& name) : TestBase(name) {}

void TestRoute::test_single_expert() {
    int d_model = 16;
    int n_experts = 4;
    int n_active_experts = 1;
    
    float x_norm[16] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f,
                        3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f};
    
    float w_router[64]; // 4 experts × 16 d_model
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            if (i == 3) {
                w_router[i * 16 + j] = 1.0f; // expert 3 gets highest score
            } else if (i == j % 3) {
                w_router[i * 16 + j] = 1.0f;
            } else {
                w_router[i * 16 + j] = 0.0f;
            }
        }
    }
    
    int active_experts[1] = {0};
    float active_experts_scores[1] = {0.0f};
    float active_experts_weights[1] = {0.0f};
    float moe_scores[4] = {0.0f};
    
    cpu::route(x_norm, active_experts, active_experts_scores, active_experts_weights,
          moe_scores, w_router, d_model, n_experts, n_active_experts);
    
    assert_equal(3, active_experts[0], 1e-6f, "Highest scoring expert selected");
    assert_equal(1.0f, active_experts_weights[0], 1e-6f, "Single expert assigned weight 1.0");
}

void TestRoute::test_top_k_selection() {
    int d_model = 16;
    int n_experts = 5;
    int n_active_experts = 3;
    
    float x_norm[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    float w_router[80]; // 5 experts × 16 d_model
    for (int i = 0; i < 5; i++) {
        float weight = (i == 0) ? 2.0f : (i == 1) ? 4.0f : (i == 2) ? 1.0f : 
                      (i == 3) ? 6.0f : 3.0f;
        for (int j = 0; j < 16; j++) {
            w_router[i * 16 + j] = weight;
        }
    }
    
    int active_experts[3] = {0};
    float active_experts_scores[3] = {0.0f};
    float active_experts_weights[3] = {0.0f};
    float moe_scores[5] = {0.0f};
    
    cpu::route(x_norm, active_experts, active_experts_scores, active_experts_weights,
          moe_scores, w_router, d_model, n_experts, n_active_experts);
    
    assert_equal(3, active_experts[0], 1e-6f, "1st expert chosen, highest scoring");
    assert_equal(1, active_experts[1], 1e-6f, "2nd expert chosen, second highest");
    assert_equal(4, active_experts[2], 1e-6f, "3rd expert chosen, third highest");
}

void TestRoute::test_score_normalization() {
    int d_model = 16;
    int n_experts = 3;
    int n_active_experts = 2;
    
    float x_norm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    float w_router[48]; // 3 experts × 16 d_model
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 16; j++) {
            if (j == 0) {
                w_router[i * 16 + j] = (i == 0) ? 10.0f : (i == 1) ? 5.0f : 1.0f;
            } else {
                w_router[i * 16 + j] = 0.0f;
            }
        }
    }
    
    int active_experts[2] = {0};
    float active_experts_scores[2] = {0.0f};
    float active_experts_weights[2] = {0.0f};
    float moe_scores[3] = {0.0f};
    
    cpu::route(x_norm, active_experts, active_experts_scores, active_experts_weights,
          moe_scores, w_router, d_model, n_experts, n_active_experts);
    
    float weight_sum = active_experts_weights[0] + active_experts_weights[1];
    assert_equal(1.0f, weight_sum, 1e-6f, "Expert weights sum to 1.0");
    assert_true(active_experts_weights[0] > active_experts_weights[1], "Higher score gets higher weight");
}

void TestRoute::test_expert_ordering() {
    int d_model = 16;
    int n_experts = 4;
    int n_active_experts = 4;
    
    float x_norm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    float w_router[64]; // 4 experts × 16 d_model
    float weights[4] = {2.0f, 4.0f, 1.0f, 3.0f}; // scores: 2, 4, 1, 3
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            w_router[i * 16 + j] = (j == 0) ? weights[i] : 0.0f;
        }
    }
    
    int active_experts[4] = {0};
    float active_experts_scores[4] = {0.0f};
    float active_experts_weights[4] = {0.0f};
    float moe_scores[4] = {0.0f};
    
    cpu::route(x_norm, active_experts, active_experts_scores, active_experts_weights,
          moe_scores, w_router, d_model, n_experts, n_active_experts);
    
    assert_equal(1, active_experts[0], 1e-6f, "Expert 1 ranked first (score=4)");
    assert_equal(3, active_experts[1], 1e-6f, "Expert 3 ranked second (score=3)");
    assert_equal(0, active_experts[2], 1e-6f, "Expert 0 ranked third (score=2)");
    assert_equal(2, active_experts[3], 1e-6f, "Expert 2 ranked fourth (score=1)");
    
    assert_true(active_experts_scores[0] >= active_experts_scores[1], "Scores in descending order");
    assert_true(active_experts_scores[1] >= active_experts_scores[2], "Scores in descending order");
    assert_true(active_experts_scores[2] >= active_experts_scores[3], "Scores in descending order");
}

void TestRoute::run_all_tests() {
    test_single_expert();
    test_top_k_selection();
    test_score_normalization();
    test_expert_ordering();
}