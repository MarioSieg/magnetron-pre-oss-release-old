#pragma once

#include <gtest/gtest.h>
#include <msml.h>

// Helper to compile DAG and execute
inline auto msml_tensor_evaluate(msml_tensor_t* root, msml_graph_eval_order_t order=MSML_GRAPH_EVAL_ORDER_FORWARD) -> msml_tensor_t* {
    auto* ctx = msml_tensor_get_ctx(root);
    auto* gra = msml_compute_graph_compile(ctx, root, order, nullptr);
    return msml_compute_graph_execute(gra);
}
