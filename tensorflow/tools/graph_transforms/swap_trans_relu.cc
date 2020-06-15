/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status SwapTransRelu(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
  // Make sure we don't get rid of any nodes used as graph inputs or outputs.
  std::set<string> required_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(NodeNameFromInput(input));
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(NodeNameFromInput(output));
  }
  // LOG(INFO) << "Here 1";

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
        {"Relu",
          {
            {"Transpose",
              {
                {"*"},     // input_node
                {"Const"}  // permutation
              }
            },
          }
        },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore, &required_nodes](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& relu_node = match.node;
        LOG(INFO) << "Here 2" << match.node.name();
        const NodeDef& transpose_node = match.inputs[0].node;
        const NodeDef& perm_node = match.inputs[0].inputs[1].node;  // Const
        const NodeDef& input_node = match.inputs[0].inputs[0].node;

        // Check that nodes that we use are not used somewhere else.
        for (const auto& node :
             {relu_node, transpose_node, perm_node, input_node}) {
          if (required_nodes.count(node.name())) {
            // Return original nodes.
            LOG(INFO) << "Skipping replacement for " << node.name();
            CopyOriginalMatch(match, new_nodes);
            return Status::OK();
          }
        }

        // Construct the new nodes.
        NodeDef new_relu_node;
        new_relu_node = relu_node;
        new_relu_node.mutable_input()->at(0) = input_node.name();

        NodeDef new_transpose_node;
        new_transpose_node = transpose_node;
        new_transpose_node.mutable_input()->at(0) = new_relu_node.name();

        new_nodes->push_back(input_node);
        new_nodes->push_back(new_relu_node);
        new_nodes->push_back(perm_node);
        new_nodes->push_back(new_transpose_node);

        // Rename references appropriately
        string target_name = new_transpose_node.name();
        inputs_to_rename[relu_node.name()] = target_name;
        inputs_to_rename["^" + relu_node.name()] = "^" + target_name;
        nodes_to_ignore.insert(new_transpose_node.name());

        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Make sure all references to removed nodes now point to their inputs.
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}  // namespace graph_transforms

REGISTER_GRAPH_TRANSFORM("swap_trans_relu", SwapTransRelu);

}  // namespace graph_transforms
}  // namespace tensorflow
