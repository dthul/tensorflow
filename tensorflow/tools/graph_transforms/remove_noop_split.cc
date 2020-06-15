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

// Deletes split nodes which are noops.
Status RemoveNoopSplit(const GraphDef& input_graph_def,
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

  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Split", {{"*"}, {"*"}}},  // clang-format on
      [&inputs_to_rename, &required_nodes](const NodeMatch& match,
                                           const std::set<string>& input_nodes,
                                           const std::set<string>& output_nodes,
                                           std::vector<NodeDef>* new_nodes) {
        const NodeDef& replace_node = match.node;
        // If this node is needed in the inputs or outputs don't replace
        // it.
        if (required_nodes.count(replace_node.name())) {
          LOG(INFO) << "Skipping replacement for " << replace_node.name();
          CopyOriginalMatch(match, new_nodes);
          return Status::OK();
        }
        // Check if this split is a noop
        if (match.node.attr().count("num_split") > 0) {
          const AttrValue& attr = match.node.attr().at("num_split");
          if (attr.value_case() == AttrValue::ValueCase::kI && attr.i() == 1) {
            // This split is a noop, remove it
            const NodeDef& input_node = match.inputs[1].node;
            string target_name = input_node.name();
            for (const string& input : replace_node.input()) {
              if (!input.compare(0, target_name.size(), target_name)) {
                if (input.size() == target_name.size() ||
                    input[target_name.size()] == ':') {
                  target_name = input;
                  break;
                }
              }
            }
            inputs_to_rename[replace_node.name()] = target_name;
            inputs_to_rename["^" + replace_node.name()] = "^" + target_name;
            new_nodes->push_back(input_node);
            return Status::OK();
          }
        }
        // Not a noop, keep the original
        CopyOriginalMatch(match, new_nodes);
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Make sure all references to removed nodes now point to their inputs.
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("remove_noop_split", RemoveNoopSplit);

}  // namespace graph_transforms
}  // namespace tensorflow
