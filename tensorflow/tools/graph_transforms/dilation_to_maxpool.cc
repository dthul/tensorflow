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
Status Dilation2DToMaxPool2D(const GraphDef& input_graph_def,
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

  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Dilation2D", {
        {"*"},
        {"*"}
      }},  // clang-format on
      [&required_nodes](const NodeMatch& match,
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
        // Warning: Here we just assume that we can replace the dilation by a
        // max pool. That is usually not true.
        const NodeDef& input_node = match.inputs[0].node;

        NodeDef maxpool_node;
        maxpool_node.set_op("MaxPool");
        maxpool_node.set_name(match.node.name());
        maxpool_node.set_device(match.node.device());
        // SetNodeAttr("dtype", DT_FLOAT, &maxpool_node);
        auto list =
            maxpool_node.mutable_attr()->operator[]("ksize").mutable_list();
        list->add_i(1);
        list->add_i(2);
        list->add_i(2);
        list->add_i(1);
        // SetNodeAttr("ksize", {2, 2}, &maxpool_node);
        CopyNodeAttr(match.node, "strides", "strides", &maxpool_node);
        CopyNodeAttr(match.node, "padding", "padding", &maxpool_node);
        // CopyNodeAttr(match.node, "data_format", "data_format",
        // &maxpool_node);
        if (match.node.attr().count("use_cudnn_on_gpu") > 0) {
          CopyNodeAttr(match.node, "use_cudnn_on_gpu", "use_cudnn_on_gpu",
                       &maxpool_node);  // ?
        }
        AddNodeInput(input_node.name(), &maxpool_node);

        new_nodes->push_back(input_node);
        new_nodes->push_back(maxpool_node);
        return Status::OK();
      },
      {true}, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("dilation2d_to_maxpool2d", Dilation2DToMaxPool2D);

}  // namespace graph_transforms
}  // namespace tensorflow
