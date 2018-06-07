#First step save graph.pb by running in inference mode, with is_training as a boolean variable set to False not placeholder

#Second Run freeze graph to convert all variables to constant operations
python -m tensorflow.python.tools.freeze_graph --input_graph graph.pb --input_checkpoint experiments/unet_resnet18/checkpoints/best/-69936 --output_graph graph_frozen_resnet18.pb --output_node_names=network/output/ArgMax

#Third call optimize_for_inference for batchnorm folding and merging operations
python -m tensorflow.python.tools.optimize_for_inference --input graph_frozen_resnet18.pb --output graph_optimized_resnet18.pb --input_names=network/input/Placeholder --output_names=network/output/ArgMax
