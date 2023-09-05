# Image classification

Implementation of a quantum convolutional neural network based on the [data re-uploading scheme](https://arxiv.org/abs/1907.02085). The algorithm has been adapted to work as a tensorflow conv2d-like layer

## Contents

* **utils.py**: the basic blocks of the circuits
* **reuploading_circuit.py**: the re-uploading algorithm implemented as a dense layer
* **quantum_conv.py**: the adaptation of the algorithm as a convolutional layer
* **demo_qconv.ipynb**: a simple demonstration of the functioning 
