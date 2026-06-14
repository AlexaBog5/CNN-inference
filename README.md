# CNN Inference and Weight Quantization

Study project for the course **Systems Engineering and Architecting for Edge Computing**.

## Overview

This project implements inference for a Convolutional Neural Network (CNN) and explores memory-efficient deployment techniques for edge devices.

Implemented tasks:

* CNN forward inference
* Weight quantization using 8-bit integers

The network used in this project as an example is LeNet.

## Task 1 – CNN Inference

A complete forward pass of the network was implemented in C++.

Implemented layers:

* 2d Convolution
* Fully Connected Layers (Linear)
* 2d Max Pooling
* Activation functions (ReLU, SoftMax)


The implementation performs inference on pretrained network parameters and produces classification outputs.

## Task 2 – Weight Quantization

To reduce memory consumption, model weights were quantized from floating-point values to 8-bit integers.

### Quantization Scheme

For each layer:

* Weights are stored as `int8_t`
* One floating-point scale is stored
* One integer zero point is stored

Biases remain in floating-point format because:

* Their number is relatively small
* They have a stronger influence on numerical precision

### Memory Analysis

Storage per layer:

Before quantization:

```text
sizeof(float) × num_weights
= 32 × num_weights bits
```

After quantization:

```text
sizeof(int8_t) × num_weights
+ sizeof(int8_t)
+ sizeof(float)

= 8 × num_weights + 40 bits
```

Memory reduction:

```text
24 × num_weights − 40 bits
```

### LeNet Example

Number of weights:

```text
6 × 1 × 5 × 5
+ 16 × 6 × 5 × 5
+ 16 × 5 × 5 × 120
+ 120 × 84
+ 84 × 10

= 61,470 weights
```

Number of quantized layers:

```text
13
```

Total memory reduction:

```text
24 × 61,470 − 40 × 13

= 1,474,760 bits
≈ 180 KB
```

## Repository Structure

### Main Branch

Contains the CNN inference implementation.

### Quantization Branch

Contains the quantized version of the network using int8 weights.

## Technologies

* C++
* CMake
* CNN / LeNet
* Integer Quantization
* Edge AI Concepts
