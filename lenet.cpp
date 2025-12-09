#define MNIST_PRE_PAD

#include "minicnn_task.hpp"
#include "student.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"


void addLayersLenet(NeuralNetwork& net) {
    // 1x32x32
    net.add(new Conv2d(1, 6, 5));      // Conv2d layer: 1 input channel, 6 output channels, 5x5 kernel
    // 6x28x28
    net.add(new ReLu());               // ReLu activation
    // 6x28x28
    net.add(new MaxPool2d(2, 2));     // MaxPool2d layer: 2x2 kernel, stride 2
    // 6x14x14
    net.add(new Conv2d(6, 16, 5));    // Conv2d layer: 6 input channels, 16 output channels, 5x5 kernel
    // 16x10x10
    net.add(new ReLu());               // ReLu activation
    // 16x10x10
    net.add(new MaxPool2d(2, 2));     // MaxPool2d layer: 2x2 kernel, stride 2
    // 16x5x5
    net.add(new Flatten());            // Flatten layer
    // 16x5x5
    net.add(new Linear(16 * 5 * 5, 120)); // Linear layer: input features 16*5*5, output features 120
    // 120
    net.add(new ReLu());               // ReLu activation
    // 120
    net.add(new Linear(120, 84));     // Linear layer: input features 120, output features 84
    // 84
    net.add(new ReLu());               // ReLu activation
    // 84
    net.add(new Linear(84, 10));      // Linear layer: input features 84, output features 10
    // 10
    net.add(new SoftMax());           // SoftMax activation
    // 10
}

int main() {
    // init neural network
    NeuralNetwork net(true);

    // add lenet layers
    addLayersLenet(net);

    // load weights and biases
    std::string weights_file = "data/data-mnist-lenet.raw";
    net.load(weights_file);

    // load data
    std::string mnist_path = "data/data-mnist-t10k-images-idx3-ubyte";
    MNIST mnist(mnist_path);

    // for each image
    for (size_t i = 0; i < mnist.N(); ++i) {
        Tensor img = mnist.at(i);

        // predict probabilities
        Tensor output = net.predict(img);

        // find the label with the highest probability
        auto* tensor_start = output.data();
        size_t predicted_label = std::distance(tensor_start, std::max_element(tensor_start, tensor_start + output.size()));
        
        // just print it for now
        std::cout << "Prediction for image " << i << ": " << predicted_label << std::endl;
        // TODO: save images + predictions ?
    }
}