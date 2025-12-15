#define MNIST_PRE_PAD

#include "minicnn_task.hpp"
#include "student.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"
#include <filesystem>

void saveImagePGM(const std::string& filename,
                  Tensor& img)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file");

    // PGM header (P5 = binary)
    file << "P5\n" << img.W << " " << img.H << "\n255\n";

    auto* ptr = img.data();
    for (size_t i = 0; i < img.size(); ++i) {
        file << static_cast<uint8_t>(std::min(std::max(*ptr * 255.f, 0.f), 255.f));
        ptr++;
    }

    file.close();
}

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
    // set pathes that are defined in CMakeLists.txt
    std::filesystem::path baseDir = PROJECT_SOURCE_DIR;
    std::string weights_file = (baseDir / WEIGHTS_LOCAL_PATH).generic_string();
    std::string mnist_path = (baseDir / MNIST_LOCAL_PATH).generic_string();
    std::string output_path = (baseDir / OUTPUT_LOCAL_PATH).generic_string();
    if (TO_SAVE_IMAGES)
        std::filesystem::create_directories(output_path);

    // init neural network
    NeuralNetwork net(TO_DEBUG);

    // add lenet layers
    addLayersLenet(net);

    // load weights and biases
    net.load(weights_file);

    // load data
    MNIST mnist(mnist_path);

    // for each image in range NUM_IMAGES that is set in CMakeLists.txt
    for (size_t i = 0; i < NUM_IMAGES; ++i) {
        if (i >= mnist.N()) {
            break;
        }

        Tensor img = mnist.at(i);

        // predict probabilities
        Tensor output = net.predict(img);

        // find the label with the highest probability
        auto* tensor_start = output.data();
        size_t predicted_label = std::distance(tensor_start, std::max_element(tensor_start, tensor_start + output.size()));
        
        // print the result
        std::cout << "Prediction for image " << i << ": " << predicted_label << std::endl;

        // save the image with the predicted label in the filename
        // as pgm file (it is chosen to avoid dependencies on image libraries)
        // if TO_SAVE_IMAGES is set to true in CMakeLists.txt
        if (TO_SAVE_IMAGES){
            saveImagePGM(output_path + "mnist_image_" + std::to_string(i) + "_pred_" + std::to_string(predicted_label) + ".pgm", img);
        }
    }
}