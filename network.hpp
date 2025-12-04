#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}


class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        // TODO: additional required methods

        void set_input(const Tensor& input) {
            if (!check_input(input)) {
                throw std::runtime_error("Invalid input tensor shape");
            }
            input_ = input;
        }
        Tensor get_output() const {
            return output_;
        }

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;

        virtual bool check_input(const Tensor& input) = 0;
};


class Conv2d : public Layer {
    public:
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::Conv2d) {
            in_channels_ = in_channels;
            out_channels_ = out_channels;
            kernel_size_ = kernel_size;
            stride_ = stride;
            pad_ = pad;

            weights_ = Tensor(out_channels_, in_channels_, kernel_size_, kernel_size_);
            bias_ = Tensor(out_channels_);
        }
    
        void fwd() override {
            if (input_.C != in_channels_) {
                throw std::runtime_error("Invalid input channels for Conv2d layer");
            }
            if (input_.W + 2 * pad_ < kernel_size_ || input_.H + 2 * pad_ < kernel_size_)
                throw std::runtime_error("Kernel size is larger than input dimensions in Conv2d layer");

            size_t output_w = (input_.W + 2 * pad_ - kernel_size_) / stride_ + 1;
            size_t output_h = (input_.H + 2 * pad_ - kernel_size_) / stride_ + 1;
            output_ = Tensor(input_.N, out_channels_, output_h, output_w);

            // over images
            for (size_t n = 0; n < input_.N; ++n) {
                // over output channels (convolutions)
                for (size_t out = 0; out < out_channels_; ++out) {
                    // calculate output feature map
                    for (size_t h = 0; h < output_h; ++h) {
                        for (size_t w = 0; w < output_w; ++w) {
                            // start with the bias
                            float curr_conv = bias_(out);
                            // accumulate over input channels
                            for (size_t in = 0; in < in_channels_; ++in) {
                                // convolution
                                for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                    for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                        // assume zero-padding and skip these computations
                                        // if bellow 0
                                        if (h * stride_ + kh < pad_ || w * stride_ + kw < pad_)
                                            continue;
                                        size_t in_h = h * stride_ + kh - pad_;
                                        size_t in_w = w * stride_ + kw - pad_;
                                        // if above input dimensions
                                        if (in_h >= input_.H || in_w >= input_.W)
                                            continue;
                                        curr_conv += input_(n, in, in_h, in_w) 
                                                    * weights_(out, in, kh, kw);
                                    }
                                }
                            }
                            output_(n, out, h, w) = curr_conv;
                        }
                    }
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            // TODO
        }
    private:
        size_t in_channels_;
        size_t out_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t pad_;

        bool check_input(const Tensor& input) override {
            return input.C == in_channels_;
        }
};


class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {}
    // TODO
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d) {}
    // TODO
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {}
    // TODO
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    // TODO
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    // TODO
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            // TODO
        }

        void load(std::string file) {
            // TODO
        }

        Tensor predict(Tensor input) {
            // TODO
        }

    private:
        bool debug_;
        // TODO: storage for layers
};

#endif // NETWORK_HPP
