#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>  
#include <cmath> 

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
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {
            in_features_ = in_features;
            out_features_ = out_features;
        }

        void fwd() override {
            output_ = Tensor(input_.N, out_features_, 1, 1);

            // over images
            for (size_t n = 0; n < input_.N; ++n) {
                // over output nodes
                for (size_t out = 0; out < out_features_; ++out) {
                    // start with the bias
                    float curr = bias_(out);
                    // accumulate over input nodes
                    for (size_t in = 0; in < in_features_; ++in) {
                        curr += input_(n, in, 0, 0) * weights_(out, in, 0, 0);
                    }
                    output_(n, out, 0, 0) = curr;
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            // TODO

            // weights: (out, in, 1, 1))
        }

    private:
        size_t in_features_;
        size_t out_features_;

        bool check_input(const Tensor& input) override {
            return input.C == in_features_ && input.H == 1 && input.W == 1;
        }
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d) {
            kernel_size_ = kernel_size;
            stride_ = stride;
            pad_ = pad;
        }

        void fwd() override {
            if (input_.W + 2 * pad_ < kernel_size_ || input_.H + 2 * pad_ < kernel_size_)
                throw std::runtime_error("Kernel size is larger than input dimensions in Conv2d layer");

            size_t output_w = (input_.W + 2 * pad_ - kernel_size_) / stride_ + 1;
            size_t output_h = (input_.H + 2 * pad_ - kernel_size_) / stride_ + 1;
            output_ = Tensor(input_.N, input_.C, output_h, output_w);

            // over images
            for (size_t n = 0; n < input_.N; ++n) {
                // over channels
                for (size_t c = 0; c < input_.C; ++c) {
                    // calculate output feature map
                    for (size_t h = 0; h < output_h; ++h) {
                        for (size_t w = 0; w < output_w; ++w) {
                            // find max value in the kernel
                            float curr_max = -std::numeric_limits<float>::infinity();
                            for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                    // skip padding areas
                                    // if bellow 0
                                    if (h * stride_ + kh < pad_ || w * stride_ + kw < pad_) {
                                        continue;
                                    }
                                    size_t in_h = h * stride_ + kh - pad_;
                                    size_t in_w = w * stride_ + kw - pad_;
                                    // if above input dimensions
                                    if (in_h >= input_.H || in_w >= input_.W) {
                                        continue;
                                    }
                                    curr_max = std::max(curr_max, input_(n, c, in_h, in_w));
                                }
                            }
                        // if all vakues are in the padding area, set to 0
                        if (curr_max == -std::numeric_limits<float>::infinity()) {
                            curr_max = 0.0f;
                        }
                        output_(n, c, h, w) = curr_max;
                        }
                    }
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            return;
        }

    private:
        size_t kernel_size_;
        size_t stride_;
        size_t pad_;

        bool check_input(const Tensor& input) override {
            return true; 
        }
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {}
    
        void fwd() override {
            output_ = Tensor(input_.N, input_.C, input_.H, input_.W);
            // for each value
            for (size_t n = 0; n < input_.N; ++n) {
                for (size_t c = 0; c < input_.C; ++c) {
                    for (size_t h = 0; h < input_.H; ++h) {
                        for (size_t w = 0; w < input_.W; ++w) {
                            // 0 if negative, else keep value
                            output_(n, c, h, w) = std::max(0.0f, input_(n, c, h, w));
                        }
                    }
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            return;
        }

    private:
        bool check_input(const Tensor& input) override {
            return true; 
        }
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}

        void fwd() override {
            output_ = Tensor(input_.N, input_.C, input_.H, input_.W);
            // for each value
            for (size_t n = 0; n < input_.N; ++n) {
                float sum_exp = 0.;
                for (size_t c = 0; c < input_.C; ++c) {
                    for (size_t h = 0; h < input_.H; ++h) {
                        for (size_t w = 0; w < input_.W; ++w) {
                            float curr_exp = std::exp(input_(n, c, h, w));
                            sum_exp += curr_exp;
                            // store the exponentials of the inputs in their respective positions in output
                            output_(n, c, h, w) = curr_exp;
                        }
                    }
                }
                sum_exp = std::max(sum_exp, 1e-10f); // avoid division by zero
                // devide each exponential by the sum of exponentials to get probabilities
                for (size_t c = 0; c < input_.C; ++c) {
                    for (size_t h = 0; h < input_.H; ++h) {
                        for (size_t w = 0; w < input_.W; ++w) {
                            output_(n, c, h, w) /= sum_exp;
                        }
                    }
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            return;
        }

    private:
        bool check_input(const Tensor& input) override {
            return input.H == 1 && input.W == 1; // we expect a vector input
        }
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    
        void fwd() override {
            output_ = Tensor(input_.N, input_.C * input_.H * input_.W, 1, 1);
            // for each value
            for (size_t n = 0; n < input_.N; ++n) {
                for (size_t c = 0; c < input_.C; ++c) {
                    for (size_t h = 0; h < input_.H; ++h) {
                        for (size_t w = 0; w < input_.W; ++w) {
                            // copy value form the input array to the right position in the output array
                            output_(n, c * input_.H * input_.W + h * input_.W + w, 1, 1) = input_(n, c, h, w);
                        }
                    }
                }
            }
        }

        void read_weights_bias(std::ifstream& is) override {
            return;
        }
    private:
        bool check_input(const Tensor& input) override {
            return true; 
        }
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            layers_.push_back(std::unique_ptr<Layer>(layer));
        }

        void load(std::string file) {
            // not sure - TODO
            for (auto& layer : layers_) {
                std::ifstream stream(file, std::ios::binary);
                if (!stream.is_open()) {
                    throw std::runtime_error("Could not open file: " + file);
                }
                layer->read_weights_bias(stream);
            }
        }

        Tensor predict(Tensor input) {
            for (auto& layer : layers_) {
                layer->set_input(input);
                layer->fwd();
                if (debug_) {
                    layer->print();
                }
                input = layer->get_output();
            }
            return input;
        }

    private:
        bool debug_;
        // TODO: storage for layers
        std::vector<std::unique_ptr<Layer>> layers_;
};

#endif // NETWORK_HPP
