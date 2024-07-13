use crate::{activation::Activation, matrix::Matrix};

pub struct Network {
    layers: Vec<usize>,
    data: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    /// creates a new Network using the layers, learning rate and activation function
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation: Activation) -> Self {
        for i in layers.iter() {
            assert_ne!(i, &0, "layer {} cannot have zero nodes.", i);
        }

        let (weights, biases) = layers.windows(2).map(|ele| (Matrix::random(ele[0], ele[1]), Matrix::random(1, ele[1]))).collect::<(Vec::<Matrix>, Vec::<Matrix>)>();
        Self {
            weights,
            biases,
            activation,
            learning_rate,
            data: Vec::with_capacity(layers.len()),
            layers,
        }
    }

    fn _new_from(layers: Vec<usize>, weights: Vec<Matrix>, biases: Vec<Matrix>, learning_rate: f64) -> Self {
        Self {
            weights,
            biases,
            activation: Activation::SIGMOID,
            learning_rate,
            data: Vec::with_capacity(layers.len()),
            layers,
        }
    }

    /// performs a forward pass for the Network
    pub fn feed_forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.layers[0], "invalid input. Input should be of length {}", self.layers[0]);
        self.data.clear();
        self.data.push(Matrix::from(input));

        (0..self.layers.len() - 1).fold(Matrix::from(input), |mut acc, i| {
            acc = acc.multiply(&self.weights[i]).add_matrix(&self.biases[i]).map(self.activation.function());
            self.data.push(acc.clone());
            acc
        }).into()
    }

    /// adjusts the weights of the Netwok based on the out from the forward pass and the expected target
    fn back_propagate(&mut self, output: Vec<f64>, target: &[f64]) {
        assert_eq!(&target.len(), self.layers.last().unwrap(), "target dimention mismatch.");
        assert_eq!(output.len(), target.len(), "dimention mismatch.");

        let output = Matrix::from(output);
        let target = Matrix::from(target);

        (0..self.layers.len() - 1).rev().fold((target.subtract_matrix(&output), output.map(self.activation.derivative())), |(error, mut gradients), i| {
            gradients = gradients.elementwise_multiply(&error).map(|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].add_matrix(&self.data[i].transpose().multiply(&gradients));
            self.biases[i] =  self.biases[i].add_matrix(&gradients);

            (error.multiply(&self.weights[i].transpose()), self.data[i].map(self.activation.derivative()))
        });
    }

    /// trains the model using the given inputs and the expected outputs
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, target: Vec<Vec<f64>>, epochs: u32) {
        (0..epochs).for_each(|i| {
            if epochs < 100 || (i + 1) % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i + 1, epochs);
            }

            inputs.iter().enumerate().for_each(|(i, input)| {
                let output = self.feed_forward(input);
                self.back_propagate(output, target[i].as_ref());
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{activation::Activation, matrix::Matrix};
    use super::Network;

    #[test]
    fn test_forward_pass() {
        let layers = vec!{ 2, 3, 2 };
        let weights = vec!{
            Matrix::new(2, 3, vec!{ 0.1, 0.7, 0.4, 0.3, 0.3, 0.8 }),
            Matrix::new(3, 2, vec!{ 0.8, 0.3, 0.3, 0.4, 0.7, 0.1 }),
        };
        let biases = vec!{
            Matrix::new(1, 3, vec!{ 2f64, 5f64, 6f64 }),
            Matrix::new(1, 2, vec!{ 6f64, 5f64 }),
        };
        let mut network = Network::_new_from(layers, weights, biases, 0f64);
        assert_eq!(vec!{ 0.9995886316919916, 0.9969766238609902 }, network.feed_forward(&[8f64, 8f64]));
    }

    #[test]
    #[should_panic]
    fn test_zero_layer_network() {
        let layers = vec!{ 3, 6, 0, 1 };
        let _ = Network::new(layers, 0.5, Activation::SIGMOID);
    }
}
