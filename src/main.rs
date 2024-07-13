use neural_network::{activation::Activation, network::Network};

fn main() {

    let inputs = vec!{
        vec!{0.0, 0.0},
        vec!{1.0, 0.0},
        vec!{0.0, 1.0},
        vec!{1.0, 1.0},
    };

    let outputs = vec!{
        vec!{1.0},
        vec!{0.0},
        vec!{0.0},
        vec!{1.0},
    };

    let mut network = Network::new(vec!{ 2, 3, 1 }, 0.5, Activation::SIGMOID);
    network.train(inputs, outputs, 100000);

    println!("0 and 0: {:?}", network.feed_forward(&[0.0, 0.0]));
    println!("0 and 1: {:?}", network.feed_forward(&[0.0, 1.0]));
    println!("1 and 0: {:?}", network.feed_forward(&[1.0, 0.0]));
    println!("1 and 1: {:?}", network.feed_forward(&[1.0, 1.0]));
}
