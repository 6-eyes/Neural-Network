use std::f64::consts::E;

pub enum Activation {
    SIGMOID,
    RELU,
}

impl Activation {
    pub fn function(&self) -> fn(f64) -> f64 {
        match self {
            Activation::SIGMOID => |x| 1f64 / (1f64 + E.powf(-x)),
            Activation::RELU => |x| x.max(0f64),
        }
    }

    pub fn derivative(&self) -> fn(f64) -> f64 {
        match self {
            Activation::SIGMOID => |x| x * (1f64 - x),
            Activation::RELU => |x| match x {
                n if n > 0f64 => 1f64,
                _ => 0f64,
            },
        }
    }
}
