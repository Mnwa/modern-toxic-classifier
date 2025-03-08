mod classifier;
use crate::classifier::{Device, ModernBertForSequenceClassificationLabeled};

fn main() {
    let path = std::env::var("MODEL_PATH").unwrap();
    let model = ModernBertForSequenceClassificationLabeled::load(path, Device::Cpu).unwrap();

    let stdin = std::io::stdin();
    stdin.lines().for_each(|line| {
        let output = model.classify(line.into_iter().collect()).unwrap();
        println!("{:?}", output);
    });
}
