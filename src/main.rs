mod classifier;
mod model;

use crate::classifier::{Device, ModernBertForSequenceClassificationLabeled};

fn main() {
    let model = ModernBertForSequenceClassificationLabeled::load(
        "/Users/mnwa/python/negfilter/toxic2/",
        Device::Cpu,
    )
    .unwrap();

    let stdin = std::io::stdin();
    stdin.lines().for_each(|line| {
        let output = model.classify(line.into_iter().collect()).unwrap();
        println!("{:?}", output);
    });
}
