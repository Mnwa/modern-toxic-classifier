use candle_transformers::models::modernbert::Config as ModernBertConfig;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Config {
    #[serde(flatten)]
    pub modernbert_config: ModernBertConfig,
    #[serde(flatten)]
    pub classifier_config: ClassifierConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    CLS,
    MEAN,
    UNKNOWN,
}

impl Default for ClassifierPooling {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    #[serde(default)]
    pub classifier_pooling: ClassifierPooling,
}
