pub use candle_core::Device;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBertForSequenceClassification};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Debug)]
pub enum Error {
    TokenizerError(tokenizers::Error),
    CandleError(candle_core::Error),
    SerializationError(serde_json::Error),
    IOError(std::io::Error),
}

impl From<tokenizers::Error> for Error {
    fn from(err: tokenizers::Error) -> Self {
        Self::TokenizerError(err)
    }
}

impl From<candle_core::Error> for Error {
    fn from(err: candle_core::Error) -> Self {
        Self::CandleError(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::IOError(err)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::TokenizerError(e) => write!(f, "tokenizer error: {}", e),
            Self::CandleError(e) => write!(f, "candle error: {}", e),
            Self::SerializationError(e) => write!(f, "serialization error: {}", e),
            Self::IOError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone)]
pub struct ModernBertForSequenceClassificationLabeled {
    model: ModernBertForSequenceClassification,
    id_to_label: HashMap<String, String>,
    tokenizer: Tokenizer,
    device: Device,
}

impl ModernBertForSequenceClassificationLabeled {
    pub fn load<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
        let mut dir = PathBuf::from(path.as_ref());

        dir.push("config.json");
        let config_reader = File::open(&dir)?;
        let config: Config = serde_json::from_reader(config_reader)?;
        dir.pop();

        dir.push("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&dir)?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                pad_id: config.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)?;
        dir.pop();

        dir.push("model.safetensors");
        let model_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&dir], DType::F32, &device) }?;
        dir.pop();

        let model = ModernBertForSequenceClassification::load(model_builder, &config)
            .map_err(Error::CandleError)?;
        let id_to_label = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.id2label.clone())
            .unwrap_or_default();
        Ok(Self {
            model,
            id_to_label,
            tokenizer,
            device,
        })
    }

    pub fn classify(&self, prompts: Vec<String>) -> Result<Vec<ClassifyResponse>> {
        let encoders = self.tokenizer.encode_batch(prompts, true)?;
        let mut inputs_ids = Vec::with_capacity(encoders.len());
        let mut attentions_mask = Vec::with_capacity(encoders.len());
        for encoder in encoders {
            inputs_ids.push(Tensor::new(encoder.get_ids(), &self.device)?);
            attentions_mask.push(Tensor::new(encoder.get_attention_mask(), &self.device)?);
        }
        let xs = self.model.forward(
            &Tensor::stack(inputs_ids.as_slice(), 0)?,
            &Tensor::stack(attentions_mask.as_slice(), 0)?,
        )?;
        let results: Vec<Vec<f32>> = xs.to_vec2()?;
        Ok(results
            .into_iter()
            .map(|responses| {
                let (i, score) = responses
                    .into_iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();
                ClassifyResponse {
                    label: self.id_to_label.get(&i.to_string()).cloned().unwrap(),
                    score,
                }
            })
            .collect())
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ClassifyResponse {
    label: String,
    score: f32,
}
