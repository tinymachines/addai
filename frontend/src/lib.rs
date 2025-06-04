mod utils;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EEGPowerBands {
    pub delta: u32,
    pub theta: u32,
    pub low_alpha: u32,
    pub high_alpha: u32,
    pub low_beta: u32,
    pub high_beta: u32,
    pub low_gamma: u32,
    pub mid_gamma: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EEGData {
    pub timestamp: String,
    pub signal_quality: u8,
    pub attention: u8,
    pub meditation: u8,
    pub eeg_power: Option<EEGPowerBands>,
}

#[wasm_bindgen]
pub struct DataProcessor {
    history: Vec<EEGData>,
    max_history: usize,
}

#[wasm_bindgen]
impl DataProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(max_history: usize) -> DataProcessor {
        utils::set_panic_hook();
        
        DataProcessor {
            history: Vec::new(),
            max_history,
        }
    }

    #[wasm_bindgen]
    pub fn add_data(&mut self, data: &JsValue) -> Result<(), JsValue> {
        let eeg_data: EEGData = serde_wasm_bindgen::from_value(data.clone())?;
        
        self.history.push(eeg_data);
        
        // Keep only the most recent data points
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        
        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_latest(&self) -> Result<JsValue, JsValue> {
        match self.history.last() {
            Some(data) => Ok(serde_wasm_bindgen::to_value(data)?),
            None => Ok(JsValue::NULL),
        }
    }

    #[wasm_bindgen]
    pub fn get_attention_history(&self, points: usize) -> Vec<u8> {
        self.history
            .iter()
            .rev()
            .take(points)
            .map(|d| d.attention)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    #[wasm_bindgen]
    pub fn get_meditation_history(&self, points: usize) -> Vec<u8> {
        self.history
            .iter()
            .rev()
            .take(points)
            .map(|d| d.meditation)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    #[wasm_bindgen]
    pub fn calculate_attention_average(&self, points: usize) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        
        let recent: Vec<u8> = self.history
            .iter()
            .rev()
            .take(points)
            .map(|d| d.attention)
            .collect();
        
        if recent.is_empty() {
            0.0
        } else {
            recent.iter().map(|&x| x as f64).sum::<f64>() / recent.len() as f64
        }
    }

    #[wasm_bindgen]
    pub fn calculate_meditation_average(&self, points: usize) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        
        let recent: Vec<u8> = self.history
            .iter()
            .rev()
            .take(points)
            .map(|d| d.meditation)
            .collect();
        
        if recent.is_empty() {
            0.0
        } else {
            recent.iter().map(|&x| x as f64).sum::<f64>() / recent.len() as f64
        }
    }

    #[wasm_bindgen]
    pub fn get_signal_quality(&self) -> u8 {
        self.history
            .last()
            .map(|d| d.signal_quality)
            .unwrap_or(200)
    }

    #[wasm_bindgen]
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    #[wasm_bindgen]
    pub fn get_history_size(&self) -> usize {
        self.history.len()
    }
}