use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Cosine similarity between two equal-length float slices.
///
/// Returns a value in [0, 1] for ReLU-activated embeddings:
///   1.0 = identical direction (very similar images)
///   0.0 = perpendicular (completely unrelated)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// One row in the index: where the image lives and its embedding vector.
#[derive(Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub embedding: Vec<f32>,
}

/// The full in-memory index: a list of entries.
///
/// Serialized to / deserialized from a JSON file on disk.
#[derive(Serialize, Deserialize, Default)]
pub struct Index {
    pub entries: Vec<Entry>,
}

impl Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one image's embedding to the index.
    pub fn add(&mut self, path: String, embedding: Vec<f32>) {
        self.entries.push(Entry { path, embedding });
    }

    /// Write the index to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("Cannot create index file: {}", path.display()))?;
        serde_json::to_writer(file, self)
            .context("Failed to serialize index")?;
        Ok(())
    }

    /// Load a previously saved index from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Cannot open index file: {}", path.display()))?;
        let index = serde_json::from_reader(file)
            .context("Failed to deserialize index")?;
        Ok(index)
    }

    /// Return the top-`k` entries most similar to `query`, sorted by
    /// cosine similarity descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
        // Score every entry
        let mut scored: Vec<(&str, f32)> = self
            .entries
            .iter()
            .map(|e| (e.path.as_str(), cosine_similarity(query, &e.embedding)))
            .collect();

        // Sort highest similarity first.
        // partial_cmp handles NaN safely by treating it as equal (NaN shouldn't
        // appear here, but defensive code is good practice).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
