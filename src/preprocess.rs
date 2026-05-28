use anyhow::{Context, Result};
use burn::prelude::*;
use image::{DynamicImage, imageops::FilterType};
use std::path::Path;

pub const IMAGE_SIZE: usize = 224;

// Per-channel mean and std computed over the ImageNet training set.
// Every ResNet model trained on ImageNet expects input normalized by these values.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Load an image from disk and return a normalized tensor ready for the model.
///
/// Output shape: [1, 3, 224, 224]  (batch=1, channels=3, height, width)
pub fn load_image<B: Backend>(path: &Path, device: &B::Device) -> Result<Tensor<B, 4>> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?;
    Ok(preprocess(&img, device))
}

/// Convert a DynamicImage into a normalized burn Tensor of shape [1, 3, H, W].
pub fn preprocess<B: Backend>(img: &DynamicImage, device: &B::Device) -> Tensor<B, 4> {
    let size = IMAGE_SIZE as u32;

    // Step 1 – resize to the expected square input size.
    // Lanczos3 is a high-quality resampling filter (slower than Nearest but
    // produces much sharper results, which matters for a feature extractor).
    let img = img.resize_exact(size, size, FilterType::Lanczos3);

    // Step 2 – convert to 8-bit RGB (drops alpha channel if present, handles
    // grayscale by replicating the single channel to R, G, and B).
    let rgb = img.to_rgb8();

    // Step 3 – build a flat Vec<f32> in CHW order with normalization applied.
    //
    // The raw pixel buffer is HWC: [R, G, B, R, G, B, ...] row by row.
    // We need CHW: all R values first (channel 0), then all G (channel 1),
    // then all B (channel 2).
    //
    // At the same time we apply ImageNet normalization:
    //     normalized = (pixel_0_to_1 - mean) / std
    let h = IMAGE_SIZE;
    let w = IMAGE_SIZE;
    let mut chw = vec![0.0f32; 3 * h * w];

    for row in 0..h {
        for col in 0..w {
            // `get_pixel` returns a Rgba<u8>-style struct; to_rgb8 means the
            // underlying buffer is already RGB so this is just an index into it.
            let pixel = rgb.get_pixel(col as u32, row as u32);
            for c in 0..3 {
                let value_0_1 = pixel.0[c] as f32 / 255.0;
                let normalized = (value_0_1 - MEAN[c]) / STD[c];
                // CHW index: channel stride is h*w, row stride is w
                chw[c * h * w + row * w + col] = normalized;
            }
        }
    }

    // Step 4 – wrap the flat Vec into a burn TensorData, then into a Tensor.
    // TensorData holds raw bytes + shape + dtype metadata; from_data moves it
    // onto the device (GPU memory if using the WGPU backend).
    let data = TensorData::new(chw, [1, 3, h, w]);
    Tensor::<B, 4>::from_data(data, device)
}
