# lensme

Visual image search engine written in Rust. Give it a folder of images and a query image — it returns the most visually similar matches using a pretrained ResNet-18 encoder running on your GPU.

## How it works

1. Every image in the folder is passed through a pretrained ResNet-18 (ImageNet weights) with the classification head removed. The result is a 512-dimensional **embedding vector** — a compact numerical fingerprint of what the image contains.
2. Embeddings are saved to an index file on disk.
3. At query time, the query image gets the same treatment, and the index is searched for the nearest neighbors by cosine similarity.

```
Image folder
    │
    ▼
ResNet-18 encoder (pretrained, GPU-accelerated via Metal on Apple Silicon)
    │
    ▼
512-dim embedding vector per image
    │
    ▼
Index file  ──►  cosine similarity search  ──►  top-K nearest neighbors
```

## Requirements

- Rust 1.75+
- macOS with Apple Silicon (M1/M2/M3/M4) — the WGPU backend targets Metal automatically

## Build

```bash
git clone https://github.com/ramikeller/lensme
cd lensme
cargo build --release
```

The first build takes a few minutes — it compiles the burn ML framework and generates the burn model from the bundled ONNX file.

## Usage

**Index a folder of images:**

```bash
lensme index ./photos
```

Walks `./photos` recursively, embeds every `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, or `.tiff` image, and writes the index to `lensme.index`.

Options:
```
--output <PATH>   Where to write the index file  [default: lensme.index]
```

**Query by image:**

```bash
lensme query --image query.jpg --k 5
```

Returns the top 5 most visually similar images from the index with their cosine similarity scores.

Options:
```
--image <PATH>   Query image
--k <N>          Number of results to return  [default: 5]
--index <PATH>   Index file to search         [default: lensme.index]
```

## Tech stack

| Component | Library |
|---|---|
| ML framework | [burn](https://github.com/tracel-ai/burn) 0.21 |
| GPU backend | burn-wgpu → Metal (Apple Silicon) |
| Vision encoder | ResNet-18, pretrained on ImageNet |
| Model import | burn-import (ONNX → burn at build time) |
| Image loading | [image](https://github.com/image-rs/image) |
| CLI | [clap](https://github.com/clap-rs/clap) |

## Project structure

```
build.rs                     — converts ONNX to burn model at compile time
models/resnet18_encoder.onnx — pretrained ResNet-18 without classification head
src/
  main.rs                    — CLI entry point
  preprocess.rs              — image loading, resize, ImageNet normalization
  model.rs                   — includes the generated burn model
```
