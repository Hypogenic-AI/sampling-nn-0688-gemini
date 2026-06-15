# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size.

## Dataset 1: CIFAR-10

### Overview
- **Source**: [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10)
- **Size**: 60,000 images (50,000 train, 10,000 test), 32x32 color.
- **Format**: HuggingFace Dataset
- **Task**: Image classification (10 classes)

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar10")
dataset.save_to_disk("datasets/cifar10")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cifar10")
```

### Sample Data
See `datasets/cifar10_sample.json` for metadata samples.
