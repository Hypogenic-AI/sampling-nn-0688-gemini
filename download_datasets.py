from datasets import load_dataset
import os
import json

os.makedirs("datasets", exist_ok=True)

print("Downloading CIFAR-10...")
dataset = load_dataset("uoft-cs/cifar10")
dataset.save_to_disk("datasets/cifar10")

train_sample = dataset['train'][:10]
train_sample_meta = {k: v for k, v in train_sample.items() if k != 'img'}
with open('datasets/cifar10_sample.json', 'w') as f:
    json.dump(train_sample_meta, f, indent=2)

print("CIFAR-10 downloaded and sampled.")
