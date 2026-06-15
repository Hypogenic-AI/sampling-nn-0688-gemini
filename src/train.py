import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import os
import json
import time
import argparse
from tqdm import tqdm
from .model import VGG

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / total, correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / total, correct / total

def calculate_ece(model, loader, device, n_bins=10):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Calculating ECE", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets)
            
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    confidences, predictions = torch.max(all_probs, dim=1)
    accuracies = predictions.eq(all_targets)
    
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def run_experiment(args):
    print(f"Starting experiment: {args['exp_name']}")
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = load_from_disk(args['dataset_path'])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    def train_transform_fn(examples):
        examples['pixel_values'] = [transform_train(image.convert("RGB")) for image in examples['img']]
        return examples

    def test_transform_fn(examples):
        examples['pixel_values'] = [transform_test(image.convert("RGB")) for image in examples['img']]
        return examples
    
    train_dataset = dataset['train'].with_transform(train_transform_fn)
    test_dataset = dataset['test'].with_transform(test_transform_fn)
    
    def collate_fn(batch):
        return (
            torch.stack([x['pixel_values'] for x in batch]),
            torch.tensor([x['label'] for x in batch])
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = VGG(args['vgg_name'], nclass=10, activation=args['activation'], tau=args['tau'], hard=args['hard']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0
    for epoch in range(args['epochs']):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args['epochs']} ({epoch_time:.1f}s): Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"results/{args['exp_name']}_best.pt")

    # Final evaluation
    model.load_state_dict(torch.load(f"results/{args['exp_name']}_best.pt"))
    final_ece = calculate_ece(model, test_loader, device)
    print(f"Final Best Test Acc: {best_acc:.4f}, ECE: {final_ece:.4f}")
    
    results = {
        'args': args,
        'history': history,
        'best_acc': best_acc,
        'final_ece': final_ece
    }
    
    with open(os.path.join('results', f"{args['exp_name']}.json"), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--vgg_name", type=str, default="VGG11")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default="datasets/cifar10")
    
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    run_experiment(args)
