import json
import matplotlib.pyplot as plt
import glob
import os

def plot_results():
    files = glob.glob('results/*.json')
    plt.figure(figsize=(12, 6))
    
    for f in files:
        with open(f, 'r') as j:
            data = json.load(j)
            name = os.path.basename(f).replace('.json', '')
            if 'history' in data:
                acc = data['history']['test_acc']
                plt.plot(acc, label=name)
    
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/test_accuracy.png')
    print("Plot saved to figures/test_accuracy.png")

if __name__ == "__main__":
    plot_results()
