import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='mnist')
    parser.add_argument('--local_steps', nargs='+', type=int)
    return parser.parse_args()

def load_tensorboard_data(path):
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        
        # Try to get test accuracy data
        try:
            test_accuracy = [(s.step, s.value) for s in event_acc.Scalars('Test/Metric')]
        except KeyError:
            print(f"Warning: Test/Metric not found in {path}")
            return None, None
            
        steps, values = zip(*test_accuracy)
        return np.array(steps), np.array(values)
    except Exception as e:
        print(f"Error loading data from {path}: {str(e)}")
        return None, None

def main():
    args = parse_args()
    
    plt.figure(figsize=(10, 6))
    
    has_valid_data = False
    for local_steps in args.local_steps:
        log_dir = f"logs/mnist_local_steps_{local_steps}/global"
        steps, accuracies = load_tensorboard_data(log_dir)
        
        if steps is not None and accuracies is not None:
            plt.plot(steps, accuracies, label=f'Local Steps = {local_steps}')
            has_valid_data = True
    
    if not has_valid_data:
        print("No valid data found to plot")
        return
        
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Local Steps on Test Accuracy (Non-IID)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/local_steps_comparison.png')
    plt.close()
    
    # Print final accuracies
    # Thực hiện in ra định dạng CSV
    print("local_steps,final_test_accuracy")
    for local_steps in args.local_steps:
        log_dir = f"logs/mnist_local_steps_{local_steps}/global"
        steps, accuracies = load_tensorboard_data(log_dir)
        if accuracies is not None:
            print(f"{local_steps},{accuracies[-1]:.4f}")

if __name__ == "__main__":
    main() 