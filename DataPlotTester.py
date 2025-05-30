import matplotlib.pyplot as plt
import csv
import os
from glob import glob
from collections import defaultdict

input_dir = 'datasets'
output_dir = 'datasetsVisualized'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file
for filepath in glob(os.path.join(input_dir, '*.csv')):
    data = defaultdict(lambda: {'x': [], 'y': []})

    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or row[0].startswith('#') or row[0].lower() == 'x':
                continue  # skip header or comments
            x, y, label = float(row[0]), float(row[1]), str(row[2])
            data[label]['x'].append(x)
            data[label]['y'].append(y)

    # Plotting
    plt.figure(figsize=(6, 6))
    for label, coords in data.items():
        plt.scatter(coords['x'], coords['y'], label=f'Class {label}', s=20)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{os.path.basename(filepath)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, os.path.basename(filepath).replace('.csv', '.png'))
    plt.savefig(output_path)
    plt.close()
