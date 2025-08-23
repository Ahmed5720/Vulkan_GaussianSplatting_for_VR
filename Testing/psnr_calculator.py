import os
import cv2
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

def calculate_psnr(gt_image, test_image):
    """
    Calculate PSNR between ground truth and test image
    """
    # Convert images to float32 for calculation
    gt = gt_image.astype(np.float32)
    test = test_image.astype(np.float32)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gt - test) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def parse_filename(filename):
    """
    Extract parameters from filename using regex
    Format: test_017_a0.00_s0.5_si2_fps157.png
    """
    pattern = r'test_(\d+)_a([\d.]+)_s([\d.]+)_si(\d+)_fps(\d+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        image_num = int(match.group(1))
        alpha = float(match.group(2))
        scale = float(match.group(3))
        sorting_interval = int(match.group(4))
        fps = int(match.group(5))
        
        return {
            'image_num': image_num,
            'alpha': alpha,
            'scale': scale,
            'sorting_interval': sorting_interval,
            'fps': fps
        }
    else:
        return None

def get_pareto_frontier(points):
    """
    Find the Pareto frontier points (maximize fps, maximize psnr)
    """
    # Sort points by fps (descending) and psnr (descending)
    sorted_points = sorted(points, key=lambda x: (-x['fps'], -x['psnr']))
    
    pareto_front = []
    max_psnr = -float('inf')
    
    for point in sorted_points:
        if point['psnr'] > max_psnr:
            pareto_front.append(point)
            max_psnr = point['psnr']
    
    # Sort by psnr for plotting
    pareto_front.sort(key=lambda x: x['psnr'])
    
    return pareto_front

def plot_results(results_df):
    """
    Create a plot with fps on y-axis, psnr on x-axis,
    with image number labels and Pareto frontier
    """
    if results_df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Convert DataFrame to list of points for Pareto calculation
    points = []
    for _, row in results_df.iterrows():
        points.append({
            'image_num': row['image_num'],
            'psnr': row['psnr'],
            'fps': row['fps']
        })
    
    # Get Pareto frontier
    pareto_front = get_pareto_frontier(points)
    
    # Plot all points with image number labels
    for _, row in results_df.iterrows():
        plt.scatter(row['psnr'], row['fps'], color='blue', alpha=0.6, s=50)
        plt.annotate(str(row['image_num']), 
                    (row['psnr'], row['fps']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)
    
    # Plot Pareto frontier
    if len(pareto_front) > 1:
        pareto_psnr = [point['psnr'] for point in pareto_front]
        pareto_fps = [point['fps'] for point in pareto_front]
        
        plt.plot(pareto_psnr, pareto_fps, 'r-', linewidth=2, label='Pareto Frontier')
        plt.scatter(pareto_psnr, pareto_fps, color='red', s=80, marker='o', label='Pareto Points')
    
    # Customize plot
    plt.xlabel('PSNR (dB)', fontsize=12)
    plt.ylabel('FPS', fontsize=12)
    plt.title('FPS vs PSNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some padding for better visualization
    plt.xlim(results_df['psnr'].min() - 1, results_df['psnr'].max() + 1)
    plt.ylim(results_df['fps'].min() - 10, results_df['fps'].max() + 10)
    
    plt.tight_layout()
    plt.savefig('fps_psnr_pareto.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as 'fps_psnr_pareto.png'")

def process_images():
    """
    Main function to process all images and generate CSV
    """
    # Define paths
    runs_folder = 'runs'
    gt_path = os.path.join(runs_folder, 'gt.png')
    
    # Load ground truth image
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth image not found at {gt_path}")
        return None
    
    gt_image = cv2.imread(gt_path)
    if gt_image is None:
        print(f"Error: Could not load ground truth image from {gt_path}")
        return None
    
    # Prepare data storage
    results = []
    
    # Process each image in the runs folder
    for filename in os.listdir(runs_folder):
        if filename == 'gt.png' or not filename.endswith('.png'):
            continue
        
        # Parse parameters from filename
        params = parse_filename(filename)
        if params is None:
            print(f"Warning: Could not parse filename {filename}, skipping...")
            continue
        
        # Load test image
        test_path = os.path.join(runs_folder, filename)
        test_image = cv2.imread(test_path)
        
        if test_image is None:
            print(f"Warning: Could not load image {filename}, skipping...")
            continue
        
        # Calculate PSNR
        psnr = calculate_psnr(gt_image, test_image)
        
        # Calculate score (fps / psnr)
        score = params['fps'] * psnr
        
        # Store results
        results.append({
            'image_num': params['image_num'],
            'alpha': params['alpha'],
            'scale': params['scale'],
            'sorting_interval': params['sorting_interval'],
            'fps': params['fps'],
            'psnr': psnr,
            'score': score
        })
        
        print(f"Processed {filename}: PSNR={psnr:.2f}, Score={score:.4f}")
    
    # Sort results by image number for better organization
    results.sort(key=lambda x: x['image_num'])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = 'psnr_results.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"\nResults saved to {csv_filename}")
    print(f"Processed {len(results)} images successfully")
    
    return df

if __name__ == "__main__":
    # Create the runs folder if it doesn't exist
    runs_folder = Path('runs')
    runs_folder.mkdir(exist_ok=True)
    
    # Process images and generate CSV
    results_df = process_images()
    
    # Display summary
    if results_df is not None and not results_df.empty:
        print("\nSummary:")
        print(f"Average PSNR: {results_df['psnr'].mean():.2f}")
        print(f"Average Score: {results_df['score'].mean():.4f}")
        print(f"Best PSNR: {results_df['psnr'].max():.2f}")
        print(f"Best Score: {results_df['score'].min():.4f} (lower is better)")
        
        # Create the plot
        plot_results(results_df)