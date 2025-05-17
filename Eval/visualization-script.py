#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ERAS QA Model Evaluation Visualization Script
For converting evaluation result CSV files into various visualization charts
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import argparse
from datetime import datetime
import re
from matplotlib.font_manager import FontProperties
from collections import Counter

# Using standard English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

# 設定風格
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-pastel")

# 全局顏色配置
COLORS = {
    "primary": "#4e79a7",
    "secondary": "#f28e2b",
    "tertiary": "#59a14f",
    "quaternary": "#e15759",
    "category_colors": sns.color_palette("husl", 8)
}

def load_results(result_files: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Load one or more result files
    
    Args:
        result_files: List of result file paths
        
    Returns:
        Tuple[List[pd.DataFrame], List[str]]: List of dataframes and model names
    """
    dataframes = []
    model_names = []
    
    for file_path in result_files:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            # Load file
            df = pd.read_csv(file_path)
            dataframes.append(df)
            
            # Extract model name from filename
            file_name = os.path.basename(file_path)
            match = re.search(r'([^_]+)_(with_rag|no_rag)', file_name)
            if match:
                model_name = f"{match.group(1)} ({'RAG' if match.group(2) == 'with_rag' else 'No RAG'})"
            else:
                model_name = file_name.split('.')[0]
            model_names.append(model_name)
            
            print(f"Loaded {file_name} with {len(df)} result items")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return dataframes, model_names

def plot_metrics_overview(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Plot model metrics overview
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", 
               "rouge-1", "rouge-2", "rouge-l", "meteor"]
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    bar_width = 0.15
    x = np.arange(len(metrics))
    
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        # Calculate average metric values
        avg_metrics = [df[metric].mean() for metric in metrics]
        
        # Plot bars
        bars = ax.bar(x + i * bar_width, avg_metrics, bar_width, 
                      label=model_name, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
    
    # Add labels and legend
    ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Evaluation Metrics Overview', fontsize=16, fontweight='bold')
    ax.set_xticks(x + bar_width * (len(dataframes) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics overview chart saved to {output_dir}/metrics_overview.png")

def plot_response_time(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Plot response time boxplot
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 6))
    
    response_times = []
    for df in dataframes:
        response_times.append(df["response_time"].values)
    
    sns.boxplot(data=response_times, orient='h', palette=COLORS["category_colors"][:len(dataframes)])
    
    plt.xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Response Time Comparison', fontsize=16, fontweight='bold')
    plt.yticks(range(len(model_names)), model_names)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Response time chart saved to {output_dir}/response_time.png")

def plot_category_performance(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Plot performance comparison across different categories
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    # Check if all dataframes have category column
    if not all('category' in df.columns for df in dataframes):
        print("Warning: Some result files don't have 'category' column, skipping category performance analysis")
        return
    
    # Get all categories and convert them to ASCII if they contain non-ASCII characters
    all_categories = set()
    category_mapping = {}  # Original category -> ASCII version
    
    for df in dataframes:
        for cat in df['category'].unique():
            if cat not in category_mapping:
                # If category contains non-ASCII characters, create an ASCII version
                if any(ord(c) > 127 for c in str(cat)):
                    ascii_cat = f"category_{hash(str(cat)) % 1000}"
                else:
                    ascii_cat = str(cat)
                category_mapping[cat] = ascii_cat
                all_categories.add(ascii_cat)
    
    all_categories = sorted(list(all_categories))
    
    # Select metrics to analyze
    metrics = ["bleu-1", "rouge-l", "meteor"]
    
    for metric in metrics:
        plt.figure(figsize=(12, 7))
        
        # Calculate average metric for each category in each dataframe
        category_data = []
        
        for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
            cat_means = []
            for orig_cat, ascii_cat in category_mapping.items():
                category_df = df[df['category'] == orig_cat]
                if len(category_df) > 0 and ascii_cat in all_categories:
                    cat_means.append(category_df[metric].mean())
                else:
                    cat_means.append(0)
            category_data.append(cat_means)
        
        # Set up chart
        x = np.arange(len(all_categories))
        bar_width = 0.8 / len(dataframes)
        
        for i, (cat_means, model_name) in enumerate(zip(category_data, model_names)):
            offset = (i - len(dataframes) / 2 + 0.5) * bar_width
            plt.bar(x + offset, cat_means, width=bar_width, label=model_name, 
                   color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
        
        plt.xlabel('Category', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
        plt.title(f'{metric} Performance by Category', fontsize=16, fontweight='bold')
        plt.xticks(x, all_categories, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_{metric}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Category {metric} performance chart saved to {output_dir}/category_{metric}_comparison.png")

def plot_response_length_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Analyze response length distribution and relationship with metrics
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 6))
    
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        sns.histplot(df["response_length"], kde=True, 
                    color=COLORS["category_colors"][i % len(COLORS["category_colors"])], 
                    alpha=0.6, label=model_name)
    
    plt.xlabel('Response Length (characters)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Response Length Distribution', fontsize=16, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Response length distribution chart saved to {output_dir}/response_length_distribution.png")
    
    # Analyze relationship between response length and metrics
    metrics = ["bleu-1", "rouge-l", "meteor"]
    
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for j, metric in enumerate(metrics):
            axes[j].scatter(df["response_length"], df[metric], 
                          alpha=0.5, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
            
            # Plot regression line
            m, b = np.polyfit(df["response_length"], df[metric], 1)
            axes[j].plot(df["response_length"], m*df["response_length"] + b, 
                       color='red', linestyle='--', linewidth=1)
            
            axes[j].set_xlabel('Response Length', fontsize=10)
            axes[j].set_ylabel(metric, fontsize=10)
            axes[j].set_title(f'Response Length vs {metric}', fontsize=12)
            
            # Calculate correlation coefficient
            corr = df["response_length"].corr(df[metric])
            axes[j].annotate(f'Correlation: {corr:.3f}', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=9, ha='left', va='top')
        
        plt.suptitle(f'{model_name} - Response Length vs Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_')}_length_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Response length vs metrics chart saved to {output_dir}/{model_name.replace(' ', '_')}_length_metrics.png")

def plot_reference_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Analyze RAG reference usage
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    # Check if all dataframes have used_rag_resources column
    if not all('used_rag_resources' in df.columns for df in dataframes):
        print("Warning: Some result files don't have 'used_rag_resources' column, skipping RAG reference analysis")
        return
    
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if "with_rag" not in model_name.lower():
            continue
            
        # Count how many times each resource was referenced
        all_resources = []
        for resources in df["used_rag_resources"].dropna():
            if resources and isinstance(resources, str):
                # Handle potential non-ASCII characters in resource names
                cleaned_resources = []
                for r in resources.split(','):
                    r = r.strip()
                    # Replace non-ASCII resource names with a hash-based name
                    if any(ord(c) > 127 for c in r):
                        r = f"resource_{hash(r) % 1000}"
                    cleaned_resources.append(r)
                all_resources.extend(cleaned_resources)
        
        resource_counts = Counter(all_resources)
        
        # Create bar chart of top 10 most referenced resources
        plt.figure(figsize=(10, 6))
        
        top_resources = dict(resource_counts.most_common(10))
        plt.bar(top_resources.keys(), top_resources.values(), 
               color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
        
        plt.xlabel('Resource File', fontsize=12, fontweight='bold')
        plt.ylabel('Reference Count', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Top 10 Referenced Resources', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_')}_reference_counts.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Reference analysis chart saved to {output_dir}/{model_name.replace(' ', '_')}_reference_counts.png")

def plot_radar_chart(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Draw radar chart to compare model performance across metrics
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", 
               "rouge-1", "rouge-2", "rouge-l", "meteor"]
    
    # Calculate average values for each model on each metric
    avg_values = []
    for df in dataframes:
        avg_values.append([df[metric].mean() for metric in metrics])
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, (values, model_name) in enumerate(zip(avg_values, model_names)):
        values += values[:1]  # Close the values list
        ax.plot(angles, values, 'o-', linewidth=2, 
               color=COLORS["category_colors"][i % len(COLORS["category_colors"])], 
               label=model_name)
        ax.fill(angles, values, alpha=0.1, 
               color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
    
    # Set labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set y-axis limit to 1.2 times the maximum value among all metrics
    ax.set_ylim(0, max([max(v[:-1]) for v in avg_values]) * 1.2)
    
    plt.title('Model Evaluation Metrics Radar Chart', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics radar chart saved to {output_dir}/metrics_radar_chart.png")
    # 最大值的 1.2 倍，確保有足夠的空間
    ax.set_ylim(0, max([max(v[:-1]) for v in avg_values]) * 1.2)
    
    plt.title('模型評估指標雷達圖', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存指標雷達圖到 {output_dir}/metrics_radar_chart.png")

def plot_failure_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Analyze the worst performing questions to identify model weaknesses
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    combined_metrics = ["bleu-1", "rouge-l", "meteor"]
    
    # Create a failure analysis for each model
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        # Calculate combined score (average of metrics)
        df['combined_score'] = df[combined_metrics].mean(axis=1)
        
        # Get the 10 worst questions
        worst_questions = df.nsmallest(10, 'combined_score')
        
        # Analyze category distribution of worst questions if category column exists
        if 'category' in df.columns:
            # Transliterate category names to ASCII
            worst_questions = worst_questions.copy()
            if 'category' in worst_questions.columns:
                # Convert category names to ASCII-only by replacing CJK characters with romanized versions
                worst_questions['category_ascii'] = worst_questions['category'].apply(
                    lambda x: f"category_{hash(str(x)) % 1000}" if any(ord(c) > 127 for c in str(x)) else str(x)
                )
                
                # Create a display ID combining transliterated category and question ID
                worst_questions['display_id'] = worst_questions.apply(
                    lambda x: f"{x['category_ascii']}_{x['qa_id']}" if 'qa_id' in x else str(x.name), 
                    axis=1
                )
            
            plt.figure(figsize=(12, 7))
            
            # Plot bar chart with question IDs on x-axis and scores on y-axis
            sns.barplot(x='display_id', y='combined_score', data=worst_questions, 
                       palette='viridis', hue='category_ascii' if 'category_ascii' in worst_questions.columns else None)
            
            plt.xlabel('Question ID', fontsize=12, fontweight='bold')
            plt.ylabel('Combined Score', fontsize=12, fontweight='bold')
            plt.title(f'{model_name} - 10 Worst Performing Questions', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            if 'category_ascii' in worst_questions.columns:
                plt.legend(title='Category')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name.replace(' ', '_')}_worst_questions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Worst questions analysis chart saved to {output_dir}/{model_name.replace(' ', '_')}_worst_questions.png")
            
            # Analyze category distribution
            if 'category' in worst_questions.columns:
                # Use ASCII categories for pie chart
                category_counts = worst_questions['category_ascii'].value_counts()
                
                plt.figure(figsize=(8, 6))
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
                       colors=COLORS["category_colors"][:len(category_counts)])
                plt.axis('equal')
                plt.title(f'{model_name} - Category Distribution of Worst Questions', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{model_name.replace(' ', '_')}_worst_categories.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Worst question category distribution chart saved to {output_dir}/{model_name.replace(' ', '_')}_worst_categories.png")

def plot_scatter_metrics(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    Draw scatter plots showing relationship between two metrics (e.g., BLEU vs ROUGE-L)
    
    Args:
        dataframes: List of result dataframes
        model_names: List of model names
        output_dir: Output directory
    """
    # Select metrics pairs to compare
    metric_pairs = [("bleu-1", "rouge-l"), ("bleu-1", "meteor"), ("rouge-l", "meteor")]
    
    for metric_x, metric_y in metric_pairs:
        plt.figure(figsize=(10, 8))
        
        for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
            plt.scatter(df[metric_x], df[metric_y], 
                       alpha=0.6, label=model_name,
                       color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
        
        plt.xlabel(f'{metric_x} Score', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric_y} Score', fontsize=12, fontweight='bold')
        plt.title(f'{metric_x} vs {metric_y} Relationship', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_x}_vs_{metric_y}_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric_x} vs {metric_y} scatter plot saved to {output_dir}/{metric_x}_vs_{metric_y}_scatter.png")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ERAS QA Model Evaluation Visualization")
    parser.add_argument("--result_files", type=str, nargs='+', required=True,
                        help="Evaluation result CSV file paths (multiple files can be specified for comparison)")
    parser.add_argument("--output_base_dir", type=str, default="./Eval/visualizations",
                        help="Base output directory for visualization charts")
    parser.add_argument("--use_csv_name_folder", action="store_true", default=True,
                        help="Create subfolder with CSV name for each result file")
    
    args = parser.parse_args()
    
    # Ensure base output directory exists
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Load result files
    dataframes, model_names = load_results(args.result_files)
    
    if not dataframes:
        print("Error: No result files could be loaded")
        sys.exit(1)
    
    # Determine output directory
    if args.use_csv_name_folder and len(args.result_files) == 1:
        # Use CSV filename as folder name
        csv_filename = os.path.basename(args.result_files[0])
        csv_name = os.path.splitext(csv_filename)[0]  # Remove extension
        output_dir = os.path.join(args.output_base_dir, csv_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir} (based on CSV filename)")
    else:
        # For multiple files, use a timestamp-based folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_base_dir, f"comparison_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir} (comparison of multiple files)")
    
    print(f"\nStarting visualization generation for {len(dataframes)} models...")
    
    # Generate visualizations
    plot_metrics_overview(dataframes, model_names, output_dir)
    plot_response_time(dataframes, model_names, output_dir)
    plot_category_performance(dataframes, model_names, output_dir)
    plot_response_length_analysis(dataframes, model_names, output_dir)
    plot_reference_analysis(dataframes, model_names, output_dir)
    plot_radar_chart(dataframes, model_names, output_dir)
    plot_failure_analysis(dataframes, model_names, output_dir)
    plot_scatter_metrics(dataframes, model_names, output_dir)
    
    print(f"\nVisualization complete! All charts saved to {output_dir} directory")

if __name__ == "__main__":
    main()