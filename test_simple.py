#!/usr/bin/env python3
"""
Simple test script for Roboflow dataset discovery without TensorFlow dependencies
"""

import os
import json
from pathlib import Path

def discover_roboflow_datasets(base_dir="datasets"):
    """
    Simple dataset discovery function without TensorFlow dependencies
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found!")
        return []
    
    datasets = []
    
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            dataset_info = analyze_dataset_structure(dataset_dir)
            if dataset_info:
                datasets.append(dataset_info)
    
    return datasets

def find_coco_annotations(directory):
    """
    Find COCO annotation files in a directory
    """
    patterns = [
        '_annotations.coco.json',
        'annotations.json',
        '_annotations.json'
    ]
    
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    
    # Check for any JSON file that might be annotations
    json_files = list(directory.glob('*.json'))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if it has COCO format keys
            required_keys = ['images', 'annotations', 'categories']
            if all(key in data for key in required_keys):
                return json_file
        except:
            continue
    
    return None

def analyze_dataset_structure(dataset_dir):
    """
    Analyze the structure of a dataset directory
    """
    dataset_info = {
        'name': dataset_dir.name,
        'path': str(dataset_dir),
        'structure_type': None,
        'splits': {},
        'classes': set(),
        'total_images': 0,
        'total_annotations': 0
    }
    
    # Check for train/valid/test structure
    has_splits = False
    possible_splits = ['train', 'valid', 'validation', 'val', 'test']
    
    for split_name in possible_splits:
        split_dir = dataset_dir / split_name
        if split_dir.exists() and split_dir.is_dir():
            annotations_file = find_coco_annotations(split_dir)
            if annotations_file:
                has_splits = True
                normalized_name = 'val' if split_name in ['valid', 'validation'] else split_name
                
                split_info = analyze_split(split_dir, annotations_file)
                if split_info:
                    dataset_info['splits'][normalized_name] = split_info
                    dataset_info['classes'].update(split_info['classes'])
                    dataset_info['total_images'] += split_info['num_images']
                    dataset_info['total_annotations'] += split_info['num_annotations']
    
    # If no splits, check root directory
    if not has_splits:
        annotations_file = find_coco_annotations(dataset_dir)
        if annotations_file:
            split_info = analyze_split(dataset_dir, annotations_file)
            if split_info:
                dataset_info['splits']['all'] = split_info
                dataset_info['classes'].update(split_info['classes'])
                dataset_info['total_images'] = split_info['num_images']
                dataset_info['total_annotations'] = split_info['num_annotations']
                has_splits = True
    
    if has_splits:
        dataset_info['structure_type'] = 'split' if len(dataset_info['splits']) > 1 else 'single'
        dataset_info['classes'] = sorted(list(dataset_info['classes']))
        return dataset_info
    
    return None

def analyze_split(split_dir, annotations_file):
    """
    Analyze a specific split
    """
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        split_info = {
            'images_dir': str(split_dir),
            'annotations_file': str(annotations_file),
            'num_images': len(annotations['images']),
            'num_annotations': len(annotations['annotations']),
            'classes': [cat['name'] for cat in annotations['categories']],
            'categories': annotations['categories']
        }
        
        return split_info
    except Exception as e:
        print(f"Error analyzing split {split_dir}: {e}")
        return None

def main():
    print("=== Testing Roboflow Dataset Discovery ===")
    
    datasets = discover_roboflow_datasets("datasets")
    
    if not datasets:
        print("❌ No Roboflow datasets found!")
        print("Make sure you have Roboflow datasets in the 'datasets/' directory")
        return
    
    print(f"✅ Found {len(datasets)} datasets!")
    
    all_classes = set()
    total_images = 0
    total_annotations = 0
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. Dataset: {dataset['name']}")
        print(f"   Path: {dataset['path']}")
        print(f"   Structure: {dataset['structure_type']}")
        print(f"   Splits: {list(dataset['splits'].keys())}")
        print(f"   Classes: {dataset['classes']}")
        print(f"   Images: {dataset['total_images']}")
        print(f"   Annotations: {dataset['total_annotations']}")
        
        all_classes.update(dataset['classes'])
        total_images += dataset['total_images']
        total_annotations += dataset['total_annotations']
    
    print(f"\n=== Summary ===")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total unique classes: {len(all_classes)}")
    print(f"All classes found: {sorted(list(all_classes))}")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    
    print("\n✅ Dataset structure analysis completed!")
    print("Your datasets are ready to be used with the training script.")
    print("\nTo train with these datasets, run:")
    print("  python3 trainining_v3.py --roboflow")

if __name__ == "__main__":
    main()