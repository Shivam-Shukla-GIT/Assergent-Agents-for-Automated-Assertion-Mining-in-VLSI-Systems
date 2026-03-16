#!/usr/bin/env python3
"""
stratified_splitter.py - Stratified Train/Test Split for Good Generalization
Implements stratified sampling to:

Maintain proportional representation across complexity levels
Ensure diversity in both knowledge DB and test set
Prevent overfitting through balanced distribution

Usage:
python stratified_splitter.py --classification classification_report.json 
--input database/ 
--output-train knowledge_db/ 
--output-test test_set/ 
"""
import os
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class StratifiedSplitter:
    """Stratified sampling for train/test split"""
    def __init__(self, train_size: int = None, test_size: int = None, random_seed: int = 42):
        self.train_size = train_size
        self.test_size = test_size
        self.total_size = train_size + test_size if train_size is not None and test_size is not None else 0
        random.seed(random_seed)
        
        # Target distributions (proportional)
        self.target_distribution = {
            'simple': {'train': 0.40, 'test': 0.40},    # 40% each
            'medium': {'train': 0.40, 'test': 0.40},    # 40% each
            'complex': {'train': 0.20, 'test': 0.20}    # 20% each
        }

    def load_classification(self, classification_file: str) -> Dict:
        """Load classification report"""
        with open(classification_file, 'r') as f:
            return json.load(f)

    def stratified_split(self, metadata_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified split maintaining proportions across all dimensions
        while ensuring all files are used
        """
        # Group by overall complexity
        complexity_groups = {
            'simple': [],
            'medium': [],
            'complex': []
        }
        
        # Count total in each complexity group
        complexity_counts = {k: 0 for k in complexity_groups.keys()}
        
        for metadata in metadata_list:
            complexity = metadata['overall_complexity']['class']
            complexity_groups[complexity].append(metadata)
            complexity_counts[complexity] += 1
        
        # Calculate total available files
        total_available = sum(complexity_counts.values())
        
        # Adjust train_size and test_size if they weren't explicitly set
        if self.train_size is None or self.test_size is None:
            self.train_size = int(total_available * (2/3))
            self.test_size = total_available - self.train_size
        
        # Shuffle each group for randomization
        for group in complexity_groups.values():
            random.shuffle(group)
        
        # Calculate initial target counts for each complexity level based on ideal distribution
        train_targets = {}
        test_targets = {}
        allocated_train = 0
        allocated_test = 0
        
        # First pass: Allocate based on ideal distribution but limited by available files
        for complexity, count in complexity_counts.items():
            # Calculate ideal targets
            ideal_train = int(self.train_size * self.target_distribution[complexity]['train'])
            ideal_test = int(self.test_size * self.target_distribution[complexity]['test'])
            
            # If we need more than available, scale down proportionally
            if ideal_train + ideal_test > count:
                # Use all files in this category, distributed proportionally
                train_ratio = ideal_train / (ideal_train + ideal_test)
                train_targets[complexity] = int(count * train_ratio)
                test_targets[complexity] = count - train_targets[complexity]
            else:
                # Use the ideal targets
                train_targets[complexity] = ideal_train
                test_targets[complexity] = ideal_test
            
            allocated_train += train_targets[complexity]
            allocated_test += test_targets[complexity]
        
        # Second pass: Distribute any remaining files to maintain overall train/test ratio
        remaining_train = self.train_size - allocated_train
        remaining_test = self.test_size - allocated_test
        
        # Calculate how many files we can still take from each category
        available_remaining = {
            complexity: count - (train_targets[complexity] + test_targets[complexity])
            for complexity, count in complexity_counts.items()
        }
        
        # Distribute remaining based on how many are available in each category
        total_remaining_available = sum(available_remaining.values())
        if total_remaining_available > 0:
            # First add to train set
            for complexity, available in available_remaining.items():
                if remaining_train <= 0:
                    break
                # Take proportional share of remaining files
                share = min(
                    remaining_train, 
                    int(available * (remaining_train / total_remaining_available))
                )
                if share <= 0 and available > 0 and remaining_train > 0:
                    # Ensure at least 1 file is taken if available and needed
                    share = 1
                
                train_targets[complexity] += share
                available_remaining[complexity] -= share
                remaining_train -= share
            
            # If still have train files to allocate, take from categories with remaining
            while remaining_train > 0 and sum(available_remaining.values()) > 0:
                for complexity, available in available_remaining.items():
                    if available > 0 and remaining_train > 0:
                        train_targets[complexity] += 1
                        available_remaining[complexity] -= 1
                        remaining_train -= 1
            
            # Now add to test set with what's left
            for complexity, available in available_remaining.items():
                if available > 0:
                    added = min(available, remaining_test)
                    test_targets[complexity] += added
                    remaining_test -= added
        
        # Prepare the final sets
        train_set = []
        test_set = []
        
        # Split each complexity group according to the calculated targets
        for complexity, group in complexity_groups.items():
            train_count = train_targets[complexity]
            test_count = test_targets[complexity]
            
            # Add to train set
            train_set.extend(group[:train_count])
            
            # Add to test set
            test_set.extend(group[train_count:train_count + test_count])
            
            print(f"{complexity.upper():8s}: {len(group):3d} total ? "
                  f"{train_count:3d} train, {test_count:3d} test")
        
        # Verify all files were used
        total_used = len(train_set) + len(test_set)
        if total_used < total_available:
            print(f"Warning: Only used {total_used} out of {total_available} files!")
        
        return train_set, test_set

    def check_distribution_quality(self, train_set: List[Dict], test_set: List[Dict]):
        """
        Check if distribution maintains good coverage across dimensions
        """
        print("\n" + "="*70)
        print("DISTRIBUTION QUALITY CHECK")
        print("="*70)
        
        def count_by_dimension(dataset, dimension_key):
            counts = defaultdict(int)
            for item in dataset:
                if dimension_key == 'overall':
                    key = item['overall_complexity']['class']
                else:
                    key = item['classifications'][dimension_key]['class']
                counts[key] += 1
            return counts
        
        # Check each dimension
        dimensions = ['overall', 'temporal', 'signal', 'svf', 'design', 'semantic']
        
        for dim in dimensions:
            print(f"\n{dim.upper()} Distribution:")
            print(f"{'Category':<30s} {'Train':<10s} {'Test':<10s} {'Coverage':<10s}")
            print("-" * 70)
            
            train_counts = count_by_dimension(train_set, dim)
            test_counts = count_by_dimension(test_set, dim)
            
            all_categories = set(train_counts.keys()) | set(test_counts.keys())
            
            for category in sorted(all_categories):
                train_c = train_counts.get(category, 0)
                test_c = test_counts.get(category, 0)
                coverage = "? Both" if train_c > 0 and test_c > 0 else "? Missing"
                
                print(f"{category:<30s} {train_c:<10d} {test_c:<10d} {coverage:<10s}")
        
        print("="*70)

    def check_paraphrase_leakage(self, train_set: List[Dict], test_set: List[Dict]):
        """
        Check if similar specifications are split across train/test
        """
        print("\n" + "="*70)
        print("PARAPHRASE LEAKAGE CHECK")
        print("="*70)
        
        # Simple heuristic: check if specs share many words
        from collections import Counter
        
        def get_word_set(spec):
            # Normalize and get words
            return set(spec.lower().split())
        
        potential_leaks = []
        
        for train_item in train_set:
            train_words = get_word_set(train_item['english_spec'])
            
            for test_item in test_set:
                test_words = get_word_set(test_item['english_spec'])
                
                # Calculate Jaccard similarity
                intersection = len(train_words & test_words)
                union = len(train_words | test_words)
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity > 0.7:  # High similarity threshold
                        potential_leaks.append({
                            'train': train_item['spec_id'],
                            'test': test_item['spec_id'],
                            'similarity': similarity
                        })
        
        if potential_leaks:
            print(f"\n? Found {len(potential_leaks)} potential paraphrase leaks:")
            for leak in potential_leaks[:5]:  # Show first 5
                print(f"  Train: {leak['train']} <-> Test: {leak['test']} "
                      f"(similarity: {leak['similarity']:.2f})")
            if len(potential_leaks) > 5:
                print(f"  ... and {len(potential_leaks) - 5} more")
            print("\n? WARNING: Consider manual review to prevent data leakage!")
        else:
            print("? No significant paraphrase leakage detected")
        
        print("="*70)

    def copy_files(self, dataset: List[Dict], input_dir: str, output_dir: str):
        """Copy files to output directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for item in dataset:
            src_file = Path(input_dir) / item['file']
            dst_file = output_path / item['file']
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: Source file not found: {src_file}")

    def save_metadata(self, train_set: List[Dict], test_set: List[Dict], output_dir: str):
        """Save split metadata for reproducibility"""
        metadata = {
            'split_config': {
                'train_size': self.train_size,
                'test_size': self.test_size,
                'target_distribution': self.target_distribution
            },
            'train_set': [item['spec_id'] for item in train_set],
            'test_set': [item['spec_id'] for item in test_set],
            'train_metadata': train_set,
            'test_metadata': test_set
        }
        
        output_path = Path(output_dir) / 'split_metadata.json'
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n? Split metadata saved to: {output_path}")

    def run(self, classification_file: str, input_dir: str, 
            output_train: str, output_test: str):
        """
        Main execution: stratified split with quality checks
        """
        print("="*70)
        print("STRATIFIED TRAIN/TEST SPLIT")
        print("="*70)
        
        # Load classification
        print(f"\nLoading classification from: {classification_file}")
        classification = self.load_classification(classification_file)
        
        metadata_list = classification['detailed_metadata']
        total_files = len(metadata_list)
        print(f"Total assertions: {total_files}")
        
        # If train_size and test_size not specified, use defaults
        if self.train_size is None or self.test_size is None:
            # Default: 2/3 train, 1/3 test
            self.train_size = int(total_files * (2/3))
            self.test_size = total_files - self.train_size
        
        # Perform stratified split
        print(f"\nTarget: {self.train_size} train, {self.test_size} test")
        print("\nStratified split by complexity:")
        train_set, test_set = self.stratified_split(metadata_list)
        
        print(f"\nActual: {len(train_set)} train, {len(test_set)} test")
        
        # Quality checks
        self.check_distribution_quality(train_set, test_set)
        self.check_paraphrase_leakage(train_set, test_set)
        
        # Copy files
        print("\nCopying files...")
        self.copy_files(train_set, input_dir, output_train)
        self.copy_files(test_set, input_dir, output_test)
        
        print(f"? Train set: {len(train_set)} files ? {output_train}")
        print(f"? Test set: {len(test_set)} files ? {output_test}")
        
        # Save metadata
        self.save_metadata(train_set, test_set, Path(output_train).parent)
        
        print("\n" + "="*70)
        print("SPLIT COMPLETE")
        print("="*70)

def get_user_confirmation(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user"""
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default:
        prompt += " [Y/n] "
    else:
        prompt += " [y/N] "
    
    while True:
        choice = input(prompt).lower()
        if choice == '':
            return default
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

def main():
    parser = argparse.ArgumentParser(
        description='Stratified train/test split for good generalization'
    )
    parser.add_argument('--classification', required=True,
        help='Classification report JSON file')
    parser.add_argument('--input', required=True,
        help='Input directory with all assertion files')
    parser.add_argument('--output-train', default='knowledge_db/',
        help='Output directory for training set')
    parser.add_argument('--output-test', default='test_set/',
        help='Output directory for test set')
    parser.add_argument('--train-size', type=int, default=None,
        help='Number of examples for training (default: 2/3 of total)')
    parser.add_argument('--test-size', type=int, default=None,
        help='Number of examples for testing (default: 1/3 of total)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility')
    parser.add_argument('--interactive', action='store_true',
        help='Run in interactive mode with prompts')
    args = parser.parse_args()

    # Load classification to determine total file count
    print(f"\nLoading classification from: {args.classification}")
    try:
        with open(args.classification, 'r') as f:
            classification = json.load(f)
            
        if 'detailed_metadata' in classification:
            total_files = len(classification['detailed_metadata'])
        elif 'pairs' in classification:
            total_files = len(classification['pairs'])
        else:
            total_files = 0
            print("Warning: Could not determine total file count from classification file.")
    except Exception as e:
        print(f"Error loading classification file: {e}")
        total_files = 0
    
    # Calculate default split if not provided (2/3 train, 1/3 test)
    train_size = args.train_size
    test_size = args.test_size
    
    if train_size is None and test_size is None:
        if total_files > 0:
            train_size = int(total_files * (2/3))
            test_size = total_files - train_size
        else:
            train_size = 100  # Fallback to default if we can't determine total
            test_size = 50
    elif train_size is not None and test_size is None:
        if total_files > 0:
            test_size = total_files - train_size
        else:
            test_size = train_size // 2
    elif train_size is None and test_size is not None:
        if total_files > 0:
            train_size = total_files - test_size
        else:
            train_size = test_size * 2
    
    # Verify totals don't exceed available files
    if total_files > 0 and train_size + test_size > total_files:
        print(f"Warning: Requested train_size ({train_size}) + test_size ({test_size}) = {train_size + test_size}")
        print(f"         This exceeds total available files ({total_files})")
        if args.interactive:
            if not get_user_confirmation("Adjust sizes to fit available files?"):
                print("Aborting split.")
                return 1
        # Adjust sizes proportionally
        ratio = train_size / (train_size + test_size)
        train_size = int(total_files * ratio)
        test_size = total_files - train_size
        print(f"Adjusted to: train_size = {train_size}, test_size = {test_size}")
    
    # Interactive mode - allow user to adjust the split
    if args.interactive:
        print(f"\n{'-'*60}")
        print(f"TRAIN/TEST SPLIT CONFIGURATION")
        print(f"{'-'*60}")
        print(f"Total available files: {total_files}")
        
        print(f"\nCurrent split:")
        print(f"- Training set: {train_size} files ({train_size/total_files*100:.1f}% of total)")
        print(f"- Testing set:  {test_size} files ({test_size/total_files*100:.1f}% of total)")
        
        if not get_user_confirmation("\nUse this split?"):
            # Offer input options
            print("\nHow would you like to specify the split?")
            print("1. Enter training set size")
            print("2. Enter training set percentage")
            split_option = input("Select option [1]: ").strip()
            
            if not split_option or split_option == "1":
                # Custom split by size
                while True:
                    try:
                        train_size = int(input(f"\nEnter training set size (1-{total_files-1}): "))
                        if train_size <= 0:
                            print("Training set size must be positive.")
                            continue
                            
                        test_size = total_files - train_size
                        if test_size <= 0:
                            print(f"With {total_files} total files, training set cannot exceed {total_files-1}.")
                            continue
                            
                        print(f"\nCustom split:")
                        print(f"- Training set: {train_size} files ({train_size/total_files*100:.1f}%)")
                        print(f"- Testing set:  {test_size} files ({test_size/total_files*100:.1f}%)")
                        
                        if get_user_confirmation("\nConfirm this split?"):
                            break
                            
                    except ValueError:
                        print("Please enter a valid number.")
            else:
                # Custom split by percentage
                while True:
                    try:
                        train_pct = float(input("\nEnter training set percentage (1-99): "))
                        if train_pct <= 0 or train_pct >= 100:
                            print("Percentage must be between 1 and 99.")
                            continue
                        
                        train_size = int((train_pct/100) * total_files)
                        test_size = total_files - train_size
                        
                        # Ensure at least one file in each set
                        if train_size <= 0:
                            train_size = 1
                            test_size = total_files - 1
                        elif test_size <= 0:
                            test_size = 1
                            train_size = total_files - 1
                        
                        print(f"\nCustom split:")
                        print(f"- Training set: {train_size} files ({train_size/total_files*100:.1f}%)")
                        print(f"- Testing set:  {test_size} files ({test_size/total_files*100:.1f}%)")
                        
                        if get_user_confirmation("\nConfirm this split?"):
                            break
                            
                    except ValueError:
                        print("Please enter a valid percentage.")
    
    splitter = StratifiedSplitter(
        train_size=train_size,
        test_size=test_size,
        random_seed=args.seed
    )

    splitter.run(
        classification_file=args.classification,
        input_dir=args.input,
        output_train=args.output_train,
        output_test=args.output_test
    )

if __name__ == "__main__":
    main()
