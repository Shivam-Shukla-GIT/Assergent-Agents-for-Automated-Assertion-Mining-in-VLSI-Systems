#!/usr/bin/env python3
"""
prepare_database.py - Enhanced Database Generator with Classification Integration

Supports multiple input formats:
1. Standard format (parse_database_file):
   specs: "specification text"
   assertion: "assertion code"
   RTL: "rtl code"

2. Stratified format (parse_assertion_file):
   english_spec

   sva_assertion

   design_context

   signals (comma-separated)

Usage:
python prepare_database.py [--input INPUT_DIR] [--output OUTPUT_FILE] [--classification CLASSIFICATION_FILE] [--format FORMAT]
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

def parse_database_file(filepath: str) -> Dict:
    """
    Parse a single database file in original format
    
    Expected format:
        specs: "specification text"
        assertion: "assertion code"
        RTL: "rtl code"
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract sections
    specs_match = re.search(r'specs:\s*["\']?(.*?)["\']?\s*(?:assertion:|$)', content, re.DOTALL)
    assertion_match = re.search(r'assertion:\s*["\']?(.*?)["\']?\s*(?:RTL:|$)', content, re.DOTALL)
    rtl_match = re.search(r'RTL:\s*["\']?(.*?)["\']?\s*$', content, re.DOTALL)
    
    if not specs_match or not assertion_match:
        print(f"Warning: Could not parse {filepath}")
        return None
    
    spec = specs_match.group(1).strip().strip('"').strip("'")
    assertion = assertion_match.group(1).strip().strip('"').strip("'")
    rtl = rtl_match.group(1).strip().strip('"').strip("'") if rtl_match else ""
    
    return {
        'spec': spec,
        'assertion': assertion,
        'rtl': rtl
    }

def parse_assertion_file(filepath: str) -> Dict:
    """
    Parse a single assertion file in stratified format
    
    Expected format:
        english_spec

        sva_assertion

        design_context

        signals (comma-separated)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse file content
    sections = content.strip().split('\n\n')
    
    if len(sections) < 2:
        print(f"Warning: Skipping malformed file {filepath}")
        return None
    
    english_spec = sections[0] if len(sections) > 0 else ""
    sva_assertion = sections[1] if len(sections) > 1 else ""
    design_context = sections[2] if len(sections) > 2 else ""
    signals_str = sections[3] if len(sections) > 3 else ""
    
    # Extract signals
    signals = [s.strip() for s in signals_str.split(',') if s.strip()]
    
    return {
        'english_spec': english_spec,
        'sva_assertion': sva_assertion,
        'design_context': design_context,
        'signals': signals
    }

def extract_signals_from_assertion(assertion: str) -> List[str]:
    """Extract signal names from assertion"""
    # Remove SVA keywords and operators
    cleaned = re.sub(r'[@(){}|;]', ' ', assertion)
    cleaned = re.sub(r'\b(property|assert|posedge|negedge|disable|iff|past|rose|fell|stable|endproperty)\b', ' ', cleaned)
    
    # Find potential signal names
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned)
    
    # Filter out common values and keywords
    keywords = {'clk', 'rst_n', 'rst', 'clock', 'reset', 'b', 'd', 'h'}
    signals = [w for w in words if w not in keywords and len(w) > 1]
    
    return list(set(signals))

def infer_design_context(spec: str, assertion: str) -> str:
    """Infer design context from specification and assertion"""
    text = (spec + " " + assertion).lower()
    
    contexts = {
        'multiplexer': ['mux', 'select', 'sel', 'input 0', 'input 1'],
        'counter': ['count', 'increment', 'decrement'],
        'state_machine': ['state', 'idle', 'active', 'transition'],
        'arbiter': ['grant', 'request', 'arbitration', 'mutex'],
        'processor': ['hazard', 'register', 'instruction', 'pipeline'],
        'fifo': ['fifo', 'full', 'empty', 'write', 'read'],
        'memory': ['memory', 'address', 'data', 'read', 'write']
    }
    
    for context, keywords in contexts.items():
        if any(kw in text for kw in keywords):
            return context
    
    return 'general'

def classify_temporal_complexity(sva_code: str) -> Tuple[str, int]:
    """
    Classify temporal complexity based on SVA operators
    Returns: (level_name, score_0_100)
    """
    sva_lower = sva_code.lower()
    
    # Level 4: Complex sequences
    if ('throughout' in sva_lower or 'within' in sva_lower or 
        sva_code.count('##') > 2 or '[*' in sva_lower):
        return ('complex_sequence', 90)
    
    # Level 3: Multi-cycle delays
    if '##' in sva_code and re.search(r'##\s*[2-9]', sva_code):
        return ('multi_cycle', 70)
    
    # Level 2: Next-cycle implication
    if '|=>' in sva_code:
        return ('next_cycle', 50)
    
    # Level 1: Same-cycle implication
    if '|->' in sva_code:
        return ('same_cycle', 30)
    
    # Level 0: Combinational
    return ('combinational', 10)

def classify_signal_complexity(sva_code: str, signals: List[str]) -> Tuple[str, int]:
    """
    Classify based on number of signals and logical complexity
    Returns: (category, score_0_100)
    """
    signal_count = len(signals)
    
    # Count logical operators
    and_count = sva_code.count('&&') + sva_code.count('&')
    or_count = sva_code.count('||') + sva_code.count('|')
    not_count = sva_code.count('!')
    
    logical_complexity = and_count + or_count + not_count
    
    if signal_count >= 7 or logical_complexity > 10:
        return ('complex_multi_signal', 90)
    elif signal_count >= 4 or logical_complexity > 5:
        return ('multi_signal_conjunction', 60)
    elif signal_count >= 2:
        return ('dual_signal', 30)
    else:
        return ('single_signal', 10)

def classify_sampled_value_functions(sva_code: str) -> Tuple[str, int]:
    """
    Classify based on sampled value function usage
    Returns: (category, score_0_100)
    """
    svf_patterns = {
        'past': r'\$past\s*\(',
        'rose': r'\$rose\s*\(',
        'fell': r'\$fell\s*\(',
        'stable': r'\$stable\s*\(',
        'changed': r'\$changed\s*\(',
        'onehot': r'\$onehot\s*\(',
        'countones': r'\$countones\s*\('
    }
    
    found_svfs = []
    for name, pattern in svf_patterns.items():
        if re.search(pattern, sva_code):
            found_svfs.append(name)
    
    # Check for $past with delay > 1
    past_with_delay = re.search(r'\$past\s*\([^,]+,\s*([2-9]|\d{2,})\s*\)', sva_code)
    
    if len(found_svfs) >= 3 or past_with_delay:
        return ('advanced_svf', 85)
    elif len(found_svfs) >= 2:
        return ('multiple_svf', 60)
    elif len(found_svfs) == 1:
        return ('basic_svf', 40)
    else:
        return ('no_svf', 0)

def classify_design_context(design_context: str, rtl_code: str = None) -> Tuple[str, int]:
    """
    Classify based on design complexity
    Returns: (category, score_0_100)
    """
    design_lower = design_context.lower()
    
    # Estimate LOC if RTL provided
    loc = 0
    if rtl_code:
        loc = len([l for l in rtl_code.split('\n') if l.strip() and not l.strip().startswith('//')])
    
    # Pattern matching for design types
    complex_patterns = ['processor', 'cpu', 'cache', 'pipeline', 'interconnect', 'fabric']
    moderate_patterns = ['arbiter', 'controller', 'protocol', 'fsm', 'alu', 'memory']
    simple_patterns = ['mux', 'multiplexer', 'counter', 'decoder', 'encoder', 'shift']
    
    if any(p in design_lower for p in complex_patterns) or loc > 2000:
        return ('very_complex', 95)
    elif any(p in design_lower for p in moderate_patterns) or 500 < loc <= 2000:
        return ('complex_hierarchical', 75)
    elif loc > 200:
        return ('moderate_sequential', 50)
    elif any(p in design_lower for p in simple_patterns) or 50 < loc <= 200:
        return ('sequential_basic', 30)
    else:
        return ('simple_combinational', 10)

def classify_semantic_complexity(english_spec: str) -> Tuple[str, int]:
    """
    Classify semantic complexity based on specification language
    Returns: (category, score_0_100)
    """
    spec_lower = english_spec.lower()
    
    # Cross-module indicators
    if '.' in english_spec and any(word in spec_lower for word in ['module', 'stage', 'unit']):
        return ('cross_module', 90)
    
    # State-dependent indicators
    if any(word in spec_lower for word in ['state', 'idle', 'active', 'busy', 'transition']):
        return ('state_dependent', 75)
    
    # Multi-condition indicators
    condition_words = ['if', 'when', 'unless', 'only if', 'provided', 'given']
    condition_count = sum(1 for word in condition_words if word in spec_lower)
    
    if condition_count >= 3:
        return ('multi_condition', 65)
    elif condition_count >= 1:
        return ('conditional_logic', 40)
    else:
        return ('direct_mapping', 20)

def calculate_overall_complexity(scores: Dict[str, int]) -> Tuple[str, int]:
    """
    Calculate weighted overall complexity score
    Returns: (simple/medium/complex, score_0_100)
    """
    weights = {
        'temporal': 0.25,
        'signal': 0.20,
        'svf': 0.20,
        'design': 0.20,
        'semantic': 0.15
    }
    
    total_score = (
        weights['temporal'] * scores.get('temporal', 0) +
        weights['signal'] * scores.get('signal', 0) +
        weights['svf'] * scores.get('svf', 0) +
        weights['design'] * scores.get('design', 0) +
        weights['semantic'] * scores.get('semantic', 0)
    )
    
    if total_score <= 35:
        category = 'simple'
    elif total_score <= 65:
        category = 'medium'
    else:
        category = 'complex'
    
    return (category, int(total_score))

def classify_assertion(english_spec: str, sva_assertion: str, design_context: str, signals: List[str]) -> Dict:
    """
    Classify an assertion across all dimensions
    """
    # Classify across all dimensions
    temporal_class, temporal_score = classify_temporal_complexity(sva_assertion)
    signal_class, signal_score = classify_signal_complexity(sva_assertion, signals)
    svf_class, svf_score = classify_sampled_value_functions(sva_assertion)
    design_class, design_score = classify_design_context(design_context)
    semantic_class, semantic_score = classify_semantic_complexity(english_spec)
    
    # Calculate overall score
    scores = {
        'temporal': temporal_score,
        'signal': signal_score,
        'svf': svf_score,
        'design': design_score,
        'semantic': semantic_score
    }
    
    overall_class, overall_score = calculate_overall_complexity(scores)
    
    classification = {
        'classifications': {
            'temporal': {'class': temporal_class, 'score': temporal_score},
            'signal': {'class': signal_class, 'score': signal_score},
            'svf': {'class': svf_class, 'score': svf_score},
            'design': {'class': design_class, 'score': design_score},
            'semantic': {'class': semantic_class, 'score': semantic_score}
        },
        'overall_complexity': {
            'class': overall_class,
            'score': overall_score
        }
    }
    
    return classification

def create_spec_assertion_pair(file_data: Dict, spec_id: str, format_type: str) -> Dict:
    """
    Create a spec-assertion pair entry from parsed file data
    """
    pair = {'spec_id': spec_id}
    
    if format_type == 'original':
        # Handle original format
        spec = file_data['spec']
        assertion = file_data['assertion']
        rtl = file_data.get('rtl', '')
        
        # Extract signals if not provided
        signals = extract_signals_from_assertion(assertion)
        
        # Infer design context if not provided
        design_context = infer_design_context(spec, assertion)
        
        pair['english_spec'] = spec
        pair['sva_assertion'] = assertion
        pair['design_context'] = design_context
        pair['signals'] = signals
        
        # Get classification
        classification = classify_assertion(spec, assertion, design_context, signals)
        pair['complexity'] = classification['overall_complexity']['class']
        pair['classification'] = classification
        
    else:  # stratified format
        # Copy data directly
        pair['english_spec'] = file_data['english_spec']
        pair['sva_assertion'] = file_data['sva_assertion']
        pair['design_context'] = file_data['design_context']
        pair['signals'] = file_data['signals']
        
        # Get classification
        classification = classify_assertion(
            pair['english_spec'], pair['sva_assertion'], 
            pair['design_context'], pair['signals']
        )
        pair['complexity'] = classification['overall_complexity']['class']
        pair['classification'] = classification
    
    return pair

def load_classification_metadata(metadata_file: str) -> Dict:
    """Load classification metadata if available"""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Classification metadata not found at {metadata_file}")
        return {}

def detect_file_format(filepath: str) -> str:
    """
    Detect file format by looking at content
    Returns 'original' or 'stratified'
    """
    with open(filepath, 'r') as f:
        content = f.read(200)  # Read first 200 chars
    
    if 'specs:' in content or 'assertion:' in content:
        return 'original'
    else:
        return 'stratified'

def generate_database(input_dir: str, output_file: str, classification_file: str = None, 
                     format_type: str = 'auto') -> Dict:
    """
    Generate specs_database.json from a directory of assertion files
    
    Parameters:
    - input_dir: Directory containing assertion files
    - output_file: Path to output JSON file
    - classification_file: Optional path to pre-computed classification metadata
    - format_type: 'original', 'stratified', or 'auto' (detect automatically)
    
    Returns:
    - Database dictionary
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Load classification metadata if provided
    classification_metadata = {}
    if classification_file and os.path.exists(classification_file):
        metadata = load_classification_metadata(classification_file)
        if 'train_metadata' in metadata:
            classification_metadata = {
                item['spec_id']: item for item in metadata['train_metadata']
            }
    
    # Find all text files
    file_extensions = ('.txt', '.md', '.sv')
    assertion_files = [f for f in input_path.glob('*') 
                      if f.is_file() and (f.suffix in file_extensions or not f.suffix)]
    
    print(f"\nProcessing {len(assertion_files)} files from {input_dir}")
    
    pairs = []
    skipped = 0
    format_counts = {'original': 0, 'stratified': 0}
    
    for i, filepath in enumerate(assertion_files, 1):
        print(f"Processing {i}/{len(assertion_files)}: {filepath.name}")
        
        spec_id = filepath.stem
        
        try:
            # Determine format to use
            file_format = format_type
            if format_type == 'auto':
                file_format = detect_file_format(filepath)
                format_counts[file_format] += 1
            
            # Parse file based on format
            if file_format == 'original':
                file_data = parse_database_file(filepath)
            else:  # stratified
                file_data = parse_assertion_file(filepath)
            
            if not file_data:
                print(f"  ? Skipped: Could not parse file")
                skipped += 1
                continue
            
            # Create pair
            pair = create_spec_assertion_pair(file_data, spec_id, file_format)
            
            # If we have pre-computed classification, use it
            if spec_id in classification_metadata:
                metadata = classification_metadata[spec_id]
                pair['classification'] = {
                    'overall': metadata['overall_complexity'],
                    'temporal': metadata['classifications']['temporal'],
                    'signal': metadata['classifications']['signal'],
                    'svf': metadata['classifications']['svf'],
                    'design': metadata['classifications']['design'],
                    'semantic': metadata['classifications']['semantic']
                }
                pair['complexity'] = metadata['overall_complexity']['class']
            
            pairs.append(pair)
            
            complexity = pair['complexity'].upper()
            print(f"  ? Parsed: {pair['design_context']} - {complexity}")
            
        except Exception as e:
            skipped += 1
            print(f"  ? Error: {e}")
    
    # Create database structure
    database = {
        'metadata': {
            'version': '2.0',
            'total_pairs': len(pairs),
            'source_directory': str(input_path),
            'generated_from': input_dir,
            'classification_included': True,
            'format_stats': format_counts,
            'complexity_distribution': {
                'simple': sum(1 for p in pairs if p['complexity'] == 'simple'),
                'medium': sum(1 for p in pairs if p['complexity'] == 'medium'),
                'complex': sum(1 for p in pairs if p['complexity'] == 'complex')
            }
        },
        'pairs': pairs
    }
    
    return database

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

def prompt_for_split_config(total_files: int) -> Tuple[int, int]:
    """
    Prompt user for train/test split configuration
    Returns (train_size, test_size)
    """
    print(f"\n{'-'*60}")
    print(f"TRAIN/TEST SPLIT CONFIGURATION")
    print(f"{'-'*60}")
    print(f"Total available files: {total_files}")
    
    # Default split (2/3 train, 1/3 test)
    default_train = int(total_files * (2/3))
    default_test = total_files - default_train
    
    print(f"\nDefault split (2/3 train, 1/3 test):")
    print(f"- Training set: {default_train} files ({default_train/total_files*100:.1f}%)")
    print(f"- Testing set:  {default_test} files ({default_test/total_files*100:.1f}%)")
    
    use_default = get_user_confirmation("\nUse default 2/3 - 1/3 split?")
    
    if use_default:
        return default_train, default_test
    
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
                    return train_size, test_size
                    
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
                    return train_size, test_size
                    
            except ValueError:
                print("Please enter a valid percentage.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate specs_database.json from assertion files'
    )
    parser.add_argument('--input', default=None,
        help='Input directory with assertion files')
    parser.add_argument('--output', default=None,
        help='Output JSON file')
    parser.add_argument('--classification', default=None,
        help='Classification metadata file (split_metadata.json)')
    parser.add_argument('--format', choices=['original', 'stratified', 'auto'], default='auto',
        help='File format to use (default: auto-detect)')
    parser.add_argument('--interactive', action='store_true',
        help='Run in interactive mode with prompts')
    args = parser.parse_args()
    
    print("""
+----------------------------------------------------------+
¦   Enhanced Database Generator for ASSERT-FLOW           ¦
¦   Supports both original and stratified formats         ¦
+----------------------------------------------------------+
""")
    
    # Interactive or command-line mode
    interactive_mode = args.interactive or (args.input is None)
    
    # Get input directory
    input_dir = args.input
    if interactive_mode and input_dir is None:
        input_dir = input("Enter input directory path (default: database): ").strip()
        if not input_dir:
            input_dir = "database"
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Directory '{input_dir}' not found")
        return 1
    
    # Get output file
    output_file = args.output
    if interactive_mode and output_file is None:
        output_file = input("Enter output JSON file (default: specs_database.json): ").strip()
        if not output_file:
            output_file = "specs_database.json"
    
    if output_file is None:
        output_file = "specs_database.json"
    
    # Get format type
    format_type = args.format
    if interactive_mode and format_type == 'auto':
        format_options = {
            '1': 'original',
            '2': 'stratified',
            '3': 'auto'
        }
        print("\nFile format options:")
        print("  1. Original (specs:, assertion:, RTL: tags)")
        print("  2. Stratified (english_spec\\n\\nsva_assertion\\n\\ndesign_context\\n\\nsignals)")
        print("  3. Auto-detect (default)")
        
        format_choice = input("\nSelect file format [3]: ").strip()
        if not format_choice:
            format_choice = '3'
        
        if format_choice in format_options:
            format_type = format_options[format_choice]
        else:
            format_type = 'auto'
    
    # Get classification file
    classification_file = args.classification
    if interactive_mode and classification_file is None:
        classification_file = input("Enter classification metadata file (optional): ").strip()
    
    # Generate the database
    try:
        database = generate_database(
            input_dir=input_dir,
            output_file=output_file,
            classification_file=classification_file,
            format_type=format_type
        )
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(database, f, indent=2)
        
        print("\n" + "="*60)
        print(f"? Database saved to: {output_file}")
        
        # Print statistics
        pairs = database['pairs']
        complexity_counts = database['metadata']['complexity_distribution']
        
        print(f"\nDatabase statistics:")
        print(f"  Total pairs: {len(pairs)}")
        
        print(f"\n  Complexity distribution:")
        for complexity, count in complexity_counts.items():
            pct = (count / len(pairs) * 100) if len(pairs) > 0 else 0
            print(f"    {complexity.upper():8s}: {count:3d} ({pct:5.1f}%)")
        
        # Ask about train/test split if in interactive mode
        if interactive_mode:
            total_files = len(pairs)
            
            if get_user_confirmation(f"\nWould you like to create a train/test split for the {total_files} files?"):
                # Get split configuration
                train_size, test_size = prompt_for_split_config(total_files)
                
                # Output paths
                output_path = Path(output_file).parent
                output_basename = Path(output_file).stem
                
                split_script = f"""
# To create your train/test split, run:
python stratified_splitter.py \\
    --classification {output_file} \\
    --input {input_dir} \\
    --output-train {output_path}/train/ \\
    --output-test {output_path}/test/ \\
    --train-size {train_size} \\
    --test-size {test_size}
"""
                print("\n" + "="*60)
                print("TRAIN/TEST SPLIT COMMAND")
                print("="*60)
                print(split_script)
        
        print("\n" + "="*60)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
