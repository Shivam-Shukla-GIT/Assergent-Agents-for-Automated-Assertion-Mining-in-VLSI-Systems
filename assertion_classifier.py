#!/usr/bin/env python3
"""
assertion_classifier.py - Comprehensive Assertion Classification System
Classifies assertions across 5 dimensions:

Temporal Complexity
Signal Complexity
Sampled Value Functions
Design Context
Property Semantics

Usage:
python assertion_classifier.py --input database/ --output classification_report.json
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class AssertionClassifier:
    """Multi-dimensional assertion classifier"""
    def __init__(self):
        self.classification_report = {
            'total_assertions': 0,
            'by_complexity': {'simple': [], 'medium': [], 'complex': []},
            'by_temporal': {},
            'by_signal_count': {},
            'by_svf': {},
            'by_design': {},
            'detailed_metadata': []
        }

    def classify_temporal_complexity(self, sva_code: str) -> Tuple[str, int]:
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

    def classify_signal_complexity(self, sva_code: str, signals: List[str]) -> Tuple[str, int]:
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

    def classify_sampled_value_functions(self, sva_code: str) -> Tuple[str, int]:
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

    def classify_design_context(self, design_context: str, rtl_code: str = None) -> Tuple[str, int]:
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

    def classify_semantic_complexity(self, english_spec: str) -> Tuple[str, int]:
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

    def calculate_overall_complexity(self, scores: Dict[str, int]) -> Tuple[str, int]:
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

    def classify_assertion_file(self, filepath: str) -> Dict:
        """
        Classify a single assertion file
        Returns: metadata dictionary
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse file content (assuming format: english_spec\n\nsva_assertion\n\ndesign_context\n\nsignals)
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
        
        # Classify across all dimensions
        temporal_class, temporal_score = self.classify_temporal_complexity(sva_assertion)
        signal_class, signal_score = self.classify_signal_complexity(sva_assertion, signals)
        svf_class, svf_score = self.classify_sampled_value_functions(sva_assertion)
        design_class, design_score = self.classify_design_context(design_context)
        semantic_class, semantic_score = self.classify_semantic_complexity(english_spec)
        
        # Calculate overall
        scores = {
            'temporal': temporal_score,
            'signal': signal_score,
            'svf': svf_score,
            'design': design_score,
            'semantic': semantic_score
        }
        
        overall_class, overall_score = self.calculate_overall_complexity(scores)
        
        metadata = {
            'file': os.path.basename(filepath),
            'spec_id': os.path.splitext(os.path.basename(filepath))[0],
            'english_spec': english_spec[:100] + '...' if len(english_spec) > 100 else english_spec,
            'design_context': design_context,
            'signals': signals,
            'signal_count': len(signals),
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
        
        return metadata

    def classify_all(self, input_dir: str) -> Dict:
        """
        Classify all assertion files in directory
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all .txt files
        assertion_files = list(input_path.glob('*.txt'))
        
        print(f"Found {len(assertion_files)} assertion files")
        
        all_metadata = []
        
        for i, filepath in enumerate(assertion_files, 1):
            print(f"Classifying {i}/{len(assertion_files)}: {filepath.name}")
            
            metadata = self.classify_assertion_file(str(filepath))
            
            if metadata:
                all_metadata.append(metadata)
                
                # Update report
                complexity = metadata['overall_complexity']['class']
                self.classification_report['by_complexity'][complexity].append(metadata['spec_id'])
        
        # Generate statistics
        self.classification_report['total_assertions'] = len(all_metadata)
        self.classification_report['detailed_metadata'] = all_metadata
        
        # Count by dimensions
        for metadata in all_metadata:
            # Temporal
            temporal = metadata['classifications']['temporal']['class']
            self.classification_report['by_temporal'][temporal] = \
                self.classification_report['by_temporal'].get(temporal, 0) + 1
            
            # Signal count
            sig_count = metadata['signal_count']
            sig_key = f"{sig_count}_signals"
            self.classification_report['by_signal_count'][sig_key] = \
                self.classification_report['by_signal_count'].get(sig_key, 0) + 1
            
            # SVF
            svf = metadata['classifications']['svf']['class']
            self.classification_report['by_svf'][svf] = \
                self.classification_report['by_svf'].get(svf, 0) + 1
            
            # Design
            design = metadata['classifications']['design']['class']
            self.classification_report['by_design'][design] = \
                self.classification_report['by_design'].get(design, 0) + 1
        
        return self.classification_report

    def print_summary(self):
        """Print classification summary"""
        report = self.classification_report
        
        print("\n" + "="*70)
        print("CLASSIFICATION SUMMARY")
        print("="*70)
        
        print(f"\nTotal Assertions: {report['total_assertions']}")
        
        print("\nBy Overall Complexity:")
        for complexity in ['simple', 'medium', 'complex']:
            count = len(report['by_complexity'][complexity])
            pct = (count / report['total_assertions'] * 100) if report['total_assertions'] > 0 else 0
            print(f"  {complexity.upper():10s}: {count:3d} ({pct:5.1f}%)")
        
        print("\nBy Temporal Complexity:")
        for temporal, count in sorted(report['by_temporal'].items()):
            pct = (count / report['total_assertions'] * 100) if report['total_assertions'] > 0 else 0
            print(f"  {temporal:20s}: {count:3d} ({pct:5.1f}%)")
        
        print("\nBy Signal Count:")
        for sig, count in sorted(report['by_signal_count'].items()):
            pct = (count / report['total_assertions'] * 100) if report['total_assertions'] > 0 else 0
            print(f"  {sig:20s}: {count:3d} ({pct:5.1f}%)")
        
        print("\nBy Sampled Value Functions:")
        for svf, count in sorted(report['by_svf'].items()):
            pct = (count / report['total_assertions'] * 100) if report['total_assertions'] > 0 else 0
            print(f"  {svf:20s}: {count:3d} ({pct:5.1f}%)")
        
        print("\nBy Design Context:")
        for design, count in sorted(report['by_design'].items()):
            pct = (count / report['total_assertions'] * 100) if report['total_assertions'] > 0 else 0
            print(f"  {design:25s}: {count:3d} ({pct:5.1f}%)")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Classify assertions across multiple dimensions')
    parser.add_argument('--input', required=True, help='Input directory containing assertion .txt files')
    parser.add_argument('--output', default='classification_report.json', help='Output JSON file')
    args = parser.parse_args()

    classifier = AssertionClassifier()

    print("Starting classification...")
    report = classifier.classify_all(args.input)

    # Print summary
    classifier.print_summary()

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n? Classification report saved to: {args.output}")

if __name__ == "__main__":
    main()
