#!/usr/bin/env python3
"""
rag_assertion_generator.py - RAG-Enhanced Assertion Generation
FIXED VERSION: Using gemini-2.0-flash with timestamped result directories

Usage:
    conda activate myenv
    export GEMINI_API_KEY='your-api-key'
    python3 rag_assertion_generator.py
"""

import json
import os
import re
import subprocess
import logging
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# CRITICAL: Set this FIRST to avoid fork warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Check sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    USE_FULL_RAG = True
    print("? Using SentenceTransformer embeddings")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_FULL_RAG = False
    print("? Using TF-IDF fallback (install sentence-transformers for better results)")

# Check FAISS availability
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    print("? FAISS not available, using sklearn similarity")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_assertion_generator.log'),
        logging.StreamHandler()
    ]
)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpecAssertionPair:
    spec_id: str
    english_spec: str
    sva_assertion: str
    design_context: str
    signals: List[str]
    complexity: str
    embedding: Optional[np.ndarray] = None

@dataclass
class RTLContext:
    rtl_file: str
    rtl_code: str
    module_name: str
    signals: List[str]
    clock_signal: str
    reset_signal: str

@dataclass
class VerificationResult:
    success: bool
    status: str
    error_message: str
    error_type: str
    counterexample: Optional[str] = None

# ============================================================================
# RAG Knowledge Base
# ============================================================================

class RAGKnowledgeBase:
    """RAG knowledge base for spec?assertion retrieval"""
    
    def __init__(self, database_path: str):
        self.logger = logging.getLogger("RAG-KB")
        self.database_path = database_path
        self.pairs: List[SpecAssertionPair] = []
        
        # Initialize encoder
        if USE_FULL_RAG:
            self.logger.info("Loading SentenceTransformer model (this may take a moment)...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
            self.encoder_fitted = False
        
        self.index = None
        self.embeddings_matrix = None
        
        # Load database
        self._load_database()
        self._build_index()

    def _load_database(self):
        """Load spec-assertion pairs from JSON file"""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                pairs_data = data
            elif isinstance(data, dict) and 'pairs' in data:
                pairs_data = data['pairs']
            else:
                raise ValueError("Invalid database format")
            
            for pair_data in pairs_data:
                pair = SpecAssertionPair(
                    spec_id=pair_data.get('spec_id', ''),
                    english_spec=pair_data.get('english_spec', ''),
                    sva_assertion=pair_data.get('sva_assertion', ''),
                    design_context=pair_data.get('design_context', ''),
                    signals=pair_data.get('signals', []),
                    complexity=pair_data.get('complexity', 'medium')
                )
                self.pairs.append(pair)
            
            self.logger.info(f"? Loaded {len(self.pairs)} spec-assertion pairs")
            
        except Exception as e:
            self.logger.error(f"Error loading database: {e}")
            raise

    def _build_index(self):
        """Build semantic search index"""
        if not self.pairs:
            return
        
        search_texts = [
            f"{p.english_spec} {p.design_context} {' '.join(p.signals)}"
            for p in self.pairs
        ]
        
        if USE_FULL_RAG:
            self.logger.info("Generating embeddings...")
            embeddings = self.encoder.encode(search_texts, show_progress_bar=True)
            
            if USE_FAISS:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.index.add(normalized.astype('float32'))
                self.logger.info(f"? Built FAISS index with {self.index.ntotal} examples")
            else:
                self.embeddings_matrix = embeddings
                self.logger.info(f"? Built sklearn index with {len(self.pairs)} examples")
        else:
            self.encoder.fit(search_texts)
            self.encoder_fitted = True
            embeddings = self.encoder.transform(search_texts).toarray()
            self.embeddings_matrix = np.array(embeddings)
            self.logger.info(f"? Built TF-IDF index with {len(self.pairs)} examples")
        
        for pair, emb in zip(self.pairs, embeddings):
            pair.embedding = emb

    def retrieve_similar(self, user_spec: str, k: int = 3, 
                        min_similarity: float = 0.3) -> List[Tuple[SpecAssertionPair, float]]:
        """Retrieve k most similar spec-assertion pairs"""
        if not self.pairs:
            return []
        
        # Encode query
        if USE_FULL_RAG:
            query_emb = self.encoder.encode([user_spec], show_progress_bar=False)
            
            if USE_FAISS:
                query_norm = query_emb / np.linalg.norm(query_emb)
                similarities, indices = self.index.search(query_norm.astype('float32'), k)
                results = [(self.pairs[idx], float(sim)) 
                          for sim, idx in zip(similarities[0], indices[0]) if sim >= min_similarity]
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_emb, self.embeddings_matrix)[0]
                top_k_idx = np.argsort(similarities)[-k:][::-1]
                results = [(self.pairs[idx], float(similarities[idx])) 
                          for idx in top_k_idx if similarities[idx] >= min_similarity]
        else:
            query_emb = self.encoder.transform([user_spec]).toarray()
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_emb, self.embeddings_matrix)[0]
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            results = [(self.pairs[idx], float(similarities[idx])) 
                      for idx in top_k_idx if similarities[idx] >= min_similarity]
        
        self.logger.info(f"Retrieved {len(results)} similar specifications")
        for i, (pair, score) in enumerate(results, 1):
            self.logger.info(f"  {i}. {pair.spec_id} (similarity: {score:.3f})")
        
        return results

# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    def __init__(self):
        self.logger = logging.getLogger("PromptBuilder")

    def build_initial_prompt(self, user_spec: str, similar_pairs: List[Tuple[SpecAssertionPair, float]],
                           rtl_context: RTLContext) -> str:
        """Build initial prompt with RAG examples"""
        
        prompt = f"""SystemVerilog Assertion Generation Task

USER SPECIFICATION:
"{user_spec}"

RTL DESIGN CONTEXT:
- Module: {rtl_context.module_name}
- Available signals: {', '.join(rtl_context.signals[:30])}{'...' if len(rtl_context.signals) > 30 else ''}
- Clock signal: {rtl_context.clock_signal}
- Reset signal: {rtl_context.reset_signal}
"""
        
        if similar_pairs:
            prompt += "\n" + "="*60 + "\n"
            prompt += "SIMILAR SUCCESSFUL EXAMPLES (from knowledge base):\n"
            prompt += "="*60 + "\n"
            for i, (pair, score) in enumerate(similar_pairs[:3], 1):
                prompt += f"""

Example {i} (Similarity: {score:.3f}):
Specification: "{pair.english_spec}"
Design Context: {pair.design_context}
Signals Used: {', '.join(pair.signals[:10])}
Correct Assertion:
```systemverilog
{pair.sva_assertion}
```
"""
        
        prompt += f"""
GENERATION GUIDELINES:
- Use @(posedge {rtl_context.clock_signal}) for clock events
- ALWAYS include 'disable iff (!{rtl_context.reset_signal})' for reset handling
- Use |=> for next-cycle implications, |-> for same-cycle
- Use $past(signal) for previous cycle values (use $past(signal, 1) for explicit 1-cycle delay)
- Ensure ALL signals exist in the available signals list
- Follow SystemVerilog syntax strictly
- End with semicolon after closing parenthesis

REQUIRED FORMAT:
```systemverilog
property <property_name>;
  @(posedge {rtl_context.clock_signal}) disable iff (!{rtl_context.reset_signal})
    <trigger_condition> |<operator> <expected_result>;
endproperty
assert property (<property_name>);
```

OR for inline assertions:
```systemverilog
assert property (@(posedge {rtl_context.clock_signal}) disable iff (!{rtl_context.reset_signal})
    <trigger_condition> |<operator> <expected_result>);
```

CRITICAL: Generate ONLY the SystemVerilog assertion code. No explanations, no markdown outside code blocks.

Generate the assertion now:
"""
        return prompt

    def build_refinement_prompt(self, user_spec: str, similar_pairs: List, rtl_context: RTLContext, 
                              previous_attempt: str, verification_result: VerificationResult, 
                              iteration: int) -> str:
        """Build refinement prompt with error feedback"""
        
        prompt = self.build_initial_prompt(user_spec, similar_pairs, rtl_context)
        
        prompt += f"""
{"="*60}
PREVIOUS ATTEMPT FAILED (Iteration {iteration})
{"="*60}

Previous Assertion:
```systemverilog
{previous_attempt}
```

VERIFICATION RESULT: ? FAILED
Status: {verification_result.status}
Error Type: {verification_result.error_type}

Error Message:
{verification_result.error_message}

SPECIFIC REFINEMENT GUIDANCE:
{self._get_error_guidance(verification_result)}

CRITICAL: Fix the specific error above. Generate ONLY the corrected assertion code.

Generate the corrected assertion:
"""
        return prompt

    def _get_error_guidance(self, result: VerificationResult) -> str:
        """Provide specific guidance based on error type"""
        
        if 'syntax' in result.error_type.lower() or 'VERI' in result.error_type:
            return """- SYNTAX ERROR detected
  * Check: semicolons at end, matching parentheses/braces
  * Verify: all signal names EXACTLY match RTL (case-sensitive)
  * Ensure: proper SystemVerilog assertion syntax
  * Common issues: missing semicolon, typo in signal names, incorrect operators"""
        
        elif result.status == 'cex':
            return """- COUNTEREXAMPLE found: assertion logic is INCORRECT
  * The trigger condition occurs but expected result does NOT hold
  * Check: temporal operators (|-> for same cycle, |=> for next cycle)
  * Verify: $past() usage and timing relationships
  * Review: the similar examples above for correct patterns"""
        
        elif result.status == 'unreachable':
            return """- Property is UNREACHABLE: trigger condition NEVER occurs
  * Issue: conditions are too restrictive or impossible
  * Fix: relax trigger conditions, check signal initialization
  * Verify: reset behavior allows the property to be tested"""
        
        elif result.status == 'undetermined':
            return """- Verification UNDETERMINED: too complex or resource limits
  * Try: simplifying the assertion logic
  * Check: if assertion is too broad or has unbounded delays"""
        
        elif 'llm' in result.error_type.lower():
            return """- LLM generation error occurred
  * Ensure: proper code block formatting with ```systemverilog
  * Generate: complete, valid SystemVerilog assertion"""
        
        else:
            return """- Review error message carefully
  * Compare with similar examples
  * Ensure syntactic and semantic correctness"""

# ============================================================================
# Gemini LLM Interface (EXACT ASSERT-AGENT METHOD)
# ============================================================================

class GeminiInterface:
    """Gemini API interface using gemini-2.0-flash (updated for new API)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger("Gemini")

    def generate(self, prompt: str) -> str:
        """Generate assertion using Gemini 2.0 Flash"""
        
        # Updated URL for gemini-2.0-flash (no API key in URL)
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Updated headers with X-goog-api-key (new API format)
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            self.logger.info("Querying Gemini API (gemini-2.0-flash)...")
            
            # POST request with updated headers
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            # Response handling
            if response.status_code == 200:
                response_json = response.json()
                result = response_json['candidates'][0]['content']['parts'][0]['text']
                self.logger.info(f"? Received response ({len(result)} characters)")
                return result
            else:
                error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error querying Gemini: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

    def extract_assertion(self, response: str) -> str:
        """Extract SystemVerilog assertion from LLM response"""
        
        if response.startswith("Error:"):
            return response
        
        # Try to find systemverilog code block
        match = re.search(r'```systemverilog\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            self.logger.info("? Extracted assertion from systemverilog code block")
            return code
        
        # Try to find any code block
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Remove language identifier if present
            if code.startswith('systemverilog'):
                code = code[13:].strip()
            self.logger.info("? Extracted assertion from generic code block")
            return code
        
        # Try to find property...endproperty block with assert
        match = re.search(
            r'(property\s+\w+\s*;.*?endproperty.*?assert\s+property\s*\([^)]+\)\s*;)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if match:
            code = match.group(1).strip()
            self.logger.info("? Extracted property+assert block")
            return code
        
        # Try to find inline assert property statement
        match = re.search(
            r'(assert\s+property\s*\(@.*?\)\s*;)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if match:
            code = match.group(1).strip()
            self.logger.info("? Extracted inline assert property")
            return code
        
        # Last resort: if response contains assertion keywords, return it
        if 'assert' in response.lower() and 'property' in response.lower():
            self.logger.warning("Using full response as assertion (no code block found)")
            return response.strip()
        
        error_msg = f"Error: Could not extract assertion from response.\nResponse preview:\n{response[:500]}..."
        self.logger.error("Failed to extract assertion from response")
        return error_msg

# ============================================================================
# JasperGold Verification Engine (Using ASSERT-AGENT method)
# ============================================================================

class JasperGoldEngine:
    """JasperGold verification using ASSERT-AGENT's proven method"""
    
    def __init__(self, work_dir: str = "jasper_work"):
        self.logger = logging.getLogger("JasperGold")
        self.work_dir = work_dir
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

    def verify_assertion(self, rtl_file: str, assertion: str, module_name: str, 
                        iteration: int) -> VerificationResult:
        """Verify assertion using JasperGold"""
        
        # Check if assertion generation failed
        if assertion.startswith("Error:"):
            return VerificationResult(
                success=False,
                status='llm_error',
                error_message=assertion,
                error_type='llm_generation_failed'
            )
        
        # Embed assertion in RTL
        rtl_with_assertion = self._embed_assertion(rtl_file, assertion)
        work_rtl = os.path.join(self.work_dir, f"rtl_with_assertion_iter_{iteration}.sv")
        
        with open(work_rtl, 'w') as f:
            f.write(rtl_with_assertion)
        
        self.logger.info(f"Created RTL with assertion: {work_rtl}")
        
        # Create JasperGold script
        jg_script = self._create_jasper_script(work_rtl, module_name, iteration)
        
        # Execute JasperGold
        jasper_output = self._execute_jaspergold(jg_script, iteration)
        
        # Parse results
        return self._parse_jasper_output(jasper_output)

    def _embed_assertion(self, rtl_file: str, assertion: str) -> str:
        """Embed assertion into RTL module"""
        
        with open(rtl_file, 'r') as f:
            rtl_code = f.read()
        
        lines = rtl_code.split('\n')
        
        # Find endmodule line
        endmodule_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('endmodule'):
                endmodule_idx = i
                break
        
        if endmodule_idx == -1:
            raise ValueError("No endmodule found in RTL file")
        
        # Insert assertion before endmodule
        assertion_lines = [
            "",
            "  // RAG-generated assertion",
            f"  {assertion}",
            ""
        ]
        
        modified_lines = lines[:endmodule_idx] + assertion_lines + lines[endmodule_idx:]
        
        return '\n'.join(modified_lines)

    def _create_jasper_script(self, rtl_file: str, module_name: str, iteration: int) -> str:
        """Create JasperGold TCL script (ASSERT-AGENT style)"""
        
        script_content = f"""# RAG Assertion Verification Script - Iteration {iteration}

analyze -sv {os.path.basename(rtl_file)}
elaborate -top {module_name}

clock clk
reset rst_n

# Run formal proof
prove -all

# Extract property information
set all_props [get_property_list]

puts "=== RAG ASSERTION VERIFICATION RESULTS ==="
foreach prop $all_props {{
    set info [get_property_info $prop]
    
    # Check for failures
    if {{[string match "*status cex*" $info] || 
         [string match "*status ar_cex*" $info] ||
         [string match "*status undetermined*" $info] ||
         [string match "*status unknown*" $info] ||
         [string match "*status error*" $info]}} {{
        puts "FAILED_PROPERTY: $prop"
        puts "PROPERTY_INFO: $info"
    }}
    
    # Check for unreachable
    if {{[string match "*status unreachable*" $info]}} {{
        puts "FAILED_PROPERTY: $prop"
        puts "PROPERTY_INFO: $info"
    }}
    
    # Check for success
    if {{[string match "*status proved*" $info]}} {{
        puts "PASSED_PROPERTY: $prop"
        puts "PROPERTY_INFO: $info"
    }}
}}

puts "=== END VERIFICATION RESULTS ==="
exit
"""
        
        script_path = os.path.join(self.work_dir, f"run_iter_{iteration}.jg")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Created JasperGold script: {script_path}")
        return script_path

    def _execute_jaspergold(self, script_path: str, iteration: int) -> str:
        """Execute JasperGold using ASSERT-AGENT method"""
        
        tcsh_script = os.path.join(self.work_dir, f"run_jasper_iter_{iteration}.tcsh")
        
        abs_work_dir = os.path.abspath(self.work_dir)
        abs_script_path = os.path.abspath(script_path)
        
        tcsh_content = f"""#!/bin/tcsh
source /cadence/cshrc
cd "{abs_work_dir}"
echo "Starting JasperGold execution..."
echo "Working directory: `pwd`"
echo "Script file: {os.path.basename(abs_script_path)}"
jaspergold -batch -tcl "{os.path.basename(abs_script_path)}" >& "jasper_output_iter_{iteration}.log"
echo "JasperGold execution completed with exit code: $status"
"""
        
        with open(tcsh_script, 'w') as f:
            f.write(tcsh_content)
        os.chmod(tcsh_script, 0o755)
        
        try:
            self.logger.info(f"Executing JasperGold (iteration {iteration})...")
            
            tcsh_path = os.path.abspath(tcsh_script)
            
            # Run JasperGold
            process = subprocess.Popen(
                ["/bin/tcsh", tcsh_path],
                cwd=abs_work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=300)
                return_code = process.returncode
                
                self.logger.info(f"JasperGold completed with return code: {return_code}")
                
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error("JasperGold execution timed out (>300s)")
                return "ERROR: JasperGold execution timed out"
            
            # Read log file
            log_path = os.path.join(abs_work_dir, f"jasper_output_iter_{iteration}.log")
            
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    output = f.read()
                self.logger.info(f"Read JasperGold log: {len(output)} characters")
                return output
            else:
                self.logger.error(f"JasperGold log not found: {log_path}")
                return "ERROR: No JasperGold output log found"
                
        except Exception as e:
            self.logger.error(f"JasperGold execution failed: {e}")
            return f"ERROR: {str(e)}"

    def _parse_jasper_output(self, output: str) -> VerificationResult:
        """Parse JasperGold output for results"""
        
        # Check for errors first
        if output.startswith("ERROR:"):
            return VerificationResult(
                success=False,
                status='execution_error',
                error_message=output,
                error_type='jasper_execution_failed'
            )
        
        # Check for compilation/syntax errors
        if 'ERROR' in output and ('VERI-' in output or 'syntax' in output.lower()):
            error_lines = [line for line in output.split('\n') if 'ERROR' in line]
            return VerificationResult(
                success=False,
                status='syntax_error',
                error_message='\n'.join(error_lines[:5]),
                error_type='VERI_syntax_error'
            )
        
        # Check for passed properties
        if 'PASSED_PROPERTY:' in output or 'status proved' in output.lower():
            return VerificationResult(
                success=True,
                status='pass',
                error_message='',
                error_type=''
            )
        
        # Check for specific failure types
        if 'status cex' in output.lower():
            # Extract counterexample info if available
            cex_match = re.search(r'PROPERTY_INFO:.*?status cex.*', output, re.IGNORECASE)
            cex_info = cex_match.group(0) if cex_match else "Counterexample found"
            
            return VerificationResult(
                success=False,
                status='cex',
                error_message=cex_info,
                error_type='counterexample',
                counterexample=cex_info
            )
        
        if 'status unreachable' in output.lower():
            return VerificationResult(
                success=False,
                status='unreachable',
                error_message='Property is unreachable - trigger condition never occurs',
                error_type='unreachable'
            )
        
        if 'status undetermined' in output.lower():
            return VerificationResult(
                success=False,
                status='undetermined',
                error_message='Verification undetermined - resource/time limits or complexity',
                error_type='undetermined'
            )
        
        if 'status unknown' in output.lower():
            return VerificationResult(
                success=False,
                status='unknown',
                error_message='Verification status unknown - tool limitations',
                error_type='unknown'
            )
        
        # No clear result found
        return VerificationResult(
            success=False,
            status='unknown',
            error_message='Could not determine verification status from JasperGold output',
            error_type='parse_error'
        )

# ============================================================================
# Main RAG Assertion Generator Framework
# ============================================================================

class RAGAssertionGenerator:
    """Main framework coordinating RAG retrieval, LLM generation, and verification"""
    
    def __init__(self, database_path: str, api_key: str, rtl_file: str = None):
        self.logger = logging.getLogger("ASSERT-FLOW")
        
        # Create timestamped work directory
        self.work_dir = self._create_timestamped_workdir(rtl_file)
        self.logger.info(f"Created work directory: {self.work_dir}")
        
        # Initialize components
        self.rag_kb = RAGKnowledgeBase(database_path)
        self.prompt_builder = PromptBuilder()
        self.llm = GeminiInterface(api_key)
        self.jasper = JasperGoldEngine(self.work_dir)
    
    def _create_timestamped_workdir(self, rtl_file: str = None) -> str:
        """Create timestamped directory: results/rtlname_timestamp/"""
        
        # Create base results directory
        base_dir = Path("results")
        base_dir.mkdir(exist_ok=True)
        
        # Extract RTL filename (without path and extension)
        if rtl_file:
            rtl_name = Path(rtl_file).stem  # e.g., "cpu_core" from "cpu_core.sv"
        else:
            rtl_name = "unknown_design"
        
        # Create timestamp: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory: results/cpu_core_20251009_153045/
        work_dir = base_dir / f"{rtl_name}_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        return str(work_dir)

    def generate_assertion(self, user_spec: str, rtl_file: str, 
                          max_iterations: int = 10, k: int = 3) -> Dict:
        """Main assertion generation loop with iterative refinement"""
        
        self.logger.info("="*80)
        self.logger.info("RAG-Enhanced Assertion Generation")
        self.logger.info("="*80)
        
        # Extract RTL context
        rtl_context = self._extract_rtl_context(rtl_file)
        self.logger.info(f"RTL Context: Module={rtl_context.module_name}, "
                        f"Signals={len(rtl_context.signals)}, "
                        f"Clock={rtl_context.clock_signal}, Reset={rtl_context.reset_signal}")
        
        iteration = 0
        previous_attempt = None
        verification_result = None
        
        while iteration < max_iterations:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ITERATION {iteration + 1}/{max_iterations}")
            self.logger.info('='*60)
            
            # Step 1: Retrieve similar examples from RAG knowledge base
            similar_pairs = self.rag_kb.retrieve_similar(user_spec, k=k)
            
            if not similar_pairs:
                self.logger.warning("No similar examples found in knowledge base")
            
            # Step 2: Build prompt (initial or refinement)
            if iteration == 0:
                prompt = self.prompt_builder.build_initial_prompt(
                    user_spec, similar_pairs, rtl_context
                )
            else:
                prompt = self.prompt_builder.build_refinement_prompt(
                    user_spec, similar_pairs, rtl_context, 
                    previous_attempt, verification_result, iteration
                )
            
            # Save prompt for debugging
            prompt_file = os.path.join(self.work_dir, f"prompt_iter_{iteration+1}.txt")
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            self.logger.info(f"Saved prompt: {prompt_file}")
            
            # Step 3: Generate assertion using LLM
            llm_response = self.llm.generate(prompt)
            
            # Save LLM response
            response_file = os.path.join(self.work_dir, f"llm_response_iter_{iteration+1}.txt")
            with open(response_file, 'w') as f:
                f.write(llm_response)
            self.logger.info(f"Saved LLM response: {response_file}")
            
            # Extract assertion from response
            assertion = self.llm.extract_assertion(llm_response)
            
            if assertion.startswith("Error:"):
                self.logger.error(f"LLM generation failed: {assertion}")
                return {
                    'success': False,
                    'iterations': iteration + 1,
                    'last_error': assertion,
                    'error_type': 'llm_generation_failed',
                    'work_dir': self.work_dir
                }
            
            # Log generated assertion
            self.logger.info(f"\nGenerated Assertion:\n{'-'*60}\n{assertion}\n{'-'*60}")
            
            # Save assertion
            assertion_file = os.path.join(self.work_dir, f"assertion_iter_{iteration+1}.sv")
            with open(assertion_file, 'w') as f:
                f.write(assertion)
            
            # Step 4: Verify assertion using JasperGold
            verification_result = self.jasper.verify_assertion(
                rtl_file, assertion, rtl_context.module_name, iteration + 1
            )
            
            # Check verification result
            if verification_result.success:
                self.logger.info("\n" + "="*60)
                self.logger.info("??? VERIFICATION PASSED ???")
                self.logger.info("="*60)
                
                # Save final assertion
                final_file = os.path.join(self.work_dir, "final_assertion.sv")
                with open(final_file, 'w') as f:
                    f.write(assertion)
                
                return {
                    'success': True,
                    'assertion': assertion,
                    'iterations': iteration + 1,
                    'output_file': final_file,
                    'work_dir': self.work_dir
                }
            else:
                self.logger.warning(f"\n? Verification FAILED")
                self.logger.warning(f"Status: {verification_result.status}")
                self.logger.warning(f"Error: {verification_result.error_message}")
            
            # Update for next iteration
            previous_attempt = assertion
            iteration += 1
        
        # Max iterations reached
        self.logger.error(f"\n? Failed to generate correct assertion after {max_iterations} iterations")
        
        return {
            'success': False,
            'iterations': max_iterations,
            'last_error': verification_result.error_message if verification_result else 'Unknown error',
            'error_type': verification_result.error_type if verification_result else 'max_iterations_reached',
            'work_dir': self.work_dir
        }

    def _extract_rtl_context(self, rtl_file: str) -> RTLContext:
        """Extract RTL module context (signals, module name, clock, reset)"""
        
        with open(rtl_file, 'r') as f:
            rtl_code = f.read()
        
        # Extract module name
        module_match = re.search(r'module\s+(\w+)', rtl_code)
        module_name = module_match.group(1) if module_match else 'top'
        
        # Extract signals
        signals = self._extract_signals(rtl_code)
        
        # Identify clock and reset signals
        clock = next((s for s in signals if 'clk' in s.lower() or 'clock' in s.lower()), 'clk')
        reset = next((s for s in signals if 'rst' in s.lower() or 'reset' in s.lower()), 'rst_n')
        
        return RTLContext(
            rtl_file=rtl_file,
            rtl_code=rtl_code,
            module_name=module_name,
            signals=signals,
            clock_signal=clock,
            reset_signal=reset
        )

    def _extract_signals(self, rtl_code: str) -> List[str]:
        """Extract all signal declarations from RTL"""
        
        signals = set()
        
        patterns = [
            r'\binput\s+(?:logic\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
            r'\boutput\s+(?:logic\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
            r'\binout\s+(?:logic\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
            r'\blogic\s+(?:\[[^\]]+\]\s+)?(\w+)',
            r'\breg\s+(?:\[[^\]]+\]\s+)?(\w+)',
            r'\bwire\s+(?:\[[^\]]+\]\s+)?(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, rtl_code)
            signals.update(matches)
        
        return sorted(list(signals))

# ============================================================================
# Interactive Interface
# ============================================================================

def main():
    """Main interactive interface"""
    
    print("""
+----------------------------------------------------------+
¦   ASSERT-FLOW v2.0: RAG-Enhanced Assertion Generator    ¦
¦   Interactive Mode - Production Version                 ¦
+----------------------------------------------------------+
""")
    
    # Check API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("\n? ERROR: GEMINI_API_KEY environment variable not set")
        print("\nTo fix this:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set it: export GEMINI_API_KEY='your-key-here'")
        print("3. Verify: echo $GEMINI_API_KEY")
        return 1

    print(f"? API key found: {api_key[:10]}...{api_key[-4:]}")

    # Get user inputs
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)

    # Specification input
    spec_input = input("\nEnter specification (or file path with @): ").strip()
    if spec_input.startswith('@'):
        spec_file = spec_input[1:]
        if not os.path.exists(spec_file):
            print(f"? ERROR: Spec file not found: {spec_file}")
            return 1
        with open(spec_file, 'r') as f:
            user_spec = f.read().strip()
        print(f"? Loaded spec from: {spec_file}")
    else:
        user_spec = spec_input

    # RTL file input
    rtl_file = input("Enter RTL file path: ").strip()
    if not os.path.exists(rtl_file):
        print(f"? ERROR: RTL file not found: {rtl_file}")
        return 1
    print(f"? RTL file found: {rtl_file}")

    # Database file input
    database_file = input("Enter database JSON path (default: specs_database.json): ").strip()
    if not database_file:
        database_file = "specs_database.json"
    if not os.path.exists(database_file):
        print(f"? ERROR: Database not found: {database_file}")
        print("\nCreate it by running:")
        print("    python3 prepare_database.py")
        return 1
    print(f"? Database found: {database_file}")

    # Max iterations
    max_iter_input = input("Enter max iterations (default: 10): ").strip()
    max_iterations = int(max_iter_input) if max_iter_input.isdigit() else 10

    # Number of similar examples to retrieve
    k_input = input("Enter number of similar examples to retrieve (default: 3): ").strip()
    k = int(k_input) if k_input.isdigit() else 3

    # Summary
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Specification: {user_spec[:70]}...")
    print(f"RTL File: {rtl_file}")
    print(f"Database: {database_file}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Similar Examples (k): {k}")
    print("="*60)

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return 0

    # Initialize and run
    try:
        print("\n" + "="*60)
        print("Starting RAG Assertion Generation...")
        print("="*60 + "\n")
        
        generator = RAGAssertionGenerator(database_file, api_key, rtl_file)
        result = generator.generate_assertion(user_spec, rtl_file, max_iterations, k)
        
        # Display results
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        if result['success']:
            print("??? SUCCESS! ???")
            print(f"\nConverged in {result['iterations']} iteration(s)")
            print(f"Output file: {result['output_file']}")
            print(f"Work directory: {result['work_dir']}")
            print(f"\nFinal Assertion:")
            print("-"*80)
            print(result['assertion'])
            print("-"*80)
            print("\n? Assertion verified successfully by JasperGold")
        else:
            print("??? FAILED ???")
            print(f"\nIterations used: {result['iterations']}")
            print(f"Error type: {result.get('error_type', 'unknown')}")
            print(f"Last error: {result.get('last_error', 'Unknown error')}")
            print(f"\nWork directory: {result['work_dir']}")
            print("\nCheck the following files for debugging:")
            print("  - prompt_iter_*.txt: Prompts sent to LLM")
            print("  - llm_response_iter_*.txt: LLM responses")
            print("  - assertion_iter_*.sv: Generated assertions")
            print("  - jasper_output_iter_*.log: JasperGold logs")
        
        print("="*80)
        
        return 0 if result['success'] else 1
        
    except KeyboardInterrupt:
        print("\n\n? Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n? CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
