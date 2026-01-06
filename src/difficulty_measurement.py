import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple, Union, Dict, Any
from collections import Counter, OrderedDict, defaultdict
import re
import string
import time
from datetime import datetime

class DifficultyMeasurement:
    """
    Difficulty measurement pipeline using vLLM for sampling-based evaluation.
    Samples each question 20 times with high diversity to measure difficulty.
    """
    
    def __init__(self, model_path: str, 
                 tensor_parallel_size: int = 2, 
                 gpu_memory_utilization: float = 0.9,
                 max_tokens: int = 512,
                 num_samples: int = 20,
                 batch_size: int = 8):
        """
        Initialize the difficulty measurement pipeline.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_tokens: Maximum tokens for generation
            num_samples: Number of samples per question
            batch_size: Number of questions to process in each batch
        """
        self.model_path = model_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        print(f"Initializing difficulty measurement with model: {model_path}")
        print(f"Batch size: {batch_size}, Samples per question: {num_samples}")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Configure sampling parameters for high diversity (no n parameter for batch processing)
        self.sampling_params = SamplingParams(
            temperature=1.2,     # High temperature for diversity
            top_p=0.9,          # Higher top_p for diversity
            top_k=50,           # Higher top_k for diversity
            max_tokens=max_tokens,
            seed=None           # Random seed for each generation
        )

        # Initialize vLLM engine
        print("Initializing vLLM engine...")
        self.llm = LLM(
            model=model_path, 
            dtype="float16", 
            tensor_parallel_size=tensor_parallel_size, 
            gpu_memory_utilization=gpu_memory_utilization, 
            max_model_len=max_tokens + 1024,  # Add some buffer
            pipeline_parallel_size=1,
            trust_remote_code=True
        )
        
        print("Difficulty measurement pipeline initialized successfully!")

    def normalize_answer(self, s: str) -> str:
        """Lowercase, remove punctuation, articles and extra spaces."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
            return "".join(ch if ch not in exclude else " " for ch in text)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def extract_after_think(self, text):
        """Extract content after <think> tag"""
        if not text:
            return ""
        
        if "</think>" in text:
            parts = text.split("</think>", 1)
            result = parts[1].strip()
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            if lines:
                return lines[0]
            return result
        
        return text.strip()

    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction exactly matches ground truth (normalized).
        Returns True for EM=1, False otherwise.
        """
        pred_norm = self.normalize_answer(prediction).strip()
        gt_norm = self.normalize_answer(ground_truth).strip()
        return pred_norm == gt_norm

    def _convert_raw_prompt_to_messages(self, raw_prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert raw_prompt format to chat messages format.
        """
        return raw_prompt
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply chat template to messages.
        """
        try:
            # Try with enable_thinking first (for thinking models)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except:
            # Fallback without enable_thinking
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text
    
    def _apply_chat_template_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Apply chat template to a batch of messages.
        
        Args:
            batch_messages: List of message lists
            
        Returns:
            List of formatted text strings
        """
        batch_texts = []
        for msgs in batch_messages:
            try:
                # Try with enable_thinking first (for thinking models)
                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except:
                # Fallback without enable_thinking
                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            batch_texts.append(text)
        return batch_texts
    
    def measure_batch_difficulty(self, batch_items: List[Dict[str, Any]]) -> List[int]:
        """
        Measure the difficulty of a batch of questions by sampling multiple times.
        
        Args:
            batch_items: List of question items with raw_prompt and golden_answer
            
        Returns:
            List of difficulty scores (number of correct answers out of num_samples)
        """
        # Prepare batch data
        valid_items = []
        batch_messages = []
        
        for item in batch_items:
            raw_prompt = item.get("raw_prompt", [])
            golden_answer = item.get("golden_answer", "")
            
            if raw_prompt and golden_answer:
                messages = self._convert_raw_prompt_to_messages(raw_prompt)
                batch_messages.append(messages)
                valid_items.append(item)
        
        if not batch_messages:
            return [0] * len(batch_items)
        
        # Apply chat template to batch
        batch_texts = self._apply_chat_template_batch(batch_messages)
        
        # Create repeated prompts for multiple sampling
        repeated_texts = []
        item_indices = []
        for i, text in enumerate(batch_texts):
            for _ in range(self.num_samples):
                repeated_texts.append(text)
                item_indices.append(i)
        
        try:
            # Generate all samples at once
            outputs = self.llm.generate(repeated_texts, self.sampling_params)
            
            # Group outputs by original question
            difficulty_scores = [0] * len(valid_items)
            
            for output_idx, output in enumerate(outputs):
                if output.outputs:
                    item_idx = item_indices[output_idx]
                    generated_text = output.outputs[0].text.strip()
                    clean_answer = self.extract_after_think(generated_text).strip()
                    
                    # Check exact match
                    golden_answer = valid_items[item_idx].get("golden_answer", "")
                    if self.exact_match(clean_answer, golden_answer):
                        difficulty_scores[item_idx] += 1
            
            # Map back to original batch (including invalid items)
            final_scores = []
            valid_idx = 0
            for item in batch_items:
                if item.get("raw_prompt") and item.get("golden_answer"):
                    final_scores.append(difficulty_scores[valid_idx])
                    valid_idx += 1
                else:
                    final_scores.append(0)
            
            return final_scores
            
        except Exception as e:
            print(f"Error generating batch samples: {e}")
            return [0] * len(batch_items)
    
    def process_all_questions(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all questions in batches and measure their difficulty.
        
        Args:
            items: List of question items
            
        Returns:
            List of items with added difficulty scores
        """
        results = []
        total_items = len(items)
        
        # Process in batches
        for batch_start in tqdm(range(0, total_items, self.batch_size), 
                               desc="Processing batches", 
                               total=(total_items + self.batch_size - 1) // self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_items)
            batch_items = items[batch_start:batch_end]
            
            try:
                # Measure difficulty for the entire batch
                difficulty_scores = self.measure_batch_difficulty(batch_items)
                
                # Create result items with all original data plus difficulty
                for item, difficulty_score in zip(batch_items, difficulty_scores):
                    result = item.copy()
                    result["difficulty"] = difficulty_score
                    results.append(result)
                
                # Print progress every few batches
                if (batch_end) % (self.batch_size * 5) == 0 or batch_end == total_items:
                    avg_difficulty = sum(r["difficulty"] for r in results) / len(results) if results else 0
                    print(f"Processed {batch_end}/{total_items} questions. "
                          f"Average difficulty: {avg_difficulty:.2f}/{self.num_samples}")
                
            except KeyboardInterrupt:
                print(f"\nInterrupted at batch {batch_end}/{total_items}")
                raise
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Add items with difficulty 0 on error
                for item in batch_items:
                    result = item.copy()
                    result["difficulty"] = 0
                    results.append(result)
        
        return results
    
    def run(self, input_file: str, 
            output_file: str = None, 
            test_num: int = -1):
        """
        Main pipeline execution.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (auto-generated if None)
            test_num: Number of questions to test (-1 for all)
        """
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = os.path.basename(input_file).replace('.json', '')
            output_file = f'data/difficulty_{input_name}_{timestamp}.json'
        
        print(f"Loading data from {input_file}...")
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if test_num > 0:
            data = data[:test_num]
        
        print(f"Loaded {len(data)} questions")
        print(f"Will sample each question {self.num_samples} times")
        
        # Process all questions
        print("Starting difficulty measurement...")
        results = self.process_all_questions(data)
        
        # Calculate statistics
        difficulty_scores = [r["difficulty"] for r in results]
        total_questions = len(results)
        avg_difficulty = sum(difficulty_scores) / total_questions if total_questions > 0 else 0
        
        # Create difficulty distribution
        difficulty_distribution = Counter(difficulty_scores)
        
        # Statistics
        stats = {
            "input_file": input_file,
            "model_path": self.model_path,
            "total_questions": total_questions,
            "samples_per_question": self.num_samples,
            "average_difficulty": avg_difficulty,
            "difficulty_distribution": dict(difficulty_distribution),
            "sampling_params": {
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
                "top_k": self.sampling_params.top_k,
                "max_tokens": self.sampling_params.max_tokens
            }
        }
        
        print("\n" + "="*60)
        print("Difficulty Measurement Results:")
        print(f"Total questions: {total_questions}")
        print(f"Average difficulty: {avg_difficulty:.2f}/{self.num_samples}")
        print(f"Difficulty distribution:")
        for score, count in sorted(difficulty_distribution.items()):
            percentage = (count / total_questions) * 100
            print(f"  {score}/{self.num_samples} correct: {count} questions ({percentage:.1f}%)")
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_file}")
        print(f"Statistics saved to: {stats_file}")
        print("="*60)
        
        return results


if __name__ == "__main__":
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    # Model configuration
    model_path = "/scratch/models/Qwen3-30B-A3B-Instruct-2507"
    
    import argparse
    parser = argparse.ArgumentParser(description='Difficulty Measurement Pipeline')
    parser.add_argument('--input_path', '-i', required=False, 
                       help='Path to input JSON file')
    parser.add_argument('--output_path', '-o', required=False,
                       help='Path to output JSON file')
    parser.add_argument('--test_num', '-n', type=int, default=-1,
                       help='Number of questions to test (-1 for all)')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                       help='Batch size for processing questions')
    parser.add_argument('--num_samples', '-s', type=int, default=20,
                       help='Number of samples per question')
    args = parser.parse_args()


    input_path = args.input_path
    output_path = args.output_path
    test_num = args.test_num
    num_samples = args.num_samples
    batch_size = args.batch_size

    # Create difficulty measurement pipeline
    difficulty_pipeline = DifficultyMeasurement(
        model_path=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_tokens=8200,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Run difficulty measurement
    print("Starting Difficulty Measurement Pipeline...")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print("="*80)
    
    results = difficulty_pipeline.run(
        input_path, 
        output_path,
        test_num=args.test_num
    )
    
    print("\nDifficulty measurement completed successfully!")
    print(f"Processed {len(results)} questions")