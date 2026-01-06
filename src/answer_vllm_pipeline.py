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

class VLLMAnswerPipeline:
    """
    Answer-only pipeline using vLLM for fast batch inference.
    Uses the same data format as answer_only_pipeline.py but with vLLM backend.
    """
    
    def __init__(self, model_path: str, 
                 tensor_parallel_size: int = 2, 
                 gpu_memory_utilization: float = 0.9,
                 max_tokens: int = 6044,
                 max_output_tokens: int = 512):
        """
        Initialize the vLLM answer pipeline.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_path = model_path
        print(f"Initializing vLLM pipeline with model: {model_path}")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_output_tokens)

        # Initialize vLLM engine
        print("Initializing vLLM engine...")
        self.llm = LLM(
                    model=model_path, 
                    dtype="bfloat16", 
                    tensor_parallel_size=tensor_parallel_size, 
                    gpu_memory_utilization=gpu_memory_utilization, 
                    max_model_len=max_tokens,
                    pipeline_parallel_size=1,
                )
        
        print("vLLM pipeline initialized successfully!")

    def normalize_answer(self, s: str) -> str:
        """Lowercase, remove punctuation, articles and extra spaces."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
            return "".join(ch if ch not in exclude else " " for ch in text)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def extract_after_think(self,text):
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


    def f1_score(self, prediction: str, ground_truths: Union[str, List[str]]) -> Tuple[float, float, float]:
        """
        Compute token-level (F1, precision, recall) between a prediction and one or more ground truths.
        If the normalized prediction exactly equals any normalized ground truth, returns (1,1,1) immediately.
        Otherwise falls back to token-level scoring.
        """
        pred = self.normalize_answer(prediction).strip()

        if isinstance(ground_truths, str):
            gts = [self.normalize_answer(ground_truths)]
        else:
            gts = [self.normalize_answer(gt) for gt in ground_truths]

        if pred in gts:
            return [1.0, 1.0, 1.0, 1.0]

        best_f1 = best_prec = best_rec = 0.0
        
        for gt in gts:
            if gt in {"yes", "no", "noanswer"}:
                continue

            pred_toks = pred.split()
            gt_toks   = gt.split()
            common    = Counter(pred_toks) & Counter(gt_toks)
            same      = sum(common.values())
            if same == 0:
                continue

            prec = same / len(pred_toks)
            rec  = same / len(gt_toks)
            f1   = 2 * prec * rec / (prec + rec)

            if f1 > best_f1:
                best_f1, best_prec, best_rec = f1, prec, rec
        if pred in gts:
            best_rec = 1.0
        else:
            best_rec = 0.0
        return [best_f1, best_prec, best_rec, 0.0]



    def _convert_raw_prompt_to_messages(self, raw_prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert raw_prompt format to chat messages format.
        
        Args:
            raw_prompt: List of message dicts with 'role' and 'content'
            
        Returns:
            Chat messages in the format expected by tokenizer
        """
        return raw_prompt
    
    def _apply_chat_template_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Apply chat template to a batch of messages.
        
        Args:
            batch_messages: List of message lists
            
        Returns:
            List of formatted text strings
        """
        batch_texts = []
        enable_thinking = False
        for msgs in batch_messages:
            try:
                if 'Think' in model_path or 'think' in model_path:
                    enable_thinking = True

                # Check token count before applying chat template
                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                
                # Count tokens
                token_count = len(self.tokenizer.encode(text))
                if token_count > 7000:
                    print(f"WARNING: Message exceeds 7000 tokens ({token_count} tokens), skipping inference")
                    print(f"Message: {msgs}...")  # Print a snippet of the message
                    batch_texts.append(text[:4000])
                    continue                    
                
            except:
                # Fallback without enable_thinking
                continue
            batch_texts.append(text)
        return batch_texts
    
    def process_batch(self, items: List[Dict[str, Any]], batch_size: int = 64, checkpoint_file: str = None) -> List[Dict[str, Any]]:
        """
        Process items in batches using vLLM.
        
        Args:
            items: List of data items containing raw_prompt
            batch_size: Batch size for processing
            checkpoint_file: Path to checkpoint file for resuming
            
        Returns:
            List of items with generated answers
        """
        results = []
        answers = []
        start_idx = 0
        EM = 0
        
        # Load checkpoint if exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}...")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"Resumed from item {start_idx}")
        
        total_items = len(items)
        
        for batch_start in tqdm(range(start_idx, total_items, batch_size), 
                               desc="Processing batches", 
                               total=(total_items - start_idx) // batch_size + 1):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items[batch_start:batch_end]
            
            # Prepare batch messages
            batch_messages = []
            valid_indices = []
            
            for i, item in enumerate(batch_items):
                raw_prompt = item.get("raw_prompt", [])
                if raw_prompt:
                    messages = self._convert_raw_prompt_to_messages(raw_prompt)
                    batch_messages.append(messages)
                    valid_indices.append(i)
                else:
                    print(f"Warning: raw_prompt not found in item {batch_start + i}")


            if not batch_messages:
                continue
            
            try:
                # Apply chat template to batch
                batch_texts = self._apply_chat_template_batch(batch_messages)
                
                # Generate batch
                print(f"Generating batch {batch_start}-{batch_end}...")


                outputs = self.llm.generate(batch_texts, self.sampling_params)
                
                # Process results
                for i, valid_idx in enumerate(valid_indices):
                    item = batch_items[valid_idx]
                    result = {}
                    
                    # Extract generated text
                    if i < len(outputs):
                        generated_text = outputs[i].outputs[0].text.strip()

                        clean_answer = self.extract_after_think(generated_text).strip()


                        result["question"] = item.get("question", "")
                        golden_answer = item.get("golden_answer", "")
                        result['wrong_answer'] = item.get("wrong_answer", "")
                        result['golden_answer'] = golden_answer
                        result["pred"] = clean_answer

                        if clean_answer == golden_answer.strip():
                            EM += 1
                    else:
                        result["question"] = item.get("question", "")
                        result['golden_answer'] = item.get("golden_answer", "")
                        result["pred"] = ""
                    
                    results.append(result)
                    
            except KeyboardInterrupt:
                print(f"\nInterrupted at batch {batch_end}/{total_items}")
                if checkpoint_file:
                    print(f"Results saved to checkpoint: {checkpoint_file}")
                raise

        return results, EM
    
    def run(self, input_file: str, 
            output_file: str = None, 
            batch_size: int = 64, 
            use_checkpoint: bool = True,
            test_num: int = -1):
        """
        Main pipeline execution.
        
        Args:
            input_file: Path to input JSON file with raw_prompt data
            output_file: Path to output JSON file to save results (if None, auto-generates with timestamp)
            batch_size: Batch size for processing
            use_checkpoint: Whether to save checkpoints for resuming
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'data/training_data1/vllm_answers_{timestamp}.json'
        
        print(f"Loading data from {input_file}...")
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if test_num > 0:
            data = data[0:test_num]
        print(f"Loaded {len(data)} items")
        
        # Checkpoint file for resuming
        checkpoint_file = output_file.replace('.json', '_vllm_checkpoint.json') if use_checkpoint else None
        
        # Process all items
        print("Generating answers with vLLM...")
        results, EM = self.process_batch(data, batch_size=batch_size, checkpoint_file=checkpoint_file)

        # Calculate metrics for all results
        for result in results:
            pred = result.get("pred", "")
            golden_answer = result.get("golden_answer", "")
            f1, prec, rec, em = self.f1_score(pred, golden_answer)
            result["f1"] = f1
            result["precision"] = prec
            result["recall"] = rec
            result["em"] = em

        # Calculate F1 scores against wrong_answer for comparison
        if any("wrong_answer" in r for r in results):
            for result in results:
                pred = result.get("pred", "")
                wrong_answer = result.get("wrong_answer", "")
                if wrong_answer:
                    f1_wrong, prec_wrong, rec_wrong, em_wrong = self.f1_score(pred, wrong_answer)
                    result["f1_wrong"] = f1_wrong
                    result["precision_wrong"] = prec_wrong
                    result["recall_wrong"] = rec_wrong
                    result["em_wrong"] = em_wrong


        # Calculate and save metrics statistics
        metrics_stats = {
            "input_path": input_file,
            "model_path": self.model_path,
            "total_samples": len(results),
            "f1_mean": sum(r.get("f1", 0) for r in results) / len(results) if results else 0,
            "precision_mean": sum(r.get("precision", 0) for r in results) / len(results) if results else 0,
            "recall_mean": sum(r.get("recall", 0) for r in results) / len(results) if results else 0,
            "em_mean": sum(r.get("em", 0) for r in results) / len(results) if results else 0,
            "f1_wrong_mean": sum(r.get("f1_wrong", 0) for r in results if "f1_wrong" in r) / len([r for r in results if "f1_wrong" in r]) if results else 0,
            "precision_wrong_mean": sum(r.get("precision_wrong", 0) for r in results if "precision_wrong" in r) / len([r for r in results if "precision_wrong" in r]) if results else 0,
            "recall_wrong_mean": sum(r.get("recall_wrong", 0) for r in results if "recall_wrong" in r) / len([r for r in results if "recall_wrong" in r]) if results else 0,
            "em_wrong_mean": sum(r.get("em_wrong", 0) for r in results if "em_wrong" in r) / len([r for r in results if "em_wrong" in r]) if results else 0
        }




        print(f"EM: {EM}")
        # Save final results

            
        
        # Clean up checkpoint
        if checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"Processing complete! Generated answers for {len(results)} items")
        print(f"Output file: {output_file}")
        
        return results, EM, metrics_stats


if __name__ == "__main__":
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path1 = "/scratch/Inference_materials/pretrained_model_Instruct/Llama-3.1-8B-Instruct"
    model_path2 = 'openrlhf/poisoned_qwen_4B_round3'
    model_list = [model_path1, model_path2]

    input_path1 = 'data/prompt_data/counterfact/counterfact_30B_poisoned_musique_test_converted.json'
    input_path2 = 'data/prompt_data/counterfact/counterfact_30B_poisoned_hotpotqa_test_converted.json'
    input_path3 = 'data/prompt_data/counterfact/counterfact_30B_poisoned_2wiki_test_converted.json'
    input_path4 = 'data/prompt_data/counterfact/counterfact_30B_poisoned_bamboogle_test_converted.json'
    input_list = [input_path1, input_path2, input_path3, input_path4]

    result_metrics = []
    
    data_type = input_path1.split('/')[-2]
    input_name = os.path.basename(input_path1).replace('.json', '')

    metrics_output_path = f'output/{timestamp}_{data_type}/All_final_{input_name}_vllm_metrics.json'
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    max_output_tokens = None

    for model_path in model_list:
        if 'think' in model_path or 'Think' in model_path:
            max_output_tokens = 8192
            max_tokens = 16384
        else:
            max_output_tokens = 64
            max_tokens = 8192
        model_name = model_path.split('/')[-1]

        pipeline = VLLMAnswerPipeline(
            model_path=model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
        )

        for input_path in input_list:
            input_name = os.path.basename(input_path1).replace('.json', '')


            output_path = f'output/{timestamp}/{model_name}_{timestamp}/{input_name}_vllm.json'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)


            # Create and run pipeline
            print("Starting vLLM Answer Pipeline (Compatible Mode)...")
            print(f"Model: {model_path}")
            print(f"Input: {input_path}")
            print(f"Output: {output_path}")
            print("="*80)
            

            
            results, EM, metrics_stats = pipeline.run(
                input_path, 
                output_path, 
                batch_size=1000,
                use_checkpoint=False,
            )
            result_metrics.append(EM)
            print("\n" + "="*80)
            print("vLLM Pipeline execution completed successfully!")
            print(f"Generated {len(results)} answers")

            with open(metrics_output_path, 'a', encoding='utf-8') as f:
                json.dump(metrics_stats, f, ensure_ascii=False, indent=2)
                f.write('\n')
            print(f"Metrics saved to {metrics_output_path}")

            

            id = 0
            return_results = []
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    result["id"] = id
                    return_results.append(result)
                    id += 1
                json.dump(return_results, f, ensure_ascii=False, indent=2)
            print(f"Saving results to {output_path}...")
            time.sleep(10)
        del pipeline
        time.sleep(10)
        print("Result Metrics (EM), poisoned and pure:", result_metrics)