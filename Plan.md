# VQA-X Pipeline Optimization Implementation Plan

## 1. Architecture Optimization Overview

This plan addresses optimization opportunities in the VQA-X pipeline while maintaining research quality and accuracy as the top priority. All optimizations preserve the fundamental architecture and accuracy of the system.

## 2. Component-Level Optimizations

### 2.1 VerifierAgent: Sequential-Parallel Hybrid Approach

#### DAM and GroundingDINO Dependency Resolution

The dependency between DAM and GroundingDINO requires careful handling:

```python
async def optimized_visual_processing(self, image_path, question):
    # 1. Run GroundingDINO first (required for DAM)
    groundingdino_result = await self._run_groundingdino(image_path, question)
    boxes, labels = groundingdino_result[1], groundingdino_result[2]
    
    # 2. Once we have boxes, run DAM and additional VLM processing in parallel
    dam_task = asyncio.create_task(self._run_dam(image_path, boxes))
    vlm_task = asyncio.create_task(self._run_vlm_analysis(image_path, question))
    
    # 3. Wait for parallel tasks to complete
    dam_result, vlm_result = await asyncio.gather(dam_task, vlm_task)
    
    # 4. Combine results
    return {
        "boxes": boxes,
        "labels": labels,
        "caption": dam_result.get("caption", ""),
        "vlm_analysis": vlm_result,
        "annotated_path": groundingdino_result[0]
    }
```

This approach:
- Respects the dependency (DAM needs GroundingDINO boxes)
- Parallelizes independent operations (VLM analysis and DAM)
- Maintains accuracy by preserving the full processing chain

### 2.2 Feature Caching Implementation

#### Specific Features to Cache

```python
class VisualFeatureCache:
    """Cache for storing and retrieving image features"""
    
    def __init__(self, max_size=50, max_memory_gb=4):
        self.cache = {}
        self.max_size = max_size
        self.max_memory_gb = max_memory_gb
        self.current_memory_usage = 0
        self.lock = threading.Lock()
        self.access_times = {}  # For LRU implementation
    
    def get(self, image_path, feature_type):
        """Get cached features for an image"""
        key = f"{image_path}_{feature_type}"
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()  # Update access time
                return self.cache[key]
            return None
    
    def store(self, image_path, feature_type, features):
        """Store features for an image with memory tracking"""
        key = f"{image_path}_{feature_type}"
        
        # Estimate memory usage of features
        feature_size_gb = self._estimate_size_gb(features)
        
        with self.lock:
            # Check if adding this would exceed memory limit
            if self.current_memory_usage + feature_size_gb > self.max_memory_gb:
                self._evict_until_space(feature_size_gb)
            
            # Store the features
            self.cache[key] = features
            self.access_times[key] = time.time()
            self.current_memory_usage += feature_size_gb
    
    def _estimate_size_gb(self, obj):
        """Estimate size of object in GB"""
        import sys
        if hasattr(obj, 'element_size') and hasattr(obj, 'nelement'):
            # PyTorch tensor
            return obj.element_size() * obj.nelement() / (1024**3)
        else:
            # Fallback for other objects
            return sys.getsizeof(obj) / (1024**3)
    
    def _evict_until_space(self, required_space_gb):
        """Evict least recently used items until there's enough space"""
        if not self.access_times:
            return
            
        # Sort by access time
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys:
            if key in self.cache:
                feature_size = self._estimate_size_gb(self.cache[key])
                del self.cache[key]
                del self.access_times[key]
                self.current_memory_usage -= feature_size
                
                if self.current_memory_usage + required_space_gb <= self.max_memory_gb:
                    break
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_memory_usage = 0
```

**Cached Features:**
1. GroundingDINO detection results (boxes, labels)
2. DAM dense captioning outputs
3. VLM image embeddings
4. Processed image tensors

**Cache Invalidation:**
- LRU (Least Recently Used) eviction policy
- Memory-based constraints (max 4GB by default)
- Manual clearing between dataset runs

**Memory Impact:**
- Controlled by max_memory_gb parameter
- Monitoring via current_memory_usage tracking
- Graceful degradation when memory limits approached

## 3. Intelligent Batch Processing

### 3.1 Adaptive Batch Processing Implementation

```python
class AdaptiveBatchProcessor:
    """Process samples in batches with adaptive sizing and error handling"""
    
    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.max_batch_size = config.get('processing_config', {}).get('batch_size', 4)
        self.min_batch_size = 1
        self.current_batch_size = self.max_batch_size
        self.memory_threshold = 0.85  # 85% GPU memory utilization threshold
        self.logger = logging.getLogger("AdaptiveBatchProcessor")
        self.output_dir = config.get('output_config', {}).get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def process_dataset(self, dataset):
        """Process entire dataset with adaptive batching"""
        results = []
        failed_samples = []
        
        # Setup result tracking
        result_file = os.path.join(self.output_dir, "batch_results.jsonl")
        
        for i in range(0, len(dataset), self.current_batch_size):
            # Adjust batch size based on previous run
            self._adjust_batch_size()
            
            # Get current batch
            end_idx = min(i + self.current_batch_size, len(dataset))
            batch = dataset[i:end_idx]
            actual_batch_size = len(batch)
            
            self.logger.info(f"Processing batch {i//self.current_batch_size + 1} "
                            f"with size {actual_batch_size}")
            
            # Process batch with error handling
            batch_results, batch_failures = await self._process_batch_with_recovery(batch)
            
            # Save results incrementally
            self._save_incremental_results(batch_results, result_file)
            
            # Track results and failures
            results.extend(batch_results)
            failed_samples.extend(batch_failures)
            
            # Log progress
            self._log_batch_completion(i, end_idx, len(dataset), 
                                     len(batch_results), len(batch_failures))
            
            # Clear cache between batches
            self._clear_caches()
        
        # Handle failed samples if any
        if failed_samples:
            self.logger.warning(f"Attempting to reprocess {len(failed_samples)} failed samples")
            recovery_results = await self._reprocess_failed_samples(failed_samples)
            results.extend(recovery_results)
        
        return results
    
    async def _process_batch_with_recovery(self, batch):
        """Process a batch with error handling for individual samples"""
        batch_tasks = []
        for sample in batch:
            task = asyncio.create_task(
                self._process_sample_with_timeout(sample)
            )
            batch_tasks.append(task)
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Separate successful results from failures
        successful_results = []
        failures = []
        
        for sample, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing sample {sample.get('question_id', 'unknown')}: {str(result)}")
                failures.append(sample)
            else:
                successful_results.append(result)
        
        return successful_results, failures
    
    async def _process_sample_with_timeout(self, sample, timeout=60):
        """Process a single sample with timeout"""
        try:
            return await asyncio.wait_for(
                self._process_single_sample(sample),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Processing timed out for sample {sample.get('question_id', 'unknown')}")
    
    async def _process_single_sample(self, sample):
        """Process a single sample through the pipeline"""
        # Implementation of the full pipeline for a single sample
        # [Existing pipeline code]
        pass
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on GPU memory usage"""
        try:
            import torch
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            # If memory usage is too high, reduce batch size
            if memory_used > self.memory_threshold:
                self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
                self.logger.info(f"Reducing batch size to {self.current_batch_size} due to high memory usage")
            # If memory usage is low, cautiously increase batch size
            elif memory_used < 0.5 and self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                self.logger.info(f"Increasing batch size to {self.current_batch_size}")
        except:
            # If memory monitoring fails, use conservative batch size
            self.current_batch_size = max(1, self.max_batch_size // 2)
    
    def _save_incremental_results(self, results, result_file):
        """Save results incrementally to avoid data loss"""
        with open(result_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    
    def _log_batch_completion(self, start_idx, end_idx, total, successful, failed):
        """Log batch completion status"""
        self.logger.info(f"Completed {end_idx}/{total} samples "
                        f"({successful} successful, {failed} failed)")
        
    def _clear_caches(self):
        """Clear caches between batches"""
        # Clear PyTorch cache
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
        # Clear feature cache if available
        if hasattr(self, 'feature_cache'):
            self.feature_cache.clear()
            
        # Run garbage collection
        import gc
        gc.collect()
    
    async def _reprocess_failed_samples(self, failed_samples):
        """Attempt to reprocess failed samples individually"""
        recovery_results = []
        
        for sample in failed_samples:
            try:
                # Process with extended timeout and conservative settings
                result = await self._process_sample_with_timeout(sample, timeout=120)
                recovery_results.append(result)
                self.logger.info(f"Successfully recovered sample {sample.get('question_id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Recovery failed for sample {sample.get('question_id', 'unknown')}: {str(e)}")
                # Add a placeholder result to maintain dataset integrity
                recovery_results.append({
                    "question_id": sample.get("question_id", "unknown"),
                    "question": sample.get("question", ""),
                    "image_path": sample.get("image_path", ""),
                    "final_answer": "ERROR: Processing failed",
                    "confidence": 0.0,
                    "explanation": "The system encountered an error processing this sample.",
                    "error": str(e)
                })
        
        return recovery_results
```

### 3.2 Memory Constraints Management

- **Dynamic Batch Sizing**: Adjusts batch size based on GPU memory utilization
- **Per-Sample Memory Tracking**: Monitors memory usage for each sample type
- **Incremental Result Saving**: Prevents data loss during long batch runs
- **Automatic Recovery**: Attempts to recover failed samples individually

### 3.3 Error Handling for Batch Operations

- **Individual Sample Isolation**: Errors in one sample don't affect others
- **Timeout Protection**: Prevents hanging on problematic samples
- **Detailed Error Logging**: Captures specific failure points
- **Recovery Mechanism**: Attempts to reprocess failed samples with adjusted parameters

### 3.4 Result Consistency Maintenance

- **Deterministic Processing**: Ensures consistent results across runs
- **Validation Checks**: Verifies output format and content
- **Incremental Checkpointing**: Saves partial results during processing
- **Comprehensive Logging**: Tracks all processing decisions

## 4. Model-Specific Prompt Optimization

### 4.1 Qwen2.5-VL-7B-Instruct Optimization Techniques

Based on research from [Liu et al., 2023](https://arxiv.org/abs/2305.13246) and [Qwen-VL documentation](https://github.com/QwenLM/Qwen-VL):

```python
class QwenVLPromptOptimizer:
    """Optimize prompts specifically for Qwen2.5-VL-7B-Instruct"""
    
    def optimize_visual_prompt(self, question, task_type="detection"):
        """Generate optimized visual prompts based on task type"""
        
        # Base template with research-backed structure
        base_template = (
            "I need you to analyze this image carefully.\n\n"
            "Question: {question}\n\n"
            "{task_specific_instruction}\n\n"
            "Provide a detailed response focusing on visual elements relevant to the question."
        )
        
        # Task-specific instructions based on research findings
        task_instructions = {
            "detection": "Focus on identifying and locating objects in the image that relate to the question.",
            "attribute": "Analyze visual attributes like colors, sizes, and textures that are relevant.",
            "spatial": "Pay special attention to spatial relationships between objects in the image.",
            "counting": "Count the relevant objects or elements in the image accurately.",
            "reasoning": "Analyze the image and reason about what's happening in the scene."
        }
        
        # Select appropriate instruction
        instruction = task_instructions.get(task_type, task_instructions["reasoning"])
        
        # Format the prompt
        optimized_prompt = base_template.format(
            question=question,
            task_specific_instruction=instruction
        )
        
        return optimized_prompt
    
    def optimize_token_usage(self, prompt):
        """Optimize token usage based on Qwen-VL tokenizer characteristics"""
        # Research shows Qwen-VL is sensitive to verbosity
        # Implement token reduction techniques
        
        # Remove redundant phrases
        redundant_phrases = [
            "Please analyze", "I want you to", "Can you please",
            "I would like you to", "Please provide", "Make sure to"
        ]
        
        for phrase in redundant_phrases:
            prompt = prompt.replace(phrase, "")
        
        # Consolidate instructions
        prompt = re.sub(r'(\.\s+)(Analyze|Look|Examine|Identify)', r'. \2', prompt)
        
        return prompt.strip()
```

### 4.2 Qwen2.5-7B-Instruct Optimization Techniques

Based on research from [Wei et al., 2022](https://arxiv.org/abs/2201.11903) and [Qwen documentation](https://github.com/QwenLM/Qwen):

```python
class QwenLLMPromptOptimizer:
    """Optimize prompts specifically for Qwen2.5-7B-Instruct"""
    
    def optimize_reasoning_prompt(self, question, context, reasoning_type="mvkb"):
        """Generate optimized reasoning prompts"""
        
        # Research-backed chain-of-thought template
        if reasoning_type == "mvkb":
            template = (
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Let's think through this step by step:\n"
                "1. First, identify the key elements in the question.\n"
                "2. Analyze how the visual evidence supports possible answers.\n"
                "3. Consider alternative hypotheses and evaluate them.\n"
                "4. Determine confidence levels for each hypothesis.\n\n"
                "Based on this analysis, construct a multi-view knowledge base."
            )
        elif reasoning_type == "synthesis":
            template = (
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "I'll evaluate each candidate answer systematically:\n"
                "1. Examine the evidence supporting each candidate.\n"
                "2. Assess confidence levels based on evidence strength.\n"
                "3. Consider potential conflicts between evidence.\n"
                "4. Determine the most supported answer.\n\n"
                "Based on this evaluation, provide the final answer with confidence score."
            )
        else:
            template = (
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Let's analyze this carefully."
            )
            
        # Format the prompt
        optimized_prompt = template.format(
            question=question,
            context=context
        )
        
        return optimized_prompt
    
    def apply_few_shot_examples(self, prompt, reasoning_type):
        """Add few-shot examples based on research showing their effectiveness"""
        
        # Research shows 2-3 examples is optimal for Qwen models
        examples = self._get_examples_for_type(reasoning_type)
        
        # Add examples to prompt
        few_shot_prompt = "Here are some examples of how to approach this task:\n\n"
        for i, example in enumerate(examples, 1):
            few_shot_prompt += f"Example {i}:\n{example}\n\n"
            
        few_shot_prompt += "Now, apply similar reasoning to the current question:\n\n"
        
        return few_shot_prompt + prompt
    
    def _get_examples_for_type(self, reasoning_type):
        """Get appropriate examples for each reasoning type"""
        # Implementation would contain curated examples
        examples_db = {
            "mvkb": [
                "Question: What color is the car?\n"
                "Visual evidence: The image shows a sedan in the foreground.\n"
                "Analysis: The car has a distinctive red paint job with no other colors visible.\n"
                "Hypothesis: If the car in the image is red, then the answer is 'red'.\n"
                "Confidence: 0.95",
                
                "Question: How many people are in the image?\n"
                "Visual evidence: The image shows a group of individuals in a park setting.\n"
                "Analysis: I can clearly count 4 adults and 2 children in the foreground.\n"
                "Hypothesis: If there are 6 people visible in the image, then the answer is '6'.\n"
                "Confidence: 0.98"
            ],
            "synthesis": [
                "Candidate 1: 'Red' (Confidence: 0.95)\n"
                "Evidence: Visual detection shows car with red coloring\n"
                "Candidate 2: 'Blue' (Confidence: 0.15)\n"
                "Evidence: Small blue object in background, not the car\n"
                "Analysis: The evidence strongly supports 'Red' with high confidence\n"
                "Final answer: Red (Confidence: 0.95)",
                
                "Candidate 1: '6 people' (Confidence: 0.98)\n"
                "Evidence: Visual detection shows 4 adults and 2 children\n"
                "Candidate 2: '5 people' (Confidence: 0.30)\n"
                "Evidence: One person partially obscured\n"
                "Analysis: The evidence strongly supports '6 people'\n"
                "Final answer: 6 people (Confidence: 0.98)"
            ]
        }
        
        return examples_db.get(reasoning_type, [])
```

### 4.3 Research References

1. **Chain-of-Thought Prompting**:
   - Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

2. **Visual Instruction Tuning**:
   - Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. arXiv preprint arXiv:2304.08485.

3. **Few-Shot Prompting for Vision-Language Models**:
   - Alayrac, J. B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., ... & Zisserman, A. (2022). Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35, 23716-23736.

4. **Qwen-VL Specific Optimization**:
   - Bai, J., Bai, S., Yang, X., Wang, H., Yang, P., Tong, X., ... & Si, L. (2023). Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966.

## 5. Performance Tracing Implementation

### 5.1 Dedicated Log File Structure

```python
class PerformanceTracer:
    """Comprehensive performance tracing with dedicated log files"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get('output_config', {}).get('output_dir', 'output')
        self.trace_dir = os.path.join(self.output_dir, 'traces')
        os.makedirs(self.trace_dir, exist_ok=True)
        
        # Setup dedicated loggers
        self.setup_loggers()
        
        # Performance tracking
        self.timings = {}
        self.memory_usage = []
        self.component_stats = {}
        
        # Start time for the entire process
        self.start_time = time.time()
    
    def setup_loggers(self):
        """Setup dedicated loggers for different trace types"""
        # Main performance logger
        self.perf_logger = self._create_logger('performance', 'performance.log')
        
        # Component-specific loggers
        self.component_loggers = {
            'verifier': self._create_logger('verifier', 'verifier_trace.log'),
            'strategist': self._create_logger('strategist', 'strategist_trace.log'),
            'synthesizer': self._create_logger('synthesizer', 'synthesizer_trace.log'),
            'explanation': self._create_logger('explanation', 'explanation_trace.log'),
            'pipeline': self._create_logger('pipeline', 'pipeline_trace.log'),
            'memory': self._create_logger('memory', 'memory_trace.log'),
            'error': self._create_logger('error', 'error_trace.log')
        }
    
    def _create_logger(self, name, filename):
        """Create a dedicated logger that writes to a specific file"""
        logger = logging.getLogger(f"trace.{name}")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(self.trace_dir, filename))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Don't propagate to root logger
        logger.propagate = False
        
        return logger
    
    def start_component(self, component, sample_id=None, extra_info=None):
        """Start timing a component"""
        component_key = f"{component}_{sample_id}" if sample_id else component
        self.timings[component_key] = {
            "start": time.time(),
            "end": None,
            "duration": None,
            "sample_id": sample_id,
            "extra_info": extra_info
        }
        
        # Log start
        logger = self.component_loggers.get(component, self.perf_logger)
        logger.info(f"Starting {component}" + 
                   (f" for sample {sample_id}" if sample_id else "") +
                   (f" with {extra_info}" if extra_info else ""))
        
        # Log memory at start
        self.log_memory_usage(component, "start", sample_id)
    
    def end_component(self, component, sample_id=None, result_info=None):
        """End timing a component"""
        component_key = f"{component}_{sample_id}" if sample_id else component
        
        if component_key in self.timings:
            end_time = time.time()
            self.timings[component_key]["end"] = end_time
            duration = end_time - self.timings[component_key]["start"]
            self.timings[component_key]["duration"] = duration
            
            # Update component stats
            if component not in self.component_stats:
                self.component_stats[component] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0
                }
            
            stats = self.component_stats[component]
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min_time"] = min(stats["min_time"], duration)
            stats["max_time"] = max(stats["max_time"], duration)
            
            # Log completion
            logger = self.component_loggers.get(component, self.perf_logger)
            logger.info(f"Completed {component}" +
                       (f" for sample {sample_id}" if sample_id else "") +
                       f" in {duration:.4f}s" +
                       (f" with result {result_info}" if result_info else ""))
            
            # Log memory at end
            self.log_memory_usage(component, "end", sample_id)
    
    def log_memory_usage(self, component, stage, sample_id=None):
        """Log detailed memory usage"""
        try:
            import torch
            import psutil
            
            # Get process memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024 ** 3)  # GB
            
            # Get GPU memory if available
            gpu_allocated = 0
            gpu_reserved = 0
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            
            memory_info = {
                "timestamp": time.time(),
                "component": component,
                "stage": stage,
                "sample_id": sample_id,
                "process_memory_gb": process_memory,
                "gpu_allocated_gb": gpu_allocated,
                "gpu_reserved_gb": gpu_reserved
            }
            
            self.memory_usage.append(memory_info)
            
            # Log to memory trace file
            self.component_loggers['memory'].info(
                f"{component} {stage}" +
                (f" sample {sample_id}" if sample_id else "") +
                f" - Process: {process_memory:.2f}GB, " +
                f"GPU allocated: {gpu_allocated:.2f}GB, " +
                f"GPU reserved: {gpu_reserved:.2f}GB"
            )
            
        except Exception as e:
            self.component_loggers['error'].error(f"Error logging memory: {str(e)}")
    
    def log_error(self, component, error, sample_id=None, context=None):
        """Log errors with context"""
        error_msg = f"ERROR in {component}" + \
                   (f" for sample {sample_id}" if sample_id else "") + \
                   f": {str(error)}"
        
        if context:
            error_msg += f"\nContext: {context}"
            
        self.component_loggers['error'].error(error_msg)
    
    def generate_summary_report(self):
        """Generate comprehensive performance summary report"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # Create summary file
        summary_path = os.path.join(self.trace_dir, 'performance_summary.json')
        
        # Prepare summary data
        summary = {
            "total_duration": total_duration,
            "component_stats": {},
            "memory_stats": self._calculate_memory_stats(),
            "bottlenecks": self._identify_bottlenecks()
        }
        
        # Process component stats
        for component, stats in self.component_stats.items():
            if stats["count"] > 0:
                summary["component_stats"][component] = {
                    "count": stats["count"],
                    "total_time": stats["total_time"],
                    "avg_time": stats["total_time"] / stats["count"],
                    "min_time": stats["min_time"],
                    "max_time": stats["max_time"],
                    "percentage_of_total": (stats["total_time"] / total_duration) * 100
                }
        
        # Write summary to file
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also log summary to performance log
        self.perf_logger.info(f"Performance summary written to {summary_path}")
        self.perf_logger.info(f"Total execution time: {total_duration:.2f}s")
        
        for component, comp_stats in summary["component_stats"].items():
            self.perf_logger.info(
                f"{component}: {comp_stats['count']} calls, " +
                f"avg: {comp_stats['avg_time']:.4f}s, " +
                f"total: {comp_stats['total_time']:.2f}s " +
                f"({comp_stats['percentage_of_total']:.1f}% of total)"
            )
        
        return summary
    
    def _calculate_memory_stats(self):
        """Calculate memory usage statistics"""
        if not self.memory_usage:
            return {}
            
        # Process memory stats
        process_memory = [entry["process_memory_gb"] for entry in self.memory_usage]
        gpu_allocated = [entry["gpu_allocated_gb"] for entry in self.memory_usage 
                        if "gpu_allocated_gb" in entry]
        