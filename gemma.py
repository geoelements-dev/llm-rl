import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Potentially useful for CUDA memory debugging, but can add overhead
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # Example, adjust as needed

import random
import re
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments, # Not strictly used but kept from original
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import warnings

warnings.filterwarnings("ignore", message="You're using a LlamaTokenizerFast tokenizer") # May not be relevant for Gemma
warnings.filterwarnings(
    "ignore", message="We detected that you are passing `past_key_values` as a tuple"
)

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# -----------------------------------------------------------------------------
# Model Selection - Choose your preferred small model
# -----------------------------------------------------------------------------
# TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # OLD
GEMMA_MODEL = "google/gemma-2b-it" # NEW: Using Gemma 2B Instruct
MODEL_NAME = GEMMA_MODEL
print(f"Selected model: {MODEL_NAME}")

# NEW: Add a note about Gemma requirements
print(f"""
------------------------------------------------------------------------------------
IMPORTANT: For Gemma models ({GEMMA_MODEL}), you might need:
1. `pip install transformers>=4.38 accelerate bitsandbytes` (bitsandbytes if using 8-bit/4-bit quantization).
2. Accept the license terms on the Hugging Face model page for {GEMMA_MODEL}.
3. Log in via Hugging Face CLI: `huggingface-cli login` using a token with access to Gemma.
------------------------------------------------------------------------------------
""")


# -----------------------------------------------------------------------------
# Material data & engineering utilities
# -----------------------------------------------------------------------------
MATERIAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "Steel": {"density": 7850, "youngs_modulus": 200e9, "yield_strength": 250e6},
    "Aluminum": {"density": 2700, "youngs_modulus": 70e9, "yield_strength": 200e6},
    "Wood": {"density": 600, "youngs_modulus": 10e9, "yield_strength": 40e6},
    "Carbon_Fiber": {
        "density": 1600,
        "youngs_modulus": 150e9,
        "yield_strength": 1500e6,
    },
}
WEIGHT_PENALTY_COEFFICIENT = 1.0
STRESS_EFFICIENCY_REWARD_SCALE = 150.0 # This was defined but not used in reward calculation, keeping for now.


def evaluate_beam_design(
    design: Dict[str, float],
    env: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    req_keys = ["material", "width", "height", "length"]
    if any(k not in design or design[k] is None for k in req_keys):
        return {"invalid": True, "reason": "missing_keys"}, -50.0
    if design["material"] not in MATERIAL_PROPERTIES:
        return {"invalid": True, "reason": "invalid_material"}, -50.0
    if any(design[k] <= 0 for k in ("width", "height", "length")):
        return {"invalid": True, "reason": "negative_dimensions"}, -50.0

    mat = MATERIAL_PROPERTIES[design["material"]]
    b, h, L = design["width"], design["height"], design["length"]
    P = env["load_P"]
    a = env["load_a_fraction"] * L
    target_rd = env["target_relative_displacement"]

    I = b * h**3 / 12.0
    if I < 1e-12:
        return {"invalid": True, "reason": "zero_moment"}, -50.0

    M_max = P * a * (L - a) / L
    sigma = abs(M_max * (h / 2) / I) if I > 1e-12 else float("inf")
    delta = (
        P * a**2 * (L - a) ** 2 / (3 * mat["youngs_modulus"] * I * L)
        if I > 1e-12 and mat["youngs_modulus"] > 1e-9
        else float("inf")
    )
    rel_disp = delta / L if L > 1e-9 else float("inf")
    weight = mat["density"] * b * h * L

    displacement_failure_penalty = -50.0 if rel_disp > target_rd else 0.0
    
    displacement_reward = 0.0
    if target_rd > 1e-9 and rel_disp <= target_rd:
        utilization = rel_disp / target_rd
        if 0.7 <= utilization <= 0.9:
            displacement_reward = 50.0
        elif utilization < 0.7:
            displacement_reward = 25.0 * (utilization / 0.7)
        else:  # 0.9 < utilization <= 1.0
            displacement_reward = 50.0 * (1.0 - utilization) / 0.1

    stress_failure_penalty = -100.0 if sigma > mat["yield_strength"] else 0.0
    weight_penalty = -WEIGHT_PENALTY_COEFFICIENT * weight * 0.01
    stress_efficiency_reward = 0.0
    if sigma <= mat["yield_strength"] and mat["yield_strength"] > 1e-9:
        utilization_ratio = sigma / mat["yield_strength"]
        stress_efficiency_reward = (utilization_ratio**2) * 15.0 # STRESS_EFFICIENCY_REWARD_SCALE was 150, but 15.0 used here.

    reward = (
        stress_failure_penalty
        + displacement_failure_penalty
        + displacement_reward
        + weight_penalty
        + stress_efficiency_reward
    )
    return {
        "sigma": sigma,
        "rel_disp": rel_disp,
        "weight": weight,
        "utilization_ratio": (
            sigma / mat["yield_strength"] if mat["yield_strength"] > 1e-9 else 0
        ),
        "displacement_utilization": (
            rel_disp / target_rd if target_rd > 1e-9 else 0
        ),
        "invalid": False,
        "reward_components": {
            "stress_failure_penalty": stress_failure_penalty,
            "displacement_failure_penalty": displacement_failure_penalty,
            "displacement_reward": displacement_reward,
            "weight_penalty": weight_penalty,
            "stress_efficiency_reward": stress_efficiency_reward,
        },
    }, reward


PARSE_PATTERNS = [
    { # Pattern expecting Material, Width, Height (Length is fixed from env)
        "material": re.compile(r"Material:\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"Width:\s*([\d.]+)", re.I),
        "height": re.compile(r"Height:\s*([\d.]+)", re.I),
    },
    # Removed the second pattern that included Length, as we enforce length from env.
    # If the model *must* output length, this might need adjustment, but prompt discourages it.
]


def parse_design(text: str):
    for pattern_set in PARSE_PATTERNS:
        out = {}
        success = True
        for k, rgx in pattern_set.items():
            m = rgx.search(text)
            if not m:
                success = False
                break
            val = m.group(1).strip()
            if k == "material":
                match = next(
                    (
                        mat_name
                        for mat_name in MATERIAL_PROPERTIES
                        if mat_name.lower().replace("_", "")
                        == val.lower().replace("_", "")
                        or mat_name.lower() == val.lower()
                    ),
                    None,
                )
                if not match:
                    success = False
                    break
                out[k] = match
            else:
                try:
                    float_val = float(val)
                    if float_val <= 0: # Ensure positive dimensions
                        success = False
                        break
                    out[k] = float_val
                except ValueError:
                    success = False
                    break
        if success and out.get("material") and out.get("width") and out.get("height"):
            return out
    return None


def gen_env():
    return {
        "load_P": random.uniform(1_000.0, 20_000.0),
        "load_a_fraction": random.uniform(0.1, 0.9),
        "target_relative_displacement": random.uniform(5e-4, 3e-3),
        "length": random.uniform(1.0, 5.0),
    }


def check_model_health(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: NaN/inf detected in parameter {name}")
            return False
    return True


# MODIFIED: get_prompt_template to support Gemma
def get_prompt_template(model_name: str):
    materials = ["Aluminum", "Carbon_Fiber", "Steel", "Wood"]
    
    # Common part of the prompt for the design task
    # Note: Using .format(env=env) later for dynamic values
    design_task_description_template = (
        "Design a rectangular beam for these requirements:\n"
        "- Length of beam: {env[length]:.2f} m (This is a fixed input, do not output it.)\n" # Explicitly tell not to output length
        "- Point load: {env[load_P]:.0f} N at position {env[load_a_fraction]:.2f} along length\n"
        "- Max relative displacement: {env[target_relative_displacement]:.4f}\n"
        "- Select one material from this list: {materials_list}\n\n" # Placeholder for shuffled materials
        "Respond with ONE design that has one Material, Width, and Height in this exact format:\n"
        "Material: [Select_a_material_from_list]\nWidth: [width_in_meters]\nHeight: [height_in_meters]\n"
        "Do not include any other explanatory text, introductions, or conclusions." # Added for strictness
    )

    if "gemma" in model_name.lower():
        # Gemma's prompt format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
        # System-like instructions can be part of the first user turn.
        return lambda env: (
            f"<start_of_turn>user\n"
            f"You are a precise engineering assistant. Follow the instructions exactly.\n" # System-like instruction
            f"{design_task_description_template.format(env=env, materials_list=random.sample(materials, len(materials)))}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n" # Model will generate from here
        )
    elif "tinyllama" in model_name.lower(): # Kept for reference
        return lambda env: (
            "<|system|>\nYou are a helpful engineering assistant.\n<|user|>\n"
            f"{design_task_description_template.format(env=env, materials_list=random.sample(materials, len(materials)))}"
            "<|assistant|>\n"
        )
    return lambda env: f"Unsupported model for prompt: {model_name}"


# -----------------------------------------------------------------------------
# Training Monitoring Setup
# -----------------------------------------------------------------------------
class TrainingMonitor:
    def __init__(self, patience_epochs=500, min_success_rate=0.9, min_avg_reward=20.0, 
                 success_rate_window=50, avg_reward_window=100):
        self.patience_epochs = patience_epochs
        self.min_success_rate = min_success_rate
        self.min_avg_reward = min_avg_reward
        self.success_rate_window = success_rate_window
        self.avg_reward_window = avg_reward_window
        
        self.epoch_rewards = []
        self.epoch_success_rates = []
        self.epoch_valid_counts = []
        self.best_rewards = []
        self.reward_components = []
        
        self.best_avg_reward = -float('inf')
        self.epochs_without_improvement = 0
        self.recent_success_rates = deque(maxlen=success_rate_window)
        self.recent_avg_rewards = deque(maxlen=avg_reward_window)
        
    def update(self, avg_reward, success_rate, valid_count, reward_components=None):
        self.epoch_rewards.append(avg_reward)
        self.epoch_success_rates.append(success_rate)
        self.epoch_valid_counts.append(valid_count)
        self.best_rewards.append(max(self.epoch_rewards) if self.epoch_rewards else -float('inf')) # Handle empty list
        if reward_components:
            self.reward_components.append(reward_components)
        
        self.recent_success_rates.append(success_rate)
        self.recent_avg_rewards.append(avg_reward)
        
        current_window_avg_reward = np.mean(list(self.recent_avg_rewards)) if self.recent_avg_rewards else -float('inf')
        if current_window_avg_reward > self.best_avg_reward: # Check against moving average for stability
            self.best_avg_reward = current_window_avg_reward
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def should_stop_early(self):
        reasons = []
        if (len(self.recent_success_rates) >= self.success_rate_window and 
            all(sr >= self.min_success_rate for sr in self.recent_success_rates)):
            reasons.append(f"Success rate >= {self.min_success_rate:.1%} for {self.success_rate_window} epochs")
        
        if (len(self.recent_avg_rewards) >= self.avg_reward_window and 
            all(ar >= self.min_avg_reward for ar in self.recent_avg_rewards)): # Check if all recent rewards meet threshold
            reasons.append(f"Average reward >= {self.min_avg_reward} for {self.avg_reward_window} epochs")
        
        if self.epochs_without_improvement >= self.patience_epochs:
            reasons.append(f"No improvement in avg reward ({self.best_avg_reward:.2f}) for {self.patience_epochs} epochs")
        
        return reasons
    
    def plot_training_progress(self, save_dir):
        if not self.epoch_rewards: # Check if there's data to plot
            print("No data to plot for training progress.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.epoch_rewards) + 1)
        
        axes[0, 0].plot(epochs, self.epoch_rewards, 'b-', label='Avg Reward', alpha=0.7)
        axes[0, 0].plot(epochs, self.best_rewards, 'r-', label='Best Reward Seen', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, [sr * 100 for sr in self.epoch_success_rates], 'g-', linewidth=2)
        axes[0, 1].axhline(y=self.min_success_rate * 100, color='r', linestyle='--', alpha=0.7, label=f'{self.min_success_rate*100}% Target')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 101) # Ensure 100% is visible
        
        axes[1, 0].bar(epochs, self.epoch_valid_counts, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Valid Designs per Batch')
        axes[1, 0].set_title('Valid Designs Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        if self.reward_components and len(self.reward_components) > 0:
            components_data = {comp_name: [] for comp_name in self.reward_components[0].keys()}
            for epoch_components in self.reward_components:
                for comp_name, comp_value in epoch_components.items():
                    components_data[comp_name].append(comp_value)
            
            for comp_name, comp_values in components_data.items():
                axes[1, 1].plot(epochs[:len(comp_values)], comp_values, 
                               label=comp_name.replace('_', ' ').title(), alpha=0.8)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Avg Reward Component Value')
            axes[1, 1].set_title('Avg Reward Components Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            if len(epochs) >= 10:
                window = min(20, len(epochs) // 4 if len(epochs) // 4 > 0 else 10) # Ensure window is at least 1
                moving_avg = np.convolve(self.epoch_rewards, np.ones(window)/window, mode='valid')
                moving_epochs = epochs[window-1:]
                axes[1, 1].plot(epochs, self.epoch_rewards, 'b-', alpha=0.3, label='Raw Avg Reward')
                axes[1, 1].plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'{window}-epoch MA Reward')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].set_title('Reward Moving Average')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        recent_epochs_count = min(100, len(epochs))
        if recent_epochs_count > 0:
            recent_rewards = self.epoch_rewards[-recent_epochs_count:]
            recent_success = self.epoch_success_rates[-recent_epochs_count:]
            plot_epochs = epochs[-recent_epochs_count:]
            
            ax2 = ax.twinx()
            line1 = ax.plot(plot_epochs, recent_rewards, 'b-', linewidth=2, label='Avg Reward')
            line2 = ax2.plot(plot_epochs, [sr * 100 for sr in recent_success], 'g-', linewidth=2, label='Success Rate %')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Reward', color='b')
            ax2.set_ylabel('Success Rate (%)', color='g')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='g')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            ax.set_title(f'Recent Performance (Last {recent_epochs_count} Epochs)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Not enough data for recent performance plot.", ha='center', va='center')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/recent_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training plots saved to {save_dir}")

# -----------------------------------------------------------------------------
# RL Setup
# -----------------------------------------------------------------------------
MAX_PROMPT_LENGTH = 384 # MODIFIED: Gemma might benefit from slightly more prompt context space
MAX_NEW_TOKENS_GEN = 70 # MODIFIED: Slightly more tokens for Gemma's output style

ppo_cfg = PPOConfig(
    learning_rate=1e-7,  # MODIFIED: Gemma might be more sensitive, starting lower than 1e-6
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    cliprange=0.1,      # MODIFIED: More standard PPO cliprange
    cliprange_value=0.1,# MODIFIED: More standard PPO cliprange
    vf_coef=0.1,        # MODIFIED: Slightly higher vf_coef
    max_grad_norm=0.5,  # MODIFIED: More standard max_grad_norm
    ppo_epochs=1,       # MODIFIED: Allow a couple of passes over the data
    # log_with="wandb", # Uncomment if using wandb
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

try:
    print(f"Loading {MODEL_NAME}...")
    # For Gemma, ensure you are logged in if it's a gated model: `huggingface-cli login`
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Gemma's tokenizer might have pad_token set. If not, eos_token is a common fallback.
        # Some Gemma tokenizers might use unk_token as pad_token.
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token: # Fallback for Gemma if no eos and pad
             tokenizer.pad_token = tokenizer.unk_token
        else: # Absolute fallback
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] as pad_token.")

    tokenizer.padding_side = "left" # Crucial for generation

    model_kwargs_common = {"trust_remote_code": True}
    # Gemma models often use use_reentrant=True for gradient checkpointing if False causes issues
    gradient_checkpointing_kwargs = {"use_reentrant": True if "gemma" in MODEL_NAME.lower() else False}


    if DEVICE.type == "cuda":
        model_kwargs_cuda = {
            "torch_dtype": torch.bfloat16, # Gemma works well with bfloat16
            "device_map": "auto", # Handles multi-GPU if available
        }
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME, **model_kwargs_common, **model_kwargs_cuda
        )
        if hasattr(model, "pretrained_model"): # AutoModelForCausalLMWithValueHead wraps the base model
            model.pretrained_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            # Gemma models might not have use_cache on the config of the wrapped model directly
            if hasattr(model.pretrained_model.config, "use_cache"):
                 model.pretrained_model.config.use_cache = False
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False


        model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME,
            **model_kwargs_common,
            **model_kwargs_cuda,
        )
    else:  # CPU or MPS
        model_kwargs_other = {
            "torch_dtype": torch.float32
        }
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME, **model_kwargs_common, **model_kwargs_other
        )
        model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME, **model_kwargs_common, **model_kwargs_other
        )
        model.to(DEVICE)
        model_ref.to(DEVICE)

    print("✓ Models loaded successfully")
    if DEVICE.type == "cuda":
        if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
            print("Model is loaded in 8-bit.")
        if hasattr(model, "hf_device_map"):
            print(f"Model device map: {model.hf_device_map}")
        
        # Determine model dtype
        param_dtype = None
        if hasattr(model, "pretrained_model") and model.pretrained_model is not None:
            try:
                param_dtype = next(model.pretrained_model.parameters()).dtype
            except StopIteration: # No parameters in pretrained_model
                pass
        
        if param_dtype is None and hasattr(model, "parameters"):
            try:
                param_dtype = next(model.parameters()).dtype
            except StopIteration: # No parameters in model
                pass
        
        if param_dtype:
            print(f"Model parameter dtype: {param_dtype}")
        else:
            print("Could not determine model parameter dtype.")


except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have accepted Gemma's terms and are logged in via `huggingface-cli login`.")
    print("Try installing: pip install accelerate transformers[torch] bitsandbytes")
    exit(1)

trainer = PPOTrainer(
    config=ppo_cfg,
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
)
print("✓ PPOTrainer ready")

# -----------------------------------------------------------------------------
# Initial Model Evaluation (Baseline)
# -----------------------------------------------------------------------------
GEN_KWARGS = dict(
    min_new_tokens=15,
    max_new_tokens=MAX_NEW_TOKENS_GEN,
    top_k=50,         # Gemma might benefit from slightly higher top_k
    top_p=0.95,
    temperature=0.7,  # Gemma might be more coherent at slightly lower temp
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id, # This will be Gemma's <end_of_turn> ID
    pad_token_id=tokenizer.pad_token_id,
    renormalize_logits=True,
)


def evaluate_model_performance(
    eval_model, eval_tokenizer, prompt_template_func, num_samples=10
):
    eval_model.eval()
    results = {
        "valid_designs": 0,
        "total_samples": num_samples,
        "rewards": [],
        "designs": [],
        "responses": [],
    }
    gen_kwargs_eval = GEN_KWARGS.copy()

    for _ in range(num_samples):
        env = gen_env()
        prompt = prompt_template_func(env)
        # Ensure prompt is not too long for the model's context
        tokens = eval_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH - MAX_NEW_TOKENS_GEN # Reserve space for generation
        )

        query_device = DEVICE
        # Determine device for query tensor based on model's device mapping or direct device
        if hasattr(eval_model, "hf_device_map") and eval_model.hf_device_map:
            # If model is sharded, query should typically go to the first device or CPU then moved by generate
             query_device = eval_model.hf_device_map.get("base_model.model.embed_tokens", DEVICE) # Heuristic for first shard
        elif hasattr(eval_model, "device"):
            query_device = eval_model.device
        
        query = tokens.input_ids.to(query_device)
        attention_mask = tokens.attention_mask.to(query_device)

        with torch.no_grad():
            response_ids = eval_model.generate(
                query, attention_mask=attention_mask, **gen_kwargs_eval
            )

        input_len = query.shape[1]
        # Handle cases where response_ids might be shorter than input_len (should not happen with causal LMs)
        response_only_ids = response_ids[0][input_len:] if response_ids.shape[1] > input_len else response_ids[0]
        
        decoded_text = eval_tokenizer.decode(
            response_only_ids, skip_special_tokens=True # skip_special_tokens for Gemma is important
        ).strip() # Strip whitespace

        design = parse_design(decoded_text)
        # MODIFIED: Generic way to handle fixed length, assuming prompt guides model
        if design is None: # If parsing failed
            design = {} # Create empty dict to still assign length
        design["length"] = env["length"] # Always use environment's length as it's a fixed input

        metrics, reward = evaluate_beam_design(design, env) # Pass potentially modified design
        if not metrics.get("invalid", True): # Check if the design (after length fix) is valid
            results["valid_designs"] += 1
        results["rewards"].append(reward)
        results["designs"].append(design) # Store the (potentially fixed) design
        results["responses"].append(decoded_text)

        del query, response_ids, tokens, response_only_ids, attention_mask
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

    results["avg_reward"] = (
        sum(results["rewards"]) / len(results["rewards"]) if results["rewards"] else 0
    )
    results["success_rate"] = (
        results["valid_designs"] / results["total_samples"]
        if results["total_samples"] > 0
        else 0
    )
    # Re-evaluate for avg_valid_reward to ensure consistency
    valid_rewards_list = []
    for r, d_parsed, e in zip(results["rewards"], results["designs"], [gen_env() for _ in range(num_samples)]): # Use fresh envs for re-eval
        # The design 'd' already has length fixed.
        # We need to ensure the re-evaluation uses the same env context if metrics depend on it.
        # For simplicity here, we assume 'evaluate_beam_design' is deterministic for a given design and env.
        # The 'd' in results["designs"] is the one used for the original reward.
        metrics_check, _ = evaluate_beam_design(d_parsed, e) # Use the stored design
        if not metrics_check.get("invalid", True):
             valid_rewards_list.append(r)

    results["avg_valid_reward"] = (
        sum(valid_rewards_list) / len(valid_rewards_list) if valid_rewards_list else 0
    )

    eval_model.train() # Set model back to train mode
    return results


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
prompt_template_func = get_prompt_template(MODEL_NAME)

monitor = TrainingMonitor(
    patience_epochs=300, # MODIFIED: Slightly less patience if using a more capable model
    min_success_rate=0.9, 
    min_avg_reward=25.0, # MODIFIED: Higher target for a better model
    success_rate_window=30, # MODIFIED
    avg_reward_window=50  # MODIFIED
)

print(f"\n🔍 Evaluating initial model performance...")
initial_results = evaluate_model_performance(
    model, tokenizer, prompt_template_func, num_samples=20 # Increased samples for better baseline
)
print(
    f"Initial performance: Success rate: {initial_results['success_rate']:.1%}, Avg reward: {initial_results['avg_reward']:.1f}, Avg valid reward: {initial_results['avg_valid_reward']:.1f}"
)

EPOCHS = 3000  # MODIFIED: Might converge faster or hit limits sooner
valid_designs_count = 0
best_avg_reward_overall = -float("inf") # Renamed from best_avg_reward to avoid conflict with monitor

print(f"\nStarting training with {MODEL_NAME}")
print(f"Generation parameters: {GEN_KWARGS}")
print(f"Max prompt length (incl. response space): {MAX_PROMPT_LENGTH}")
print(f"Max new tokens for generation: {MAX_NEW_TOKENS_GEN}")
print(f"PPO Config: {ppo_cfg}")
print(f"Early stopping criteria:")
print(f"  - Success rate > 90% for 50 consecutive epochs")
print(f"  - Average reward > 20 for 100 consecutive epochs") 
print(f"  - No improvement for 500 epochs")
print(f"Fixed displacement reward system:")
print(f"  - Penalty for exceeding displacement limit: -50 points")
print(f"  - Optimal range: 70-90% displacement utilization")
print(f"  - Randomized material order in prompts")


for ep in range(EPOCHS):
    envs = [gen_env() for _ in range(ppo_cfg.batch_size)]
    prompts = [prompt_template_func(env) for env in envs]

    # Tokenize queries, ensuring they are on CPU for trl.generate if model is sharded
    queries_tokenized = [
        tokenizer(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH - MAX_NEW_TOKENS_GEN) # Reserve space
        .input_ids.squeeze(0)
        .to("cpu") # TRL generate often expects CPU tensors for queries list
        for p in prompts
    ]

    model.eval() # Set to eval for generation if it affects layers like dropout
    try:
        # Generate responses. trainer.generate handles moving tensors to model devices.
        responses_tokenized = trainer.generate(queries_tokenized, **GEN_KWARGS)
    except Exception as e:
        print(f"Generation error: {e}")
        if "out of memory" in str(e).lower() and DEVICE.type == "cuda":
            print("Attempting to clear CUDA cache after generation OOM.")
            torch.cuda.empty_cache()
        elif "cuda" in str(e).lower() and ("assert" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower()):
            print("CUDA error or NaN/inf detected during generation. Model health check:")
            check_model_health(model)
            print("Skipping this epoch and continuing training...")
            model.train()
            del queries_tokenized, prompts, envs
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            continue
        else: # Other errors
            model.train()
            del queries_tokenized, prompts, envs
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            continue # Skip epoch on other generation errors
    model.train() # Set back to train for PPO step

    decoded_responses = []
    for i, r_tokens in enumerate(responses_tokenized):
        input_len = len(queries_tokenized[i])
        response_only_tokens = r_tokens[input_len:] if len(r_tokens) > input_len else r_tokens
        decoded_text = tokenizer.decode(response_only_tokens, skip_special_tokens=True).strip()
        decoded_responses.append(decoded_text)

    rewards_list_raw, reward_tensors_list = [], [] # Raw rewards before normalization/clipping
    valid_this_batch = 0
    batch_reward_components_sum = {
        "stress_failure_penalty": 0.0, "displacement_failure_penalty": 0.0,
        "displacement_reward": 0.0, "weight_penalty": 0.0, "stress_efficiency_reward": 0.0
    }
    num_valid_for_components = 0

    for i, (txt, env) in enumerate(zip(decoded_responses, envs)):
        design = parse_design(txt)
        # MODIFIED: Generic way to handle fixed length
        if design is None:
            design = {}
        design["length"] = env["length"]

        metrics, r_val = evaluate_beam_design(design, env)
        rewards_list_raw.append(r_val) # Store raw reward for avg calculation

        # Reward processing for PPO (normalization, clipping)
        r_val_normalized = r_val / 100.0 # Scale factor
        r_val_clipped = max(min(r_val_normalized, 1.0), -2.0) # Clip to a range like [-2, 1]
        
        reward_tensors_list.append(
            torch.tensor(r_val_clipped, dtype=torch.float32).to("cpu") # PPO step expects CPU tensors
        )

        if not metrics.get("invalid", True):
            valid_this_batch += 1
            valid_designs_count += 1
            if "reward_components" in metrics:
                num_valid_for_components +=1
                for comp_name, comp_value in metrics["reward_components"].items():
                    if comp_name in batch_reward_components_sum:
                        batch_reward_components_sum[comp_name] += comp_value
        
        # Less frequent printing
        if ep < 2 or (ep % 50 == 0 and i == 0): # Print first sample of fewer epochs
            print(f"\nEpoch {ep+1}, Sample {i+1} response: {repr(txt[:150])}...") # Show more of response
            print(f"Parsed: {design}, Raw Reward: {r_val:.1f}, Processed Reward: {r_val_clipped:.2f}")
            if not metrics.get("invalid", True):
                print(f"Metrics: sigma={metrics.get('sigma',0):.2e}, rel_disp={metrics.get('rel_disp',0):.2e}, weight={metrics.get('weight',0):.1f}")
                print(f"Reward Components: {metrics.get('reward_components')}")


    if any(torch.isnan(r) or torch.isinf(r) for r in reward_tensors_list):
        print("NaN or inf detected in processed rewards. Skipping training step.")
        del queries_tokenized, responses_tokenized, reward_tensors_list, prompts, envs, rewards_list_raw, decoded_responses
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        continue

    # Standardize rewards for PPO (optional but often helpful, applied to r_val_clipped)
    # This was applied to raw rewards before, now applying to clipped rewards
    if len(reward_tensors_list) > 1:
        rewards_mean_clipped = torch.mean(torch.stack(reward_tensors_list))
        rewards_std_clipped = torch.std(torch.stack(reward_tensors_list))
        if rewards_std_clipped > 1e-5: # Avoid division by zero
            reward_tensors_list = [
                torch.tensor((r.item() - rewards_mean_clipped.item()) / rewards_std_clipped.item(), dtype=torch.float32).to("cpu")
                for r in reward_tensors_list
            ]
        # else: rewards are all same, no need to standardize further

    try:
        # Ensure all inputs to trainer.step are on the correct device (usually CPU for lists of tensors)
        stats = trainer.step(
            [q.to(DEVICE) for q in queries_tokenized],  # Queries to model device for step
            [r.to(DEVICE) for r in responses_tokenized], # Responses to model device for step
            reward_tensors_list # Standardized rewards (should be on CPU or handled by TRL)
        )
        
        # Use raw rewards for monitoring average performance
        avg_reward_batch_raw = sum(rewards_list_raw) / len(rewards_list_raw) if rewards_list_raw else 0.0
        success_rate_batch = valid_this_batch / len(rewards_list_raw) if rewards_list_raw else 0.0
        
        if avg_reward_batch_raw > best_avg_reward_overall:
            best_avg_reward_overall = avg_reward_batch_raw

        avg_reward_components_batch = {}
        if num_valid_for_components > 0:
            for comp_name, comp_sum in batch_reward_components_sum.items():
                avg_reward_components_batch[comp_name] = comp_sum / num_valid_for_components
        
        monitor.update(avg_reward_batch_raw, success_rate_batch, valid_this_batch, avg_reward_components_batch)

        if not check_model_health(model): # Check health after step
            print("Model has become corrupted with NaN/inf values after training step. Stopping training.")
            break

        kl_val_str = f"{stats.get('objective/kl', 0.0):.3f}" # More precision for KL
        loss_val_str = f"{stats.get('loss/total', 0.0):.3f}"
        print(
            f"Ep {ep+1:4d} | AvgRawRew={avg_reward_batch_raw:6.1f} | BestOverallRaw={best_avg_reward_overall:6.1f} | "
            f"Valid={valid_this_batch}/{len(rewards_list_raw)} ({success_rate_batch:.1%}) | KL={kl_val_str} | Loss={loss_val_str}"
        )
        
        if valid_this_batch == len(rewards_list_raw) and avg_reward_batch_raw > monitor.min_avg_reward: # Stricter perfect batch
            print(f"🎉 Excellent batch at epoch {ep+1}!")

        if (ep + 1) % 20 == 0: # Check early stopping more frequently
            stop_reasons = monitor.should_stop_early()
            if stop_reasons:
                print(f"\n🛑 Early stopping triggered at epoch {ep+1}:")
                for reason in stop_reasons: print(f"   ✓ {reason}")
                break

        if (ep + 1) % 100 == 0 or ((ep + 1) == EPOCHS // 2): # Plot midway too
            temp_dir = f"temp_plots_epoch_{ep+1}"
            os.makedirs(temp_dir, exist_ok=True)
            monitor.plot_training_progress(temp_dir)

    except Exception as e:
        print(f"Training step error: {e}")
        if "out of memory" in str(e).lower() and DEVICE.type == "cuda":
            print("Attempting to clear CUDA cache after training step OOM.")
            torch.cuda.empty_cache()
        # Fallback: try to check model health
        check_model_health(model)
        # No `del` here as they are handled at the end of the loop
        continue # Skip to next epoch

    finally: # Ensure cleanup happens
        del queries_tokenized, responses_tokenized, reward_tensors_list, prompts, envs, rewards_list_raw, decoded_responses
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()


print(f"\n✨ Training complete or stopped early after {ep+1} epochs!")

# Generate final plots and save model
model_save_dir = f"trained_models/{MODEL_NAME.split('/')[-1].replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Sanitize name
os.makedirs(model_save_dir, exist_ok=True)

print(f"💾 Saving trained model to {model_save_dir}...")
# For AutoModelForCausalLMWithValueHead, save the underlying pretrained model
if hasattr(model, "save_pretrained"): # This saves both value head and base model if applicable
    model.save_pretrained(model_save_dir)
elif hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "save_pretrained"):
    model.pretrained_model.save_pretrained(model_save_dir)
    # Optionally save the value head separately if needed, though PPOTrainer re-adds it on load
    if hasattr(model, "v_head"):
         torch.save(model.v_head.state_dict(), f"{model_save_dir}/value_head.pth")
else:
    torch.save(model.state_dict(), f"{model_save_dir}/pytorch_model.bin")
tokenizer.save_pretrained(model_save_dir)

monitor.plot_training_progress(model_save_dir)

print(f"\n🔍 Evaluating final model performance...")
if check_model_health(model):
    final_results = evaluate_model_performance(
        model, tokenizer, prompt_template_func, num_samples=50 # More samples for final eval
    )
else:
    print("Model is corrupted with NaN/inf values. Skipping final evaluation.")
    final_results = {"success_rate": 0.0, "avg_reward": 0.0, "avg_valid_reward": 0.0, "rewards": [], "designs": [], "responses": []}

print(f"\n📊 Performance Comparison:")
print(f"                    Initial    Final    Improvement")
print(
    f"Success Rate:       {initial_results['success_rate']:6.1%}   {final_results['success_rate']:6.1%}   {(final_results['success_rate'] - initial_results['success_rate']):+6.1%}"
)
print(
    f"Avg Reward:         {initial_results['avg_reward']:6.1f}   {final_results['avg_reward']:6.1f}   {(final_results['avg_reward'] - initial_results['avg_reward']):+6.1f}"
)
print(
    f"Avg Valid Reward:   {initial_results['avg_valid_reward']:6.1f}   {final_results['avg_valid_reward']:6.1f}   {(final_results['avg_valid_reward'] - initial_results['avg_valid_reward']):+6.1f}"
)

results_summary = {
    "model_name": MODEL_NAME,
    "ppo_config": str(ppo_cfg), # Save PPO config
    "gen_kwargs": str(GEN_KWARGS), # Save Gen KWARGS
    "training_epochs_completed": ep + 1, # ep is 0-indexed
    "total_valid_designs_training": valid_designs_count,
    "best_avg_training_reward_batch_overall": best_avg_reward_overall,
    "initial_evaluation": initial_results,
    "final_evaluation": final_results,
    "training_history": {
        "epoch_rewards": monitor.epoch_rewards,
        "epoch_success_rates": monitor.epoch_success_rates,
        "epoch_valid_counts": monitor.epoch_valid_counts,
        "best_rewards_seen_in_epoch": monitor.best_rewards, # Clarified name
        "avg_reward_components_per_epoch": monitor.reward_components,
    },
    "early_stopping_info": { # Clarified structure
        "triggered": len(monitor.should_stop_early()) > 0 if monitor.should_stop_early() else False,
        "reasons": monitor.should_stop_early() if monitor.should_stop_early() else "Not triggered or N/A",
        "epochs_without_improvement_at_stop": monitor.epochs_without_improvement
    },
    "improvement_metrics": { # Clarified structure
        "success_rate_delta": final_results.get("success_rate",0) - initial_results.get("success_rate",0),
        "avg_reward_delta": final_results.get("avg_reward",0) - initial_results.get("avg_reward",0),
        "avg_valid_reward_delta": final_results.get("avg_valid_reward",0) - initial_results.get("avg_valid_reward",0),
    },
}

# Save some example good designs from final evaluation
if final_results.get("rewards"):
    sorted_final_designs = sorted(zip(final_results["rewards"], final_results["designs"], final_results["responses"]), key=lambda x: x[0], reverse=True)
    results_summary["final_evaluation"]["top_designs_examples"] = [
        {"reward": r, "design": d, "response": resp} for r, d, resp in sorted_final_designs[:5] if d and not evaluate_beam_design(d, gen_env())[0].get("invalid")
    ]


with open(f"{model_save_dir}/evaluation_summary_results.json", "w") as f: # Renamed for clarity
    json.dump(results_summary, f, indent=2, default=lambda o: '<not serializable>') # Handle non-serializable

print(f"\n✅ Model, plots, and evaluation results saved to: {model_save_dir}")

print(f"📈 Training plots available at: {model_save_dir}/training_progress.png")
print(f"📊 Recent performance plot: {model_save_dir}/recent_performance.png")
print(f"\n🎯 Training Summary:")
print(f"   • Fixed displacement reward with optimal 70-90% utilization range")
print(f"   • Added -50 penalty for exceeding displacement threshold")
print(f"   • Randomized material order to prevent positional bias")
print(f"   • Increased learning rate to {ppo_cfg.learning_rate} for faster convergence")
print(f"   • Increased exploration with temp={GEN_KWARGS['temperature']}, top_p={GEN_KWARGS['top_p']}")
print(f"   • Comprehensive monitoring with early stopping and progress plots")