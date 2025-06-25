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
    TrainingArguments,
)  # TrainingArguments for gradient_checkpointing_kwargs
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import warnings

warnings.filterwarnings("ignore", message="You're using a LlamaTokenizerFast tokenizer")
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
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = TINYLLAMA_MODEL
print(f"Selected model: {MODEL_NAME}")

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
STRESS_EFFICIENCY_REWARD_SCALE = 150.0


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

    # FIXED DISPLACEMENT REWARD SYSTEM
    # Penalty for exceeding displacement limit (like stress failure)
    displacement_failure_penalty = -50.0 if rel_disp > target_rd else 0.0
    
    # Reward for being in optimal range (70-90% of allowable displacement)
    displacement_reward = 0.0
    if target_rd > 1e-9 and rel_disp <= target_rd:
        utilization = rel_disp / target_rd
        if 0.7 <= utilization <= 0.9:
            # Maximum reward for optimal range
            displacement_reward = 50.0
        elif utilization < 0.7:
            # Partial reward for under-utilizing (overdesign)
            displacement_reward = 25.0 * (utilization / 0.7)
        else:  # 0.9 < utilization <= 1.0
            # Decreasing reward as you approach the limit
            displacement_reward = 50.0 * (1.0 - utilization) / 0.1

    stress_failure_penalty = -100.0 if sigma > mat["yield_strength"] else 0.0
    weight_penalty = -WEIGHT_PENALTY_COEFFICIENT * weight * 0.01
    stress_efficiency_reward = 0.0
    if sigma <= mat["yield_strength"] and mat["yield_strength"] > 1e-9:
        utilization_ratio = sigma / mat["yield_strength"]
        stress_efficiency_reward = (utilization_ratio**2) * 15.0

    reward = (
        stress_failure_penalty
        + displacement_failure_penalty  # NEW: Penalty for exceeding displacement
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
            "displacement_failure_penalty": displacement_failure_penalty,  # NEW
            "displacement_reward": displacement_reward,
            "weight_penalty": weight_penalty,
            "stress_efficiency_reward": stress_efficiency_reward,
        },
    }, reward


PARSE_PATTERNS = [
    {
        "material": re.compile(r"Material:\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"Width:\s*([\d.]+)", re.I),
        "height": re.compile(r"Height:\s*([\d.]+)", re.I),
    },
    {
        "material": re.compile(r"Material:\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"Width:\s*([\d.]+)", re.I),
        "height": re.compile(r"Height:\s*([\d.]+)", re.I),
        "length": re.compile(r"Length:\s*([\d.]+)", re.I),
    },
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
                    if float_val <= 0:
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
    """Check if model parameters contain NaN or inf values"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: NaN/inf detected in parameter {name}")
            return False
    return True


def get_prompt_template(model_name: str):
    materials = ["Aluminum", "Carbon_Fiber", "Steel", "Wood"]
    
    if "tinyllama" in model_name.lower():
        return lambda env: (
            "<|system|>\nYou are a helpful engineering assistant.\n<|user|>\n"
            f"Design a rectangular beam for these requirements:\n"
            f"- Length of beam: {env['length']:.2f} m (This is a fixed input)\n"
            f"- Point load: {env['load_P']:.0f} N at position {env['load_a_fraction']:.2f} along length\n"
            f"- Max relative displacement: {env['target_relative_displacement']:.4f}\n"
            f"- Select one material from this list: {random.sample(materials, len(materials))}\n\n"
            f"Respond with ONE design that has one Material, Width, and Height in this exact format:\n"
            f"Material: [Select_a_material_from_list]\nWidth: [width_in_meters]\nHeight: [height_in_meters]\n<|assistant|>\n"
        )
    return lambda env: "Unsupported model for prompt"


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
        
        # Tracking lists
        self.epoch_rewards = []
        self.epoch_success_rates = []
        self.epoch_valid_counts = []
        self.best_rewards = []
        self.reward_components = []
        
        # Early stopping tracking
        self.best_avg_reward = -float('inf')
        self.epochs_without_improvement = 0
        self.recent_success_rates = deque(maxlen=success_rate_window)
        self.recent_avg_rewards = deque(maxlen=avg_reward_window)
        
    def update(self, avg_reward, success_rate, valid_count, reward_components=None):
        # Update tracking lists
        self.epoch_rewards.append(avg_reward)
        self.epoch_success_rates.append(success_rate)
        self.epoch_valid_counts.append(valid_count)
        self.best_rewards.append(max(self.epoch_rewards))
        if reward_components:
            self.reward_components.append(reward_components)
        
        # Update early stopping tracking
        self.recent_success_rates.append(success_rate)
        self.recent_avg_rewards.append(avg_reward)
        
        # Check for improvement
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def should_stop_early(self):
        """Check all early stopping criteria"""
        reasons = []
        
        # Criterion 1: High success rate sustained
        if (len(self.recent_success_rates) >= self.success_rate_window and 
            all(sr >= self.min_success_rate for sr in self.recent_success_rates)):
            reasons.append(f"Success rate > {self.min_success_rate:.1%} for {self.success_rate_window} epochs")
        
        # Criterion 2: High average reward sustained
        if (len(self.recent_avg_rewards) >= self.avg_reward_window and 
            all(ar >= self.min_avg_reward for ar in self.recent_avg_rewards)):
            reasons.append(f"Average reward > {self.min_avg_reward} for {self.avg_reward_window} epochs")
        
        # Criterion 3: No improvement for too long
        if self.epochs_without_improvement >= self.patience_epochs:
            reasons.append(f"No improvement for {self.patience_epochs} epochs")
        
        return reasons
    
    def plot_training_progress(self, save_dir):
        """Create comprehensive training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.epoch_rewards) + 1)
        
        # Plot 1: Average Reward over time
        axes[0, 0].plot(epochs, self.epoch_rewards, 'b-', label='Avg Reward', alpha=0.7)
        axes[0, 0].plot(epochs, self.best_rewards, 'r-', label='Best Reward', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Success Rate over time
        axes[0, 1].plot(epochs, [sr * 100 for sr in self.epoch_success_rates], 'g-', linewidth=2)
        axes[0, 1].axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Target')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # Plot 3: Valid Designs Count
        axes[1, 0].bar(epochs, self.epoch_valid_counts, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Valid Designs per Batch')
        axes[1, 0].set_title('Valid Designs Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Reward Components (if available)
        if self.reward_components and len(self.reward_components) > 0:
            # Average reward components over epochs
            components = {}
            for epoch_components in self.reward_components:
                for comp_name, comp_value in epoch_components.items():
                    if comp_name not in components:
                        components[comp_name] = []
                    components[comp_name].append(comp_value)
            
            for comp_name, comp_values in components.items():
                axes[1, 1].plot(epochs[:len(comp_values)], comp_values, 
                               label=comp_name.replace('_', ' ').title(), alpha=0.8)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Reward Component')
            axes[1, 1].set_title('Reward Components Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no reward components, show moving averages
            if len(epochs) >= 10:
                window = min(20, len(epochs) // 4)
                moving_avg = np.convolve(self.epoch_rewards, np.ones(window)/window, mode='valid')
                moving_epochs = epochs[window-1:]
                axes[1, 1].plot(epochs, self.epoch_rewards, 'b-', alpha=0.3, label='Raw')
                axes[1, 1].plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'{window}-epoch MA')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].set_title('Reward Moving Average')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary statistics plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Recent performance metrics
        recent_epochs = min(100, len(epochs))
        recent_rewards = self.epoch_rewards[-recent_epochs:]
        recent_success = self.epoch_success_rates[-recent_epochs:]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(epochs[-recent_epochs:], recent_rewards, 'b-', linewidth=2, label='Avg Reward')
        line2 = ax2.plot(epochs[-recent_epochs:], [sr * 100 for sr in recent_success], 'g-', linewidth=2, label='Success Rate %')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Reward', color='b')
        ax2.set_ylabel('Success Rate (%)', color='g')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        
        ax.set_title(f'Recent Performance (Last {recent_epochs} Epochs)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/recent_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to {save_dir}")


# -----------------------------------------------------------------------------
# RL Setup
# -----------------------------------------------------------------------------
MAX_PROMPT_LENGTH = 256
MAX_NEW_TOKENS_GEN = 60

ppo_cfg = PPOConfig(
    learning_rate=1e-6,  # MODIFIED: Increased from 5e-8 for faster learning
    batch_size=8,        # MODIFIED: Increased from 4 for better gradient estimates
    mini_batch_size=2,   # MODIFIED: Increased from 1
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    cliprange=0.01,
    cliprange_value=0.01,
    vf_coef=0.01,
    max_grad_norm=0.1,
    ppo_epochs=1,
    # log_with="wandb",
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs_common = {"trust_remote_code": True}
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    if DEVICE.type == "cuda":
        model_kwargs_cuda = {
            "torch_dtype": torch.bfloat16,  # Using bfloat16 for better stability on CUDA
            "device_map": "auto",
        }
        # For debugging precision issues, you might temporarily switch to torch.float32,
        # but this will increase memory usage significantly:
        # "torch_dtype": torch.float32,

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME, **model_kwargs_common, **model_kwargs_cuda
        )
        if hasattr(model, "pretrained_model"):
            model.pretrained_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            model.pretrained_model.config.use_cache = False
        else:  # Should not happen with AutoModelForCausalLMWithValueHead from TRL
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            model.config.use_cache = False

        model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME,
            **model_kwargs_common,
            **model_kwargs_cuda,  # Ensure ref model also uses bfloat16 on CUDA
        )
    else:  # CPU or MPS
        model_kwargs_other = {
            "torch_dtype": torch.float32
        }  # MPS typically works well with float32
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
        if (
            hasattr(model, "pretrained_model") and model.pretrained_model is not None
        ):  # Check if pretrained_model exists and is not None
            print(f"Model dtype: {model.pretrained_model.dtype}")  # CORRECTED LINE
        elif hasattr(
            model, "parameters"
        ):  # Fallback if no pretrained_model attribute or it's None
            try:
                print(f"Model parameter dtype: {next(model.parameters()).dtype}")
            except StopIteration:
                print(
                    "Model has no parameters to determine dtype from."
                )  # Should not happen for a valid model
        else:
            print("Could not determine model dtype.")


except Exception as e:
    print(f"Error loading model: {e}")
    print("Try installing: pip install accelerate transformers[torch]")
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
    top_k=40,
    top_p=0.95,      # MODIFIED: Increased from 0.9 for more exploration
    temperature=0.8,  # MODIFIED: Increased from 0.5 for more exploration
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
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
        tokens = eval_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH
        )

        query_device = DEVICE
        if DEVICE.type == "cuda" and hasattr(eval_model, "hf_device_map"):
            try:
                query_device = next(eval_model.parameters()).device
            except StopIteration:
                query_device = DEVICE
        elif hasattr(eval_model, "device"):
            query_device = eval_model.device

        query = tokens.input_ids.to(query_device)
        attention_mask = tokens.attention_mask.to(query_device)

        with torch.no_grad():
            response_ids = eval_model.generate(
                query, attention_mask=attention_mask, **gen_kwargs_eval
            )

        input_len = query.shape[1]
        response_only_ids = response_ids[0][input_len:]
        decoded_text = eval_tokenizer.decode(
            response_only_ids, skip_special_tokens=True
        )

        design = parse_design(decoded_text)
        if "tinyllama" in MODEL_NAME.lower():
            if design is None:
                design = {}
            design["length"] = env["length"]

        metrics, reward = evaluate_beam_design(design or {}, env)
        if design is not None and not metrics.get("invalid", True):
            results["valid_designs"] += 1
        results["rewards"].append(reward)
        results["designs"].append(design)
        results["responses"].append(decoded_text)

        del query, response_ids, tokens, response_only_ids
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
    valid_rewards_list = [
        r
        for r, d in zip(results["rewards"], results["designs"])
        if d is not None
        and not evaluate_beam_design(d, gen_env())[0].get("invalid", True)
    ]
    results["avg_valid_reward"] = (
        sum(valid_rewards_list) / len(valid_rewards_list) if valid_rewards_list else 0
    )

    eval_model.train()
    return results


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
prompt_template_func = get_prompt_template(MODEL_NAME)

# Initialize monitor
monitor = TrainingMonitor(
    patience_epochs=500,
    min_success_rate=0.9, 
    min_avg_reward=20.0,
    success_rate_window=50,
    avg_reward_window=100
)

print(f"\n🔍 Evaluating initial model performance...")
initial_results = evaluate_model_performance(
    model, tokenizer, prompt_template_func, num_samples=10
)
print(
    f"Initial performance: Success rate: {initial_results['success_rate']:.1%}, Avg reward: {initial_results['avg_reward']:.1f}, Avg valid reward: {initial_results['avg_valid_reward']:.1f}"
)

EPOCHS = 5000  # Increased from 100
valid_designs_count = 0
best_avg_reward = -float("inf")

print(f"\nStarting training with {MODEL_NAME}")
print(f"Generation parameters: {GEN_KWARGS}")
print(f"Max prompt length: {MAX_PROMPT_LENGTH}")
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

    queries_tokenized = [
        tokenizer(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH)
        .input_ids.squeeze(0)
        .to("cpu")
        for p in prompts
    ]

    model.eval()
    try:
        responses_tokenized = trainer.generate(queries_tokenized, **GEN_KWARGS)
    except Exception as e:
        print(f"Generation error: {e}")
        if "out of memory" in str(e).lower() and DEVICE.type == "cuda":
            print("Attempting to clear CUDA cache after generation OOM.")
            torch.cuda.empty_cache()
        elif "cuda" in str(e).lower() and "assert" in str(e).lower():
            print("CUDA assertion error detected. Model may have NaN/inf values.")
            print("Skipping this epoch and continuing training...")
            model.train()
            del queries_tokenized, prompts, envs
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue
        else:
            model.train()
            del queries_tokenized, prompts, envs
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue
    model.train()

    decoded_responses = []
    for i, r_tokens in enumerate(responses_tokenized):
        input_len = len(queries_tokenized[i])
        response_only_tokens = r_tokens[input_len:]
        decoded_text = tokenizer.decode(response_only_tokens, skip_special_tokens=True)
        decoded_responses.append(decoded_text)

    rewards_list, reward_tensors_list = [], []
    valid_this_batch = 0
    batch_reward_components = {
        "stress_failure_penalty": [],
        "displacement_failure_penalty": [],
        "displacement_reward": [],
        "weight_penalty": [],
        "stress_efficiency_reward": []
    }

    for i, (txt, env) in enumerate(zip(decoded_responses, envs)):
        design = parse_design(txt)
        if "tinyllama" in MODEL_NAME.lower():
            if design is None:
                design = {}
            design["length"] = env["length"]

        metrics, r_val = evaluate_beam_design(design or {}, env)
        r_val_normalized = r_val / 100.0
        r_val_clipped = max(min(r_val_normalized, 1.0), -10.0)
        rewards_list.append(r_val_clipped)
        reward_tensors_list.append(
            torch.tensor(r_val_clipped, dtype=torch.float32).to("cpu")
        )

        # Track reward components
        if not metrics.get("invalid", True) and "reward_components" in metrics:
            for comp_name, comp_value in metrics["reward_components"].items():
                if comp_name in batch_reward_components:
                    batch_reward_components[comp_name].append(comp_value)

        if design is not None and not metrics.get("invalid", True):
            valid_this_batch += 1
            valid_designs_count += 1

        if ep < 2 or (ep % 20 == 0 and i == 0):
            print(f"\nEpoch {ep+1}, Sample {i+1} response: {repr(txt[:100])}...")
            print(f"Parsed: {design}, Reward: {r_val:.1f}")

    if any(torch.isnan(r) or torch.isinf(r) for r in reward_tensors_list):
        print("NaN or inf detected in rewards. Skipping training step.")
        del (
            queries_tokenized,
            responses_tokenized,
            reward_tensors_list,
            prompts,
            envs,
            rewards_list,
            decoded_responses,
        )
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        continue

    if len(rewards_list) > 1:
        rewards_mean = sum(rewards_list) / len(rewards_list)
        rewards_std = (
            sum((r - rewards_mean) ** 2 for r in rewards_list) / len(rewards_list)
        ) ** 0.5
        if rewards_std > 1e-8:
            reward_tensors_list = [
                torch.tensor((r - rewards_mean) / rewards_std, dtype=torch.float32).to(
                    "cpu"
                )
                for r in rewards_list
            ]

    try:
        stats = trainer.step(
            queries_tokenized, responses_tokenized, reward_tensors_list
        )
        avg_reward_batch = (
            sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
        )
        success_rate_batch = valid_this_batch / len(rewards_list) if rewards_list else 0.0
        
        if avg_reward_batch > best_avg_reward:
            best_avg_reward = avg_reward_batch

        # Calculate average reward components for this batch
        avg_reward_components = {}
        for comp_name, comp_values in batch_reward_components.items():
            avg_reward_components[comp_name] = (
                sum(comp_values) / len(comp_values) if comp_values else 0.0
            )

        # Update monitor
        monitor.update(avg_reward_batch, success_rate_batch, valid_this_batch, avg_reward_components)

        if not check_model_health(model):
            print("Model has become corrupted with NaN/inf values. Stopping training.")
            break

        kl_val_str = f"{stats.get('objective/kl', 0.0):.2f}"
        print(
            f"Epoch {ep+1:4d} | Avg Reward={avg_reward_batch:6.1f} | Best Avg={best_avg_reward:6.1f} | "
            f"Valid Batch={valid_this_batch}/{len(rewards_list)} | Total Valid={valid_designs_count} | "
            f"Success Rate={success_rate_batch:.1%} | KL={kl_val_str}"
        )
        
        if valid_this_batch == len(rewards_list) and avg_reward_batch > 0:
            print(f"🎉 Perfect batch at epoch {ep+1}!")

        # Check for early stopping every 10 epochs
        if (ep + 1) % 10 == 0:
            stop_reasons = monitor.should_stop_early()
            if stop_reasons:
                print(f"\n🛑 Early stopping triggered at epoch {ep+1}:")
                for reason in stop_reasons:
                    print(f"   ✓ {reason}")
                break

        # Generate plots every 100 epochs
        if (ep + 1) % 100 == 0:
            temp_dir = f"temp_plots_epoch_{ep+1}"
            os.makedirs(temp_dir, exist_ok=True)
            monitor.plot_training_progress(temp_dir)
            print(f"📊 Checkpoint plots saved at epoch {ep+1}")

    except Exception as e:
        print(f"Training step error: {e}")
        if "out of memory" in str(e).lower() and DEVICE.type == "cuda":
            print("Attempting to clear CUDA cache after training step OOM.")
            torch.cuda.empty_cache()
        del (
            queries_tokenized,
            responses_tokenized,
            reward_tensors_list,
            prompts,
            envs,
            rewards_list,
            decoded_responses,
        )
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        continue

    del (
        queries_tokenized,
        responses_tokenized,
        reward_tensors_list,
        prompts,
        envs,
        rewards_list,
        decoded_responses,
    )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

print(f"\n✨ Training complete!")

# Generate final plots and save model
model_save_dir = f"trained_models/{MODEL_NAME.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_save_dir, exist_ok=True)

print(f"💾 Saving trained model to {model_save_dir}...")
if hasattr(model, "save_pretrained"):
    model.save_pretrained(model_save_dir)
else:
    torch.save(model.state_dict(), f"{model_save_dir}/pytorch_model.bin")
tokenizer.save_pretrained(model_save_dir)

# Generate final plots
monitor.plot_training_progress(model_save_dir)

print(f"\n🔍 Evaluating final model performance...")
if check_model_health(model):
    final_results = evaluate_model_performance(
        model, tokenizer, prompt_template_func, num_samples=20
    )
else:
    print("Model is corrupted with NaN/inf values. Skipping final evaluation.")
    final_results = {"success_rate": 0.0, "avg_reward": 0.0, "avg_valid_reward": 0.0}

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

# Enhanced results summary
results_summary = {
    "model_name": MODEL_NAME,
    "training_epochs_completed": ep + 1,
    "total_valid_designs_training": valid_designs_count,
    "best_avg_training_reward_batch": best_avg_reward,
    "initial_evaluation": initial_results,
    "final_evaluation": final_results,
    "training_history": {
        "epoch_rewards": monitor.epoch_rewards,
        "epoch_success_rates": monitor.epoch_success_rates,
        "epoch_valid_counts": monitor.epoch_valid_counts,
        "best_rewards": monitor.best_rewards
    },
    "early_stopping": {
        "triggered": len(monitor.should_stop_early()) > 0,
        "reasons": monitor.should_stop_early(),
        "epochs_without_improvement": monitor.epochs_without_improvement
    },
    "improvement": {
        "success_rate_delta": final_results["success_rate"] - initial_results["success_rate"],
        "avg_reward_delta": final_results["avg_reward"] - initial_results["avg_reward"],
        "avg_valid_reward_delta": final_results["avg_valid_reward"] - initial_results["avg_valid_reward"],
    },
}

with open(f"{model_save_dir}/evaluation_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

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