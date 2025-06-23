# beam_design_rl_training_english_models.py
# -----------------------------------------------------------------------------
# RL fine-tuning small English language models on beam design task
# Supports TinyLlama, Phi-2, Gemma-2B, and Llama 3.2 models
# -----------------------------------------------------------------------------

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# -----------------------------------------------------------------------------
# Model Selection - Choose your preferred small model
# -----------------------------------------------------------------------------

# Option 1: TinyLlama (1.1B) - Best for very limited resources
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Option 2: Microsoft Phi-2 (2.7B) - Good reasoning capabilities
PHI2_MODEL = "microsoft/phi-2"

# Option 3: Gemma 2B - Google's efficient model
GEMMA_MODEL = "google/gemma-2b-it"

# Option 4: Llama 3.2 1B - Meta's newest small model
LLAMA32_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Select your model here:
MODEL_NAME = TINYLLAMA_MODEL  # Change this to experiment with different models

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

def evaluate_beam_design(
    design: Dict[str, float],
    env: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Return per-design metrics and scalar reward (higher is better)."""
    
    req_keys = ["material", "width", "height", "length"]
    if any(k not in design or design[k] is None for k in req_keys):
        return {"invalid": True, "reason": "missing_keys"}, -750.0
    if design["material"] not in MATERIAL_PROPERTIES:
        return {"invalid": True, "reason": "invalid_material"}, -750.0
    if any(design[k] <= 0 for k in ("width", "height", "length")):
        return {"invalid": True, "reason": "negative_dimensions"}, -750.0

    mat = MATERIAL_PROPERTIES[design["material"]]
    b, h, L = design["width"], design["height"], design["length"]
    P = env["load_P"]
    a = env["load_a_fraction"] * L
    target_rd = env["target_relative_displacement"]

    I = b * h**3 / 12.0
    if I < 1e-12:
        return {"invalid": True, "reason": "zero_moment"}, -750.0

    M_max = P * a * (L - a) / L
    sigma = abs(M_max * (h / 2) / I)
    delta = P * a**2 * (L - a) ** 2 / (3 * mat["youngs_modulus"] * I * L)
    rel_disp = delta / L
    weight = mat["density"] * b * h * L

    # Reward components
    stress_pen = -1_000.0 if sigma > mat["yield_strength"] else 0.0
    disp_reward = max(0.0, 1.0 - rel_disp / (target_rd + 1e-9)) * 500.0
    max_w = MATERIAL_PROPERTIES["Steel"]["density"] * 0.2 * 0.2 * 5.0
    wt_reward = max(0.0, 1.0 - weight / max_w) * 200.0

    reward = stress_pen + disp_reward + wt_reward
    return {
        "sigma": sigma,
        "rel_disp": rel_disp,
        "weight": weight,
        "invalid": False,
    }, reward

# Enhanced parsing for different model outputs
PARSE_PATTERNS = [
    # Standard format
    {
        "material": re.compile(r"Material:\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"Width:\s*([\d.]+)", re.I),
        "height": re.compile(r"Height:\s*([\d.]+)", re.I),
        "length": re.compile(r"Length:\s*([\d.]+)", re.I),
    },
    # Alternative formats
    {
        "material": re.compile(r"(?:Material|Mat):\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"(?:Width|W):\s*([\d.]+)", re.I),
        "height": re.compile(r"(?:Height|H):\s*([\d.]+)", re.I),
        "length": re.compile(r"(?:Length|L):\s*([\d.]+)", re.I),
    },
    # Bullet format
    {
        "material": re.compile(r"-\s*Material:\s*([A-Za-z_]+)", re.I),
        "width": re.compile(r"-\s*Width:\s*([\d.]+)", re.I),
        "height": re.compile(r"-\s*Height:\s*([\d.]+)", re.I),
        "length": re.compile(r"-\s*Length:\s*([\d.]+)", re.I),
    }
]

def parse_design(text: str):
    """Parse beam design with multiple pattern attempts."""
    
    # Try each pattern set
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
                # Flexible material matching
                match = None
                for mat_name in MATERIAL_PROPERTIES:
                    if (mat_name.lower().replace("_", "") == val.lower().replace("_", "") or
                        mat_name.lower() == val.lower()):
                        match = mat_name
                        break
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
        
        if success:
            return out
    
    return None

def gen_env():
    return {
        "load_P": random.uniform(1_000.0, 20_000.0),
        "load_a_fraction": random.uniform(0.1, 0.9),
        "target_relative_displacement": random.uniform(5e-4, 3e-3),
    }

# -----------------------------------------------------------------------------
# Model-specific prompt templates
# -----------------------------------------------------------------------------

def get_prompt_template(model_name: str):
    """Get model-specific prompt template."""
    
    if "tinyllama" in model_name.lower():
        return lambda env: (
            "<|system|>\nYou are a helpful engineering assistant.\n<|user|>\n"
            f"Design a rectangular beam for these requirements:\n"
            f"- Point load: {env['load_P']:.0f} N at position {env['load_a_fraction']:.2f} along length\n"
            f"- Max relative deflection: {env['target_relative_displacement']:.4f}\n"
            f"- Materials: Steel, Aluminum, Wood, Carbon_Fiber\n\n"
            f"Respond in this exact format:\n"
            f"Material: Steel\n"
            f"Width: 0.1\n"
            f"Height: 0.2\n"
            f"Length: 3.0\n<|assistant|>\n"
        )
    
    elif "phi" in model_name.lower():
        return lambda env: (
            f"Design a rectangular beam with these specifications:\n"
            f"Load: {env['load_P']:.0f} N at {env['load_a_fraction']:.2f} * length\n"
            f"Max deflection ratio: {env['target_relative_displacement']:.4f}\n"
            f"Available materials: Steel, Aluminum, Wood, Carbon_Fiber\n\n"
            f"Output format:\n"
            f"Material: [material name]\n"
            f"Width: [meters]\n"
            f"Height: [meters]\n"
            f"Length: [meters]\n\n"
            f"Design:"
        )
    
    elif "gemma" in model_name.lower():
        return lambda env: (
            f"<start_of_turn>user\n"
            f"I need to design a simply supported rectangular beam.\n"
            f"Requirements:\n"
            f"- Point load: {env['load_P']:.0f} N\n"
            f"- Load position: {env['load_a_fraction']:.2f} along beam length\n"
            f"- Maximum relative displacement: {env['target_relative_displacement']:.4f}\n"
            f"- Choose from: Steel, Aluminum, Wood, Carbon_Fiber\n\n"
            f"Please respond with:\n"
            f"Material: [name]\n"
            f"Width: [value in meters]\n"
            f"Height: [value in meters]\n"
            f"Length: [value in meters]\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
    
    else:  # Default for Llama and others
        return lambda env: (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are an engineering assistant that designs beams.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Design a rectangular beam:\n"
            f"- Load: {env['load_P']:.0f} N at {env['load_a_fraction']:.2f} of length\n"
            f"- Max relative displacement: {env['target_relative_displacement']:.4f}\n"
            f"- Materials: Steel, Aluminum, Wood, Carbon_Fiber\n\n"
            f"Format:\n"
            f"Material: [name]\n"
            f"Width: [meters]\n"
            f"Height: [meters]\n"
            f"Length: [meters]<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

# -----------------------------------------------------------------------------
# RL Setup
# -----------------------------------------------------------------------------

ppo_cfg = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

try:
    # Load model and tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto" if DEVICE.type != "mps" else None
    )
    
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto" if DEVICE.type != "mps" else None
    )
    
    # Move to device
    model.to(DEVICE)
    model_ref.to(DEVICE)
    
    print("✓ Models loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Try installing: pip install accelerate transformers[torch]")
    exit(1)

# Create trainer
trainer = PPOTrainer(
    config=ppo_cfg,
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
)
print("✓ PPOTrainer ready")

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

# Model-specific generation parameters
if "phi" in MODEL_NAME.lower():
    GEN_KWARGS = dict(
        min_new_tokens=20,
        max_new_tokens=100,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
elif "gemma" in MODEL_NAME.lower():
    GEN_KWARGS = dict(
        min_new_tokens=30,
        max_new_tokens=120,
        top_k=40,
        top_p=0.95,
        temperature=0.8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
else:  # TinyLlama and others
    GEN_KWARGS = dict(
        min_new_tokens=25,
        max_new_tokens=150,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

prompt_template = get_prompt_template(MODEL_NAME)

EPOCHS = 100
valid_designs_count = 0
best_avg_reward = -1000

print(f"\nStarting training with {MODEL_NAME}")
print(f"Generation parameters: {GEN_KWARGS}")

for ep in range(EPOCHS):
    envs = [gen_env() for _ in range(ppo_cfg.batch_size)]
    prompts = [prompt_template(env) for env in envs]
    
    # Tokenize
    queries = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
        queries.append(tokens.input_ids.squeeze(0).to(DEVICE))
    
    # Generate
    try:
        responses = trainer.generate(queries, **GEN_KWARGS)
    except Exception as e:
        print(f"Generation error: {e}")
        continue
    
    # Decode responses (remove input prompt)
    decoded = []
    for i, r in enumerate(responses):
        input_len = len(queries[i])
        response_only = r[input_len:]
        decoded_text = tokenizer.decode(response_only, skip_special_tokens=True)
        decoded.append(decoded_text)
    
    # Evaluate
    rewards, reward_tensors = [], []
    valid_this_batch = 0
    
    for i, (txt, env) in enumerate(zip(decoded, envs)):
        if ep < 3 or i == 0:  # Show first few epochs and first sample
            print(f"\nSample {i+1} response: {repr(txt[:150])}...")
        
        design = parse_design(txt)
        if design is not None:
            valid_this_batch += 1
            valid_designs_count += 1
        
        metrics, r = evaluate_beam_design(design or {}, env)
        rewards.append(r)
        reward_tensors.append(torch.tensor(r, dtype=torch.float32).to(DEVICE))
        
        if ep < 3 or i == 0:
            print(f"Parsed: {design}, Reward: {r:.1f}")
    
    # Training step
    try:
        stats = trainer.step(queries, responses, reward_tensors)
        avg_reward = sum(rewards) / len(rewards)
        
        # Track best performance
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        # Get stats with fallbacks
        kl_val = (stats.get("objective/kl") or 
                 stats.get("ppo/mean_non_score_reward") or
                 stats.get("kl") or "N/A")
        loss_val = (stats.get("ppo/loss/total") or 
                   stats.get("objective/entropy") or
                   stats.get("loss") or "N/A")
        
        print(f"Epoch {ep+1:3d} | reward={avg_reward:6.1f} | best={best_avg_reward:6.1f} | "
              f"valid={valid_this_batch}/{len(rewards)} | total_valid={valid_designs_count}")
        
        # Early success check
        if valid_this_batch == len(rewards) and avg_reward > 0:
            print(f"🎉 Perfect batch at epoch {ep+1}! All designs valid with positive rewards.")
            
    except Exception as e:
        print(f"Training error: {e}")
        continue
    
    # Memory cleanup
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

print(f"\n✨ Training complete!")
print(f"Best average reward: {best_avg_reward:.1f}")
print(f"Total valid designs: {valid_designs_count}")
print(f"Model used: {MODEL_NAME}")