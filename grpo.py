import os, re, json, random, warnings
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from torch.nn.utils.rnn import pad_sequence

# ------------------------------
# Environment / Determinism
# ------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message="You're using a LlamaTokenizerFast tokenizer")
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values` as a tuple")

SEED = 1337
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# ------------------------------
# Device
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE} (cuda={torch.cuda.is_available()})")

# ------------------------------
# Model / Tokenizer
# ------------------------------
MODEL_NAME = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.padding_side = "right"

fp = dict(trust_remote_code=True, torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32)
policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **fp).to(DEVICE)

if hasattr(policy, "gradient_checkpointing_enable"):
    try:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    except TypeError:
        policy.gradient_checkpointing_enable()
if hasattr(policy.config, "use_cache"):
    policy.config.use_cache = False

# ------------------------------
# GRPO & Training Configuration
# ------------------------------
LEARNING_RATE = 5e-6
EPOCHS = 600
LOG_EVERY = 10
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

N_CANDIDATES = 4
BETA = 2.0 # Temperature for listwise loss

optimizer = AdamW(policy.parameters(), lr=LEARNING_RATE)
print("✓ GRPO setup and AdamW optimizer initialized")

# ------------------------------
# Materials / Properties
# ------------------------------
MATERIALS = ["Aluminum", "Carbon_Fiber", "Steel", "Wood"]
MATERIAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "Steel":        {"density": 7850, "youngs_modulus": 200e9, "yield_strength": 250e6},
    "Aluminum":     {"density": 2700, "youngs_modulus":  70e9, "yield_strength": 200e6},
    "Wood":         {"density":  600, "youngs_modulus":  10e9, "yield_strength":  40e6},
    "Carbon_Fiber": {"density": 1600, "youngs_modulus": 150e9, "yield_strength": 1500e6},
}

# ------------------------------
# Code deflection limit
# ------------------------------
CODE_LIMIT_CHOICES = [240, 350, 360, 480, 600]
USE_RANDOM_LIMIT = True
DEFAULT_SERVICE_RATIO = 350

# ------------------------------
# Generation config
# ------------------------------
MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS = 24

GEN_KWARGS_GRPO = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=N_CANDIDATES
)

GEN_KWARGS_EVAL = dict(
    min_new_tokens=6,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ------------------------------
# Task / Environment
# ------------------------------
def pick_service_ratio() -> int:
    return random.choice(CODE_LIMIT_CHOICES) if USE_RANDOM_LIMIT else DEFAULT_SERVICE_RATIO

def gen_env() -> Dict[str, float]:
    R = pick_service_ratio()
    return {
        "load_P": random.uniform(1_000.0, 20_000.0),
        "load_a_fraction": random.uniform(0.1, 0.9),
        "length": random.uniform(1.0, 5.0),
        "service_ratio": R,
        "target_relative_displacement": 1.0 / R,
    }

def prompt_template(env: Dict[str, float]) -> str:
    mats = random.sample(MATERIALS, len(MATERIALS))
    R = env["service_ratio"]
    return (
        "<start_of_turn>user\n"
        "You are a precise engineering assistant. Follow the instructions exactly.\n"
        "Design a rectangular beam for these requirements:\n"
        f"- Length of beam: {env['length']:.2f} m (This is fixed; do NOT output it.)\n"
        f"- Point load: {env['load_P']:.0f} N at position {env['load_a_fraction']:.2f} along length\n"
        f"- Deflection limit (serviceability): L/{R}  (i.e., δ/L ≤ 1/{R})\n"
        f"- Select one material from this list: {mats}\n\n"
        "Respond with ONE design in exactly this format (no extra text):\n"
        "Material: <one_of_list>\n"
        "Width: <meters>\n"
        "Height: <meters>\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# ------------------------------
# Parsing / Physics Evaluation
# ------------------------------
FORMAT_RX = re.compile(
    r"^\s*Material:\s*([A-Za-z_]+)\s*\n\s*Width:\s*([0-9]*\.?[0-9]+)(?:\s*m)?\s*\n\s*Height:\s*([0-9]*\.?[0-9]+)(?:\s*m)?\s*$",
    re.I
)

def parse_design(text: str):
    m = FORMAT_RX.match(text.strip())
    if not m: return None
    material_raw, w_s, h_s = m.group(1), m.group(2), m.group(3)
    material = next((k for k in MATERIAL_PROPERTIES if k.lower().replace("_","") == material_raw.lower().replace("_","")), None)
    if material is None: return None
    w = float(w_s); h = float(h_s)
    if w <= 0 or h <= 0: return None
    return {"material": material, "width": w, "height": h}

def evaluate_beam_design(design: Dict, env: Dict) -> Dict:
    mat = MATERIAL_PROPERTIES.get(design.get("material"))
    if not mat or design.get("width", 0) <= 0 or design.get("height", 0) <= 0:
        return {"invalid": True, "reason": "bad_parse"}

    b, h, L = design["width"], design["height"], env["length"]
    P, a_frac, target_rd = env["load_P"], env["load_a_fraction"], env["target_relative_displacement"]
    a = a_frac * L
    I = b * h**3 / 12.0
    if I <= 1e-18: return {"invalid": True, "reason": "zero_moment"}

    M_max = P * a * (L - a) / L
    sigma = abs(M_max * (h / 2) / I)
    delta = P * a**2 * (L - a) ** 2 / (3 * mat["youngs_modulus"] * I * L)
    rel_disp = delta / L
    weight = mat["density"] * b * h * L

    return {
        "invalid": False,
        "stress_ok": sigma <= mat["yield_strength"],
        "deflection_ok": rel_disp <= target_rd,
        "weight": weight,
        "stress_util": sigma / mat["yield_strength"],
        "deflection_util": rel_disp / target_rd if target_rd > 0 else 0.0,
    }

# ------------------------------
# GRPO Preference Model (High Utilization)
# ------------------------------
def find_best_in_group(designs: List[str], env: Dict) -> int:
    """
    MODIFICATION: This new preference model creates a "sweet spot" score
    that heavily rewards high utilization (near 90%) and penalizes both
    under-utilization (<30%) and over-engineering.
    """
    best_idx = -1
    best_score = -float('inf')

    TARGET_UTIL = 0.9  # The ideal utilization (90%)
    MIN_UTIL = 0.3     # The minimum acceptable utilization (30%)
    
    # Coefficients to balance the score components
    W_UTIL = 100.0 # Strong weight for the utilization score
    W_WEIGHT = 0.1 # Modest penalty for weight

    for i, text in enumerate(designs):
        parsed = parse_design(text)
        if not parsed:
            score = -float('inf')
            continue

        metrics = evaluate_beam_design(parsed, env)
        if metrics["invalid"]:
            score = -float('inf')
        # Hard failure condition
        elif not metrics["stress_ok"] or not metrics["deflection_ok"]:
            score = -1000.0
        else:
            # --- Calculate a score for each utilization metric ---
            
            # Stress Utilization Score
            stress_util = metrics['stress_util']
            if stress_util < MIN_UTIL:
                # Heavy penalty for being under-utilized (linear drop-off)
                stress_score = stress_util / MIN_UTIL
            else:
                # Gaussian-like score that peaks at TARGET_UTIL
                # This rewards getting close to 90% and penalizes moving away
                stress_score = np.exp(-((stress_util - TARGET_UTIL)**2) / (2 * 0.1**2))

            # Deflection Utilization Score
            defl_util = metrics['deflection_util']
            if defl_util < MIN_UTIL:
                # Heavy penalty for being under-utilized
                defl_score = defl_util / MIN_UTIL
            else:
                # Gaussian-like score that peaks at TARGET_UTIL
                defl_score = np.exp(-((defl_util - TARGET_UTIL)**2) / (2 * 0.1**2))

            # --- Combine scores ---
            # We multiply the scores to ensure both must be good
            utilization_score = stress_score * defl_score
            weight_penalty = metrics["weight"] * W_WEIGHT
            
            score = (W_UTIL * utilization_score) - weight_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_idx != -1 else 0

# ------------------------------
# GRPO Loss Calculation (Corrected)
# ------------------------------
def compute_grpo_loss(
    policy: torch.nn.Module,
    query_tensors: List[torch.Tensor],
    response_tensors: List[torch.Tensor],
    best_indices: List[int]
) -> torch.Tensor:
    full_tensors = [torch.cat([q, r.to(q.device)]) for q, r in zip(query_tensors, response_tensors)]
    padded_tensors = pad_sequence(
        full_tensors,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    ).to(DEVICE)
    attention_mask = padded_tensors.ne(tokenizer.pad_token_id).long()
    outputs = policy(
        input_ids=padded_tensors,
        attention_mask=attention_mask
    )
    logits = outputs.logits
    logits_shifted = logits[..., :-1, :].contiguous()
    labels = padded_tensors[..., 1:].contiguous()
    log_probs = F.log_softmax(logits_shifted, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    for i, q in enumerate(query_tensors):
        prompt_len = q.shape[0]
        loss_mask[i, prompt_len-1:-1] = 1.0
    masked_log_probs = token_log_probs * loss_mask[..., 1:].contiguous()
    seq_log_probs = masked_log_probs.sum(dim=-1)
    seq_log_probs = seq_log_probs.view(BATCH_SIZE, N_CANDIDATES)
    scaled_log_probs = seq_log_probs * BETA
    log_softmax_probs = F.log_softmax(scaled_log_probs, dim=-1)
    winner_indices = torch.tensor(best_indices, dtype=torch.long, device=DEVICE)
    loss = -log_softmax_probs.gather(dim=1, index=winner_indices.view(-1, 1)).mean()
    return loss

# ------------------------------
# Evaluation (for tracking progress)
# ------------------------------
class ThreeLineStopper(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.rx = re.compile(r"Material:.*\n\s*Width:.*[0-9].*\n\s*Height:.*[0-9].*\n", re.I)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        resp = input_ids[0][self.prompt_len:]
        if resp.numel() == 0: return False
        text = self.tokenizer.decode(resp)
        return bool(self.rx.search(text))

def generate_with_stopper(model, q_ids: torch.LongTensor) -> torch.LongTensor:
    stop_list = StoppingCriteriaList([ThreeLineStopper(tokenizer, prompt_len=q_ids.shape[0])])
    with torch.no_grad():
        out = model.generate(q_ids.unsqueeze(0).to(DEVICE), stopping_criteria=stop_list, **GEN_KWARGS_EVAL)[0]
    return out.detach().cpu()

def run_eval(eval_model, n_samples=20, is_final_eval=False):
    eval_model.eval()
    success_count = 0
    all_designs = []
    # MODIFICATION: Added lists to track utilization
    stress_utils, defl_utils = [], []
    print_limit = 5
    for i in range(n_samples):
        env = gen_env()
        prompt = prompt_template(env)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        seq = generate_with_stopper(eval_model, ids)
        text = tokenizer.decode(seq[ids.shape[0]:], skip_special_tokens=True).strip()
        
        is_success = False
        design = parse_design(text)
        if design:
            metrics = evaluate_beam_design(design, env)
            if not metrics["invalid"] and metrics["stress_ok"] and metrics["deflection_ok"]:
                success_count += 1
                is_success = True
                # MODIFICATION: Record utilization for successful designs
                stress_utils.append(metrics['stress_util'])
                defl_utils.append(metrics['deflection_util'])
        
        if (is_final_eval or i < print_limit):
            print("-" * 50)
            print(f"Eval Sample {i+1}/{n_samples} | Success: {'✅' if is_success else '❌'}")
            print(f"  Load: {env['load_P']:.0f}N, Length: {env['length']:.2f}m, L/{env['service_ratio']}")
            print(f"  Generated: {repr(text)}")
            if design and not metrics.get("invalid"):
                print(f"  Metrics: Stress OK? {metrics['stress_ok']}, Deflection OK? {metrics['deflection_ok']}")
                print(f"           Stress Util: {metrics['stress_util']:.2%}, Defl Util: {metrics['deflection_util']:.2%}")

        all_designs.append(text)

    eval_model.train()
    
    # MODIFICATION: Calculate and return average utilization
    avg_stress_util = np.mean(stress_utils) if stress_utils else 0.0
    avg_defl_util = np.mean(defl_utils) if defl_utils else 0.0
    
    return {
        "success_rate": success_count / n_samples,
        "avg_stress_util": avg_stress_util,
        "avg_defl_util": avg_defl_util,
        "example_design": all_designs[0] if all_designs else "N/A"
    }

# ------------------------------
# Training Loop
# ------------------------------
print("\n🔎 Initial evaluation...")
initial = run_eval(policy, n_samples=50)
print(f"Initial | Success Rate: {initial['success_rate']:.1%} | Avg Stress Util: {initial['avg_stress_util']:.1%} | Avg Defl Util: {initial['avg_defl_util']:.1%}")

total_loss = 0.0
best_success_rate = initial['success_rate']

print("\n🚀 Starting GRPO Training...")
for ep in range(1, EPOCHS + 1):
    envs = [gen_env() for _ in range(BATCH_SIZE)]
    prompts = [prompt_template(e) for e in envs]
    query_tensors = [tokenizer(p, return_tensors="pt").input_ids.squeeze(0) for p in prompts]

    policy.eval()
    response_tensors_flat = []
    with torch.no_grad():
        for q in query_tensors:
            full_outputs = policy.generate(
                q.unsqueeze(0).to(DEVICE),
                **GEN_KWARGS_GRPO
            )
            responses = [full[q.shape[0]:] for full in full_outputs]
            response_tensors_flat.extend(responses)
    policy.train()

    decoded_responses = tokenizer.batch_decode(response_tensors_flat, skip_special_tokens=True)
    best_indices = []
    valid_in_batch = 0
    for i in range(BATCH_SIZE):
        start_idx = i * N_CANDIDATES
        end_idx = start_idx + N_CANDIDATES
        group = decoded_responses[start_idx:end_idx]
        best_idx_in_group = find_best_in_group(group, envs[i])
        best_indices.append(best_idx_in_group)
        winner_text = group[best_idx_in_group]
        winner_parsed = parse_design(winner_text)
        if winner_parsed:
            m = evaluate_beam_design(winner_parsed, envs[i])
            if not m['invalid'] and m['stress_ok'] and m['deflection_ok']:
                valid_in_batch += 1

    query_tensors_flat = [q for q in query_tensors for _ in range(N_CANDIDATES)]

    loss = compute_grpo_loss(policy, query_tensors_flat, response_tensors_flat, best_indices)
    (loss / GRAD_ACCUM_STEPS).backward()
    total_loss += loss.item()

    if (ep % GRAD_ACCUM_STEPS) == 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if ep % LOG_EVERY == 0:
        avg_loss = total_loss / LOG_EVERY if ep > 1 else total_loss
        success_rate = valid_in_batch / BATCH_SIZE
        print(
            f"Ep {ep:4d} | Avg Loss={avg_loss:7.4f} | Batch Success={success_rate:6.1%} | "
            f"Winner ex: {repr(decoded_responses[best_indices[0]])}"
        )
        total_loss = 0.0

        if ep % (LOG_EVERY * 5) == 0:
            print("\n--- Periodic Evaluation ---")
            eval_results = run_eval(policy, n_samples=20)
            print(f"  > Eval Success: {eval_results['success_rate']:.1%} | Avg Stress: {eval_results['avg_stress_util']:.1%} | Avg Defl: {eval_results['avg_defl_util']:.1%}")
            print("-------------------------\n")
            if eval_results['success_rate'] > best_success_rate:
                best_success_rate = eval_results['success_rate']
                print(f"  🎉 New best success rate! Saving model...")
                save_dir = f"trained_models/gemma-2b-it-grpo_best"
                os.makedirs(save_dir, exist_ok=True)
                policy.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    del query_tensors, response_tensors_flat, query_tensors_flat, decoded_responses, loss
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    elif DEVICE.type == "mps": torch.mps.empty_cache()

print("\n✨ Training finished")

# ------------------------------
# Save + Final Eval
# ------------------------------
save_dir = f"trained_models/gemma-2b-it-grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

print(f"💾 Saving final model to {save_dir}")
policy.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("🔎 Final evaluation (100 samples)...")
final = run_eval(policy, n_samples=100, is_final_eval=True)
print(f"Final  | Success Rate: {final['success_rate']:.1%} | Avg Stress Util: {final['avg_stress_util']:.1%} | Avg Defl Util: {final['avg_defl_util']:.1%}")

print("\n📊 Summary")
print(f"    Initial Success: {initial['success_rate']:.1%}")
print(f"    Final Success  : {final['success_rate']:.1%}")
print(f"\n    Avg Stress Util: {initial['avg_stress_util']:.1%} -> {final['avg_stress_util']:.1%}")
print(f"    Avg Defl Util  : {initial['avg_defl_util']:.1%} -> {final['avg_defl_util']:.1%}")
print(f"\n    Saved to       : {save_dir}")
