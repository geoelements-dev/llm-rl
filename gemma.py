"""
RL PPO for Small LLM Engineering Design — Code-limit (L/R) + Safe PPO + 3-line stopper

What’s inside:
- Deflection criterion: δ/L ≤ 1/R (default R=350; set USE_RANDOM_LIMIT=True to vary)
- Prompt states the code limit
- Parser accepts optional " m" units
- Deterministic generation (greedy) + per-sample 3-line stopper:
  stops only after a newline following the Height line (prevents premature cutoff)
- PPO stabilized (lower LR, tighter clip, single epoch, stricter target KL)
- Stronger format shaping and tanh reward scaling to keep policy ratios sane
- KL guard to skip update if pre-step KL is too large
- Robust stat logging via stat_mean() to handle array-valued TRL stats
"""

import os, re, json, random, warnings
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

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
print(f"Using device: {DEVICE}  (cuda={torch.cuda.is_available()})")

# ------------------------------
# Model / Tokenizer
# ------------------------------
MODEL_NAME = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.padding_side = "left"

# Avoid device_map="auto" during PPO so grads flow reliably
fp = dict(trust_remote_code=True, torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, **fp).to(DEVICE)
ref    = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, **fp).to(DEVICE)

# Gradient checkpointing + disable cache
base = getattr(policy, "pretrained_model", policy)
if hasattr(base, "gradient_checkpointing_enable"):
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    except TypeError:
        base.gradient_checkpointing_enable()
if hasattr(base.config, "use_cache"):
    base.config.use_cache = False

# Freeze reference explicitly
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

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

# Weight penalty scaling (kept modest so it doesn't dominate)
WEIGHT_PENALTY_COEFFICIENT = 1.0
WEIGHT_SCALE = 1e-3  # scale raw kg so penalty magnitudes are reasonable

# ------------------------------
# Code deflection limit
# ------------------------------
CODE_LIMIT_CHOICES = [240, 350, 360, 480, 600]  # common serviceability ratios
USE_RANDOM_LIMIT = False
DEFAULT_SERVICE_RATIO = 350  # L/350

# ------------------------------
# Generation config
# ------------------------------
MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS = 24  # 3 concise lines fit comfortably here
GEN_KWARGS = dict(
    min_new_tokens=6,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,  # greedy for stability
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# ------------------------------
# PPO Config (stabilized)
# ------------------------------
ppo_cfg = PPOConfig(
    learning_rate=3e-6,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    cliprange=0.05,
    cliprange_value=0.05,
    vf_coef=0.1,
    max_grad_norm=0.5,
    ppo_epochs=2,         # single pass per batch
    target_kl=0.08,       # stricter KL target
)

trainer = PPOTrainer(
    config=ppo_cfg,
    model=policy,
    ref_model=ref,
    tokenizer=tokenizer,
)
print("✓ PPO trainer initialized")

# ------------------------------
# Utils
# ------------------------------
def stat_mean(stats: dict, key: str, default: float = 0.0) -> float:
    """Extract a scalar mean from TRL stats which may be float, Tensor, list, or ndarray."""
    val = stats.get(key, None)
    if val is None:
        return float(default)
    try:
        if hasattr(val, "detach") and hasattr(val, "cpu"):  # torch.Tensor
            v = val.detach().cpu().numpy()
            return float(np.mean(v))
        if isinstance(val, (list, tuple)):
            arr = []
            for x in val:
                if hasattr(x, "detach") and hasattr(x, "cpu"):
                    arr.append(x.detach().cpu().item() if getattr(x, "numel", lambda:1)()==1
                               else float(np.mean(x.detach().cpu().numpy())))
                else:
                    arr.append(float(x))
            return float(np.mean(arr)) if arr else float(default)
        if isinstance(val, np.ndarray):
            return float(np.mean(val))
        return float(val)
    except Exception:
        return float(default)

# ------------------------------
# Task / Environment
# ------------------------------
def pick_service_ratio() -> int:
    return random.choice(CODE_LIMIT_CHOICES) if USE_RANDOM_LIMIT else DEFAULT_SERVICE_RATIO

def gen_env() -> Dict[str, float]:
    R = pick_service_ratio()
    return {
        "load_P": random.uniform(1_000.0, 20_000.0),   # N
        "load_a_fraction": random.uniform(0.1, 0.9),   # position along length
        "length": random.uniform(1.0, 5.0),            # m
        "service_ratio": R,                             # L/R
        "target_relative_displacement": 1.0 / R,       # δ/L ≤ 1/R
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
# Parsing / Reward
# ------------------------------
FORMAT_RX = re.compile(
    r"^\s*Material:\s*([A-Za-z_]+)\s*\n\s*Width:\s*([0-9]*\.?[0-9]+)(?:\s*m)?\s*\n\s*Height:\s*([0-9]*\.?[0-9]+)(?:\s*m)?\s*$",
    re.I
)

# Stopping regex: requires a newline AFTER the Height line (prevents premature stop)
STOPPER_RX = re.compile(
    r"^\s*Material:\s*[A-Za-z_]+\s*\n\s*Width:\s*[0-9]*\.?[0-9]+(?:\s*m)?\s*\n\s*Height:\s*[0-9]*\.?[0-9]+(?:\s*m)?\s*\n$",
    re.I
)

def parse_design(text: str):
    m = FORMAT_RX.match(text.strip())
    if not m:
        return None
    material_raw, w_s, h_s = m.group(1), m.group(2), m.group(3)
    # normalize material name
    material = None
    for k in MATERIAL_PROPERTIES.keys():
        if k.lower().replace("_","") == material_raw.lower().replace("_",""):
            material = k
            break
    if material is None:
        return None
    w = float(w_s); h = float(h_s)
    if w <= 0 or h <= 0:
        return None
    return {"material": material, "width": w, "height": h}

def evaluate_beam_design(design: Dict[str, float], env: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    # strict check
    req = ("material", "width", "height", "length")
    if any(k not in design for k in req):
        return {"invalid": True, "reason": "missing_keys"}, -50.0
    if design["material"] not in MATERIAL_PROPERTIES:
        return {"invalid": True, "reason": "invalid_material"}, -50.0
    if design["width"] <= 0 or design["height"] <= 0 or design["length"] <= 0:
        return {"invalid": True, "reason": "nonpositive_dims"}, -50.0

    mat = MATERIAL_PROPERTIES[design["material"]]
    b, h, L = design["width"], design["height"], design["length"]
    P = env["load_P"]
    a = env["load_a_fraction"] * L
    target_rd = env["target_relative_displacement"]  # = 1/R

    # Section properties
    I = b * h**3 / 12.0
    if I <= 1e-18:
        return {"invalid": True, "reason": "zero_moment"}, -50.0

    # Max bending (SS beam, point load at a)
    M_max = P * a * (L - a) / L
    sigma = abs(M_max * (h / 2) / I)

    # Deflection at load point (SS beam, single P at a)
    delta = P * a**2 * (L - a) ** 2 / (3 * mat["youngs_modulus"] * I * L)
    rel_disp = delta / L

    # Simple weight measure and scaled penalty
    weight = mat["density"] * b * h * L
    weight_penalty = -WEIGHT_PENALTY_COEFFICIENT * weight * WEIGHT_SCALE

    # Penalties for violating limits
    stress_failure_penalty = -100.0 if sigma > mat["yield_strength"] else 0.0
    disp_failure_penalty = -50.0 if rel_disp > target_rd else 0.0

    # Reward for near-limit deflection utilization (prefer 70–90% of limit)
    disp_reward = 0.0
    if rel_disp <= target_rd and target_rd > 0:
        util = rel_disp / target_rd
        if 0.7 <= util <= 0.9:
            disp_reward = 50.0
        elif util < 0.7:
            disp_reward = 25.0 * (util / 0.7)
        else:  # 0.9 < util <= 1.0
            disp_reward = 50.0 * (1.0 - util) / 0.1

    # Gentle reward for using allowable stress (quadratic in utilization)
    stress_eff = 0.0
    if sigma <= mat["yield_strength"]:
        ur = sigma / mat["yield_strength"]
        stress_eff = (ur ** 2) * 15.0

    reward = stress_failure_penalty + disp_failure_penalty + disp_reward + weight_penalty + stress_eff

    return {
        "sigma": sigma,
        "rel_disp": rel_disp,
        "weight": weight,
        "utilization_ratio_stress": sigma / mat["yield_strength"],
        "utilization_ratio_deflection": rel_disp / target_rd if target_rd > 0 else 0.0,
        "service_ratio": env["service_ratio"],
        "invalid": False,
        "reward_components": {
            "stress_failure_penalty": stress_failure_penalty,
            "displacement_failure_penalty": disp_failure_penalty,
            "displacement_reward": disp_reward,
            "weight_penalty": weight_penalty,
            "stress_efficiency_reward": stress_eff,
        },
    }, reward

# Stronger shaping to prefer EXACT format (and nothing else)
def format_bonus(text: str) -> float:
    return 20.0 if FORMAT_RX.match(text.strip()) else -30.0

# ------------------------------
# 3-line Stopping Criteria (per-sample)
# ------------------------------
class ThreeLineStopper(StoppingCriteria):
    """
    Stops ONLY when the generated response (excluding the prompt) matches the
    strict 3-line format AND includes a trailing newline after the Height line.
    """
    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.rx = STOPPER_RX  # anchored + final newline required

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Expect batch size = 1 (we call generate per-sample)
        seq = input_ids[0]
        resp = seq[self.prompt_len:]
        if resp.numel() == 0:
            return False
        text = self.tokenizer.decode(resp, skip_special_tokens=True)
        return bool(self.rx.match(text))

def generate_with_stopper(model, q_ids: torch.LongTensor) -> torch.LongTensor:
    """
    Per-sample generation with greedy decoding + three-line stopper.
    """
    q_ids = q_ids.to(DEVICE)
    stop_list = StoppingCriteriaList([ThreeLineStopper(tokenizer, prompt_len=q_ids.shape[0])])
    out = model.generate(
        q_ids.unsqueeze(0),
        stopping_criteria=stop_list,
        **GEN_KWARGS
    )[0]
    return out.detach().cpu()

# ------------------------------
# Helper: empirical KL on response-only (sanity)
# ------------------------------
@torch.no_grad()
def compute_empirical_kl(policy_model, ref_model, q_list, r_list) -> float:
    vals = []
    p_base = getattr(policy_model, "pretrained_model", policy_model)
    r_base = getattr(ref_model,    "pretrained_model", ref_model)

    p_dev = next(p_base.parameters()).device
    r_dev = next(r_base.parameters()).device

    for q, r in zip(q_list, r_list):
        if r.numel() == 0:
            continue

        # policy
        q_p = q.to(p_dev, non_blocking=True)
        r_p = r.to(p_dev, non_blocking=True)
        x_p = torch.cat([q_p, r_p], dim=0).unsqueeze(0)
        attn_p = torch.ones_like(x_p)
        outp = p_base(x_p, attention_mask=attn_p).logits
        lp = torch.log_softmax(outp[:, :-1, :], dim=-1)[0, -r_p.shape[0]:, :]
        logp = lp.gather(-1, r_p.unsqueeze(-1)).squeeze(-1)

        # ref
        q_r = q.to(r_dev, non_blocking=True)
        r_r = r.to(r_dev, non_blocking=True)
        x_r = torch.cat([q_r, r_r], dim=0).unsqueeze(0)
        attn_r = torch.ones_like(x_r)
        outr = r_base(x_r, attention_mask=attn_r).logits
        lr = torch.log_softmax(outr[:, :-1, :], dim=-1)[0, -r_r.shape[0]:, :]
        logr = lr.gather(-1, r_r.unsqueeze(-1)).squeeze(-1)

        vals.append((logp - logr.to(logp.device)).mean().item())

    return float(np.mean(vals)) if vals else 0.0

# ------------------------------
# Evaluation (baseline/final)
# ------------------------------
def run_eval(eval_model, n_samples=20):
    eval_model.eval()
    rewards, valid = [], 0
    envs_used, designs, responses = [], [], []
    for _ in range(n_samples):
        env = gen_env()
        prompt = prompt_template(env)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_PROMPT_TOKENS - MAX_NEW_TOKENS).input_ids[0]
        with torch.no_grad():
            seq = generate_with_stopper(eval_model, ids)
        r_only = seq[ids.shape[0]:]
        if r_only.numel() == 0:
            r_only = seq[-1:].clone()
        text = tokenizer.decode(r_only, skip_special_tokens=True).strip()
        d = parse_design(text) or {}
        d["length"] = env["length"]
        metrics, rew = evaluate_beam_design(d, env)
        rew += format_bonus(text)
        if not metrics.get("invalid", True):
            valid += 1
        rewards.append(rew)
        envs_used.append(env); designs.append(d); responses.append(text)
    eval_model.train()
    return {
        "success_rate": valid / n_samples,
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "rewards": rewards, "designs": designs, "responses": responses, "envs": envs_used
    }

# ------------------------------
# Training Loop
# ------------------------------
EPOCHS = 600
LOG_EVERY = 10

print("\n🔎 Initial evaluation...")
initial = run_eval(policy, n_samples=20)
print(f"Initial | Success: {initial['success_rate']:.1%} | AvgReward: {initial['avg_reward']:.2f}")

best_avg = -1e9
total_valid = 0

for ep in range(1, EPOCHS + 1):
    # --- rollout
    envs = [gen_env() for _ in range(ppo_cfg.batch_size)]
    prompts = [prompt_template(e) for e in envs]

    # tokenize queries on CPU
    queries = [tokenizer(p, return_tensors="pt", truncation=True,
                         max_length=MAX_PROMPT_TOKENS - MAX_NEW_TOKENS).input_ids.squeeze(0).cpu()
               for p in prompts]

    # Per-sample generate (so stopper doesn't end batch early)
    response_full: List[torch.LongTensor] = []
    decoded: List[str] = []
    raw_rewards: List[float] = []
    valid_in_batch = 0

    policy.eval()
    with torch.no_grad():
        for q_ids in queries:
            full = generate_with_stopper(policy, q_ids)
            response_full.append(full)
    policy.train()

    # strip prompt -> response-only; decode; parse; reward
    response_only_ids: List[torch.LongTensor] = []
    for i, full in enumerate(response_full):
        q = queries[i]
        r_only = full[q.shape[0]:]
        if r_only.numel() == 0:
            r_only = full[-1:].clone()
        response_only_ids.append(r_only)

        text = tokenizer.decode(r_only, skip_special_tokens=True).strip()
        decoded.append(text)
        design = parse_design(text) or {}
        design["length"] = envs[i]["length"]

        metrics, r = evaluate_beam_design(design, envs[i])
        r += format_bonus(text)
        raw_rewards.append(r)
        if not metrics.get("invalid", True):
            valid_in_batch += 1
            total_valid += 1

    # Reward scaling + tiny amplitude for PPO stability
    rewards_for_ppo = []
    for r in raw_rewards:
        r_scaled = np.tanh(r / 40.0) * 0.2  # -> [-0.1, 0.1]
        rewards_for_ppo.append(torch.tensor(float(r_scaled), dtype=torch.float32))

    # empirical pre-step KL (response-only) just for sanity / logging
    kl_pre = compute_empirical_kl(policy, ref, queries, response_only_ids)

    # --- KL guard (skip update if policy drifted too far)
    #if abs(kl_pre) > 0.2:
    #    print(f"  **KL guard tripped ({kl_pre:+.3f}); skipping PPO step this epoch.")
    #    continue

    # --- PPO step (CRITICAL: response-only ids)
    stats = trainer.step(
        [q.to(DEVICE) for q in queries],
        [r.to(DEVICE) for r in response_only_ids],
        rewards_for_ppo
    )

    avg_r = float(np.mean(raw_rewards)) if raw_rewards else 0.0
    succ = valid_in_batch / len(raw_rewards) if raw_rewards else 0.0
    best_avg = max(best_avg, avg_r)

    if ep % LOG_EVERY == 0 or ep == 1:
        approxkl = stat_mean(stats, "ppo/policy/approxkl", 0.0)
        ratio    = stat_mean(stats, "ppo/policy/ratio",    1.0)
        clipfrac = stat_mean(stats, "ppo/policy/clipfrac", 0.0)
        kl_obj   = stat_mean(stats, "objective/kl",        0.0)
        loss_tot = stat_mean(stats, "ppo/loss/total",      0.0)
        print(
            f"Ep {ep:4d} | AvgRew={avg_r:7.2f} | Best={best_avg:7.2f} | "
            f"Valid={valid_in_batch}/{len(raw_rewards)} ({succ:.1%}) | "
            f"KL_pre={kl_pre:+.4f} | approxKL={approxkl:+.4f} | ratio={ratio:.2f} | "
            f"clipfrac={clipfrac:.2f} | KL_obj={kl_obj:+.4f} | Loss={loss_tot:.4f}"
        )
        s = decoded[0] if decoded else ""
        print("  ex:", repr(s[:140]))

    # memory hygiene
    del queries, response_full, response_only_ids, decoded, rewards_for_ppo
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

print("\n✨ Training finished")

# ------------------------------
# Save + Final Eval
# ------------------------------
save_dir = f"trained_models/{MODEL_NAME.split('/')[-1].replace('.','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

print(f"💾 Saving to {save_dir}")
policy.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("🔎 Final evaluation (50 samples)...")
final = run_eval(policy, n_samples=50)
print(f"Final  | Success: {final['success_rate']:.1%} | AvgReward: {final['avg_reward']:.2f}")

# top-5 examples
triples = sorted(zip(final["rewards"], final["designs"], final["responses"], final["envs"]), key=lambda t: t[0], reverse=True)
final["top_examples"] = triples[:5]

summary = {
    "model": MODEL_NAME,
    "epochs": EPOCHS,
    "initial": {"success_rate": initial["success_rate"], "avg_reward": initial["avg_reward"]},
    "final": {"success_rate": final["success_rate"], "avg_reward": final["avg_reward"]},
    "improvements": {
        "success_rate_delta": final["success_rate"] - initial["success_rate"],
        "avg_reward_delta": final["avg_reward"] - initial["avg_reward"],
    },
    "top_examples": [
        {
            "reward": r,
            "design": d,
            "response": resp,
            "env": e
        } for r, d, resp, e in final["top_examples"]
    ],
}
with open(os.path.join(save_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n📊 Summary")
print(f"   Success: {initial['success_rate']:.1%} → {final['success_rate']:.1%}")
print(f"   AvgRew : {initial['avg_reward']:.2f} → {final['avg_reward']:.2f}")
print(f"   Saved  : {save_dir}")
