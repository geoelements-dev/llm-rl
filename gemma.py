"""
RL PPO for Small LLM Engineering Design
- Correct TRL usage: pass response-only tokens to PPOTrainer.step
- Strict output format: exactly "Material/Width/Height" lines (no Length)
- Prints progress every 10 epochs
"""

import os, re, json, random, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datetime import datetime

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
MODEL_NAME = "google/gemma-2b-it"   # or TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.padding_side = "left"

fp = {"trust_remote_code": True}
if DEVICE.type == "cuda":
    fp.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))
else:
    fp.update(dict(torch_dtype=torch.float32))

policy = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, **fp)
ref    = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, **fp)

# Gradient checkpointing + disable cache (saves memory; important for PPO backprop)
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
# PPO Config
# ------------------------------
ppo_cfg = PPOConfig(
    learning_rate=1e-7,             # conservative for small models
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    cliprange=0.1,
    cliprange_value=0.1,
    vf_coef=0.1,
    max_grad_norm=0.5,
    ppo_epochs=1,
    target_kl=0.1,                  # optional guard (won't stop training, but logs)
)

trainer = PPOTrainer(
    config=ppo_cfg,
    model=policy,
    ref_model=ref,
    tokenizer=tokenizer,
)
print("✓ PPO trainer initialized")

# ------------------------------
# Task / Environment
# ------------------------------
MATERIALS = ["Aluminum", "Carbon_Fiber", "Steel", "Wood"]
MATERIAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "Steel": {"density": 7850, "youngs_modulus": 200e9, "yield_strength": 250e6},
    "Aluminum": {"density": 2700, "youngs_modulus": 70e9, "yield_strength": 200e6},
    "Wood": {"density": 600, "youngs_modulus": 10e9, "yield_strength": 40e6},
    "Carbon_Fiber": {"density": 1600, "youngs_modulus": 150e9, "yield_strength": 1500e6},
}
WEIGHT_PENALTY_COEFFICIENT = 1.0

MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS = 70
GEN_KWARGS = dict(
    min_new_tokens=15,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    renormalize_logits=True,
)

def gen_env() -> Dict[str, float]:
    return {
        "load_P": random.uniform(1_000.0, 20_000.0),
        "load_a_fraction": random.uniform(0.1, 0.9),
        "target_relative_displacement": random.uniform(5e-4, 3e-3),
        "length": random.uniform(1.0, 5.0),
    }

def prompt_template(env: Dict[str, float]) -> str:
    mats = random.sample(MATERIALS, len(MATERIALS))
    return (
        "<start_of_turn>user\n"
        "You are a precise engineering assistant. Follow the instructions exactly.\n"
        "Design a rectangular beam for these requirements:\n"
        f"- Length of beam: {env['length']:.2f} m (This is fixed; do NOT output it.)\n"
        f"- Point load: {env['load_P']:.0f} N at position {env['load_a_fraction']:.2f} along length\n"
        f"- Max relative displacement: {env['target_relative_displacement']:.4f}\n"
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
LINE = r"[ \t]*([A-Za-z_]+)[ \t]*"
NUM  = r"[ \t]*([0-9]*\.?[0-9]+)[ \t]*"
FORMAT_RX = re.compile(
    r"^\s*Material:\s*([A-Za-z_]+)\s*\n\s*Width:\s*([0-9]*\.?[0-9]+)\s*\n\s*Height:\s*([0-9]*\.?[0-9]+)\s*$",
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
    target_rd = env["target_relative_displacement"]

    I = b * h**3 / 12.0
    if I <= 1e-12:
        return {"invalid": True, "reason": "zero_moment"}, -50.0

    M_max = P * a * (L - a) / L
    sigma = abs(M_max * (h / 2) / I)
    delta = P * a**2 * (L - a) ** 2 / (3 * mat["youngs_modulus"] * I * L)
    rel_disp = delta / L
    weight = mat["density"] * b * h * L

    stress_failure_penalty = -100.0 if sigma > mat["yield_strength"] else 0.0
    disp_failure_penalty = -50.0 if rel_disp > target_rd else 0.0

    # reward for near-limit displacement utilization (70-90%)
    disp_reward = 0.0
    if rel_disp <= target_rd and target_rd > 0:
        util = rel_disp / target_rd
        if 0.7 <= util <= 0.9:
            disp_reward = 50.0
        elif util < 0.7:
            disp_reward = 25.0 * (util / 0.7)
        else:  # 0.9 < util <= 1
            disp_reward = 50.0 * (1.0 - util) / 0.1

    weight_penalty = -WEIGHT_PENALTY_COEFFICIENT * weight * 0.01

    # gentle reward for using allowable stress (quadratic in utilization)
    stress_eff = 0.0
    if sigma <= mat["yield_strength"]:
        ur = sigma / mat["yield_strength"]
        stress_eff = (ur ** 2) * 15.0

    reward = stress_failure_penalty + disp_failure_penalty + disp_reward + weight_penalty + stress_eff
    return {
        "sigma": sigma, "rel_disp": rel_disp, "weight": weight,
        "utilization_ratio": sigma / mat["yield_strength"],
        "displacement_utilization": rel_disp / target_rd if target_rd > 0 else 0,
        "invalid": False,
        "reward_components": {
            "stress_failure_penalty": stress_failure_penalty,
            "displacement_failure_penalty": disp_failure_penalty,
            "displacement_reward": disp_reward,
            "weight_penalty": weight_penalty,
            "stress_efficiency_reward": stress_eff,
        },
    }, reward

# Tiny shaping to strongly prefer EXACT format (and nothing else)
def format_bonus(text: str) -> float:
    return 5.0 if FORMAT_RX.match(text.strip()) else -5.0

# ------------------------------
# Helper: empirical KL on response-only (sanity)
# ------------------------------
@torch.no_grad()
def compute_empirical_kl(policy_model, ref_model, q_list, r_list) -> float:
    vals = []
    p_base = getattr(policy_model, "pretrained_model", policy_model)
    r_base = getattr(ref_model,    "pretrained_model", ref_model)

    # get actual devices (works whether single-GPU or device_map="auto")
    p_dev = next(p_base.parameters()).device
    r_dev = next(r_base.parameters()).device

    for q, r in zip(q_list, r_list):
        if r.numel() == 0:
            continue

        # --- policy side
        q_p = q.to(p_dev, non_blocking=True)
        r_p = r.to(p_dev, non_blocking=True)
        x_p = torch.cat([q_p, r_p], dim=0).unsqueeze(0)
        attn_p = torch.ones_like(x_p)
        outp = p_base(x_p, attention_mask=attn_p).logits
        lp = torch.log_softmax(outp[:, :-1, :], dim=-1)[0, -r_p.shape[0]:, :]
        logp = lp.gather(-1, r_p.unsqueeze(-1)).squeeze(-1)

        # --- ref side
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
            seq = eval_model.generate(ids.unsqueeze(0).to(DEVICE), **GEN_KWARGS)[0].cpu()
        r_only = seq[ids.shape[0]:]
        if r_only.numel() == 0:
            r_only = seq[-1:].clone()
        text = tokenizer.decode(r_only, skip_special_tokens=True).strip()
        d = parse_design(text)
        if d is None:
            d = {}
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

    # tokenize queries on CPU; TRL will move as needed
    queries = [tokenizer(p, return_tensors="pt", truncation=True,
                         max_length=MAX_PROMPT_TOKENS - MAX_NEW_TOKENS).input_ids.squeeze(0).cpu()
               for p in prompts]

    policy.eval()
    with torch.no_grad():
        full_out = trainer.generate(queries, **GEN_KWARGS)
    # normalize to CPU to avoid surprises elsewhere
    full_out = [t.detach().cpu() for t in full_out]
    policy.train()

    # strip prompt -> response-only; decode; parse; reward
    response_only_ids: List[torch.LongTensor] = []
    decoded: List[str] = []
    raw_rewards: List[float] = []
    valid_in_batch = 0

    for i, full in enumerate(full_out):
        q = queries[i]
        r_only = full[q.shape[0]:]
        if r_only.numel() == 0:
            r_only = full[-1:].clone()  # ensure at least 1 token
        response_only_ids.append(r_only)

        text = tokenizer.decode(r_only, skip_special_tokens=True).strip()
        decoded.append(text)
        design = parse_design(text) or {}
        design["length"] = envs[i]["length"]

        metrics, r = evaluate_beam_design(design, envs[i])
        r += format_bonus(text)  # shape toward exact format
        raw_rewards.append(r)
        if not metrics.get("invalid", True):
            valid_in_batch += 1
            total_valid += 1

    # simple, stable reward scaling (no double-normalization)
    # clip to [-2, 1] for PPO stability
    rewards_for_ppo = []
    for r in raw_rewards:
        rn = r / 100.0
        rc = max(min(rn, 1.0), -2.0)
        rewards_for_ppo.append(torch.tensor(rc, dtype=torch.float32))

    # empirical pre-step KL (response-only) just for sanity / logging
    kl_pre = compute_empirical_kl(policy, ref, queries, response_only_ids)

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
        kl_obj = float(stats.get("objective/kl", 0.0))
        loss_tot = float(stats.get("loss/total", 0.0))
        print(
            f"Ep {ep:4d} | AvgRew={avg_r:7.2f} | Best={best_avg:7.2f} | "
            f"Valid={valid_in_batch}/{len(raw_rewards)} ({succ:.1%}) | "
            f"KL_pre={kl_pre:+.4f} | KL_obj={kl_obj:+.4f} | Loss={loss_tot:.4f}"
        )
        # show one sample (truncated) to confirm exact format
        s = decoded[0] if decoded else ""
        print("  ex:", repr(s[:140]))

    # memory hygiene
    del queries, full_out, response_only_ids, decoded, rewards_for_ppo
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
