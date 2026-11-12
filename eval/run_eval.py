"""Evaluate on the hand-written held-out test boards (never used for training or tuning)."""
import json, sys, time, re, numpy as np
sys.path.insert(0, ".")
from b2a.pipeline import Model, extract_owner, extract_due
from sklearn.metrics import classification_report, adjusted_rand_score, f1_score, accuracy_score

def baseline_type(t):
    s = t.lower()
    if re.search(r"\b(decided|agreed|final|going with|approved|consensus|will keep)\b", s): return "decision"
    if re.search(r"\b(todo|to do|action|next step|follow up|i'll|will|owner|owns)\b", s) or "@" in s: return "action"
    if "?" in s: return "question"
    if re.search(r"\b(slow|broken|bug|issue|problem|confusing|fails|complain|stuck|too)\b", s): return "problem"
    return "idea"

m = Model("b2a/model.json", "b2a/vectors.npz")
test = json.load(open("data/test_boards.json"))
notes = [n for b in test for n in b["notes"]]
gold = [n["type"] for n in notes]
t0 = time.time(); pred = [m.classify(n["text"])[0] for n in notes]; t_cls = (time.time() - t0) / len(notes) * 1000
base = [baseline_type(n["text"]) for n in notes]
R = {}
R["notes"] = len(notes); R["boards"] = len(test)
R["type_accuracy"] = accuracy_score(gold, pred); R["type_macro_f1"] = f1_score(gold, pred, average="macro")
R["baseline_type_accuracy"] = accuracy_score(gold, base); R["baseline_type_macro_f1"] = f1_score(gold, base, average="macro")
rep = classification_report(gold, pred, output_dict=True, zero_division=0)
R["per_type"] = {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3), "f1": round(v["f1-score"], 3), "n": int(v["support"])}
                 for k, v in rep.items() if k in set(gold)}
brep = classification_report(gold, base, output_dict=True, zero_division=0)
R["baseline_action_recall"] = brep["action"]["recall"]; R["baseline_action_precision"] = brep["action"]["precision"]
# owner / due on true action items (end-to-end: only counted when the note was also classified as an action)
acts = [n for n in notes if n["type"] == "action"]
own_ok = due_ok = 0; own_e2e = due_e2e = 0
for n in acts:
    o, d = extract_owner(n["text"]), extract_due(n["text"])
    own_ok += (o == n["owner"]); due_ok += (d == n["due"])
    is_act = m.classify(n["text"])[0] == "action"
    own_e2e += is_act and (o == n["owner"]); due_e2e += is_act and (d == n["due"])
R["action_items"] = len(acts)
R["owner_accuracy"] = own_ok / len(acts); R["due_accuracy"] = due_ok / len(acts)
R["owner_and_due_both_correct_end_to_end"] = sum(1 for n in acts if m.classify(n["text"])[0] == "action"
        and extract_owner(n["text"]) == n["owner"] and extract_due(n["text"]) == n["due"]) / len(acts)
# themes
aris, sizes = [], []
t0 = time.time()
for b in test:
    lab = m.cluster([n["text"] for n in b["notes"]]); aris.append(adjusted_rand_score([n["theme"] for n in b["notes"]], lab))
    sizes.append((len(set(n["theme"] for n in b["notes"])), len(set(lab))))
R["board_ms"] = (time.time() - t0) / len(test) * 1000 + t_cls * len(notes) / len(test)
R["theme_ari_mean"] = float(np.mean(aris)); R["themes_gold_vs_found"] = sizes
R["theme_ari_baseline_one_group"] = 0.0
json.dump(R, open("eval/results.json", "w"), indent=1, default=float)
for k, v in R.items(): print(k, v if not isinstance(v, float) else round(v, 3))
print("\nERRORS:")
for n, p in zip(notes, pred):
    if p != n["type"]: print(f"  gold={n['type']:9s} pred={p:9s} | {n['text']}")
for n in acts:
    o, d = extract_owner(n["text"]), extract_due(n["text"])
    if o != n["owner"] or d != n["due"]: print(f"  owner {o} vs {n['owner']}, due {d} vs {n['due']} | {n['text']}")
