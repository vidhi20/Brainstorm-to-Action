"""Train the note-type classifier and tune the theme threshold on TRAINING boards only."""
import json, gzip, math, numpy as np, sys
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, "."); from b2a.pipeline import features, tokens, STOP, Model

train = json.load(open("data/train_boards.json"))
natural = json.load(open("data/train_natural.json"))
syn_notes = [n for b in train for n in b["notes"]]
nat_notes = [n for b in natural for n in b["notes"]]
notes = syn_notes + nat_notes
# 1) word vectors: keep the 30k most frequent GloVe words (pretrained, general English)
words, vecs = [], []
with gzip.open("/tmp/glove50.gz", "rt") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 30000: break
        p = line.rstrip().split(" "); words.append(p[0]); vecs.append([float(x) for x in p[1:]])
vecs = np.array(vecs, dtype=np.float32)
np.savez("b2a/vectors.npz", words=np.array(words), vecs=vecs)
# word rarity weight from GloVe rank (rarer words carry more topic signal)
widf = {w: math.log(2 + i / 50) for i, w in enumerate(words)}
# 2) sparse vocab
df = Counter(f for n in notes for f in set(features(n["text"])))
vocab = [f for f, c in df.items() if c >= 2]
N = len(notes); idf = [math.log((1 + N) / (1 + df[f])) + 1 for f in vocab]
# down-weight words that show up across 3+ different themes in training (template / generic words carry no topic)
from collections import defaultdict
spread = defaultdict(set)
for n in notes:
    for w in set(tokens(n["text"])): spread[w].add(n["theme"])
for w, th in spread.items():
    if len(th) >= 3 and w in widf: widf[w] = 0.05
meta = {"vocab": vocab, "idf": idf, "labels": ["action","decision","idea","problem","question"], "emb_scale": 0.6,
        "theme_threshold": 0.5, "widf_words": words, "widf": [widf[w] for w in words], "coef": [], "bias": []}
json.dump(meta, open("b2a/model.json", "w"))
m = Model("b2a/model.json", "b2a/vectors.npz")
Xs = np.array([m.featurize(n["text"]) for n in syn_notes]); ys = [n["type"] for n in syn_notes]
# model selection: leave-one-board-out CV on the NATURAL training boards only (test boards untouched)
from sklearn.metrics import f1_score
def fit(C, wnat, nat):
    Xn = np.array([m.featurize(n["text"]) for n in nat]); yn = [n["type"] for n in nat]
    X = np.vstack([Xs, Xn]) if len(nat) else Xs; y = ys + yn
    w = np.r_[np.ones(len(ys)), np.full(len(yn), wnat)]
    return LogisticRegression(C=C, max_iter=3000).fit(X, y, sample_weight=w)
best = None
for C in [1.0, 4.0, 16.0]:
    for wnat in [0.0, 5.0, 15.0, 30.0]:
        yt, yp = [], []
        for i, b in enumerate(natural):
            clf = fit(C, wnat, [n for j, bb in enumerate(natural) if j != i for n in bb["notes"]])
            yt += [n["type"] for n in b["notes"]]; yp += list(clf.predict(np.array([m.featurize(n["text"]) for n in b["notes"]])))
        f = f1_score(yt, yp, average="macro"); print(f"  CV C={C} natural_weight={wnat} macroF1={f:.3f}")
        if best is None or f > best[0]: best = (f, C, wnat)
print("selected", best)
clf = fit(best[1], best[2], nat_notes)
meta["labels"] = list(clf.classes_); meta["coef"] = clf.coef_.round(5).tolist(); meta["bias"] = clf.intercept_.round(5).tolist()
# 3) tune the theme-merge threshold on training boards (ARI)
m = Model.__new__(Model); json.dump(meta, open("b2a/model.json", "w")); m.__init__("b2a/model.json", "b2a/vectors.npz")
best = None
for th in np.arange(0.30, 0.91, 0.05):
    m.theme_threshold = th
    ari = np.mean([adjusted_rand_score([n["theme"] for n in b["notes"]], m.cluster([n["text"] for n in b["notes"]])) for b in natural])
    if best is None or ari > best[1]: best = (round(float(th), 2), ari)
meta["theme_threshold"] = best[0]
json.dump(meta, open("b2a/model.json", "w"))
print("train notes", N, "vocab", len(vocab), "theme threshold", best)
