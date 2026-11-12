"""Brainstorm to Action: core pipeline (inference). Pure numpy so it ports 1:1 to the browser demo."""
import re, json, numpy as np

STOP = set("""a an the and or but if of to in on at for from by with without about into over under is are was were be been being
it its this that these those we our us you your they their them i me my he she his her as so than too very can could should would will
just do does did not no yes all any some more most less much many each per vs via up down out off again then there here when where why
how what which who whom whose also only even still maybe let lets let's make get go use using new next last week weeks day days time
""".split())
DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
DATE_RE = re.compile(r"\b(eod (?:%s)|end of (?:the )?month|end of (?:the )?week|next (?:week|sprint|release|month|quarter)|tomorrow|today|eod|"
                     r"q[1-4]|\d{1,2}/\d{1,2}|(?:%s)[a-z]* \d{1,2}|(?:%s))\b" % (DAYS, MONTHS, DAYS), re.I)
DAY_FULL = {"mon":"monday","tue":"tuesday","tues":"tuesday","wed":"wednesday","thu":"thursday","thur":"thursday","thurs":"thursday",
            "fri":"friday","sat":"saturday","sun":"sunday"}
OWNER_BLOCK = set("""set create estimate profile todo next follow action ai we team everyone someone send add build write draft review fix update
test check run ask book file pull share make clean document measure prototype simplify audit redesign decided agreed final decision
consensus approved going i ill i'll the a""".split())
OWNER_VERBS = set("to will owns own drafts books sets builds pulls mocks writes sends runs leads handles creates checks updates reviews".split())

def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

def features(text):
    """Sparse feature list (strings) for the note-type classifier."""
    t = tokens(text); f = list(t) + [a + "_" + b for a, b in zip(t, t[1:])]
    s = text.strip()
    if "?" in s: f.append("__qmark")
    if "@" in s: f.append("__at")
    if re.match(r"^[A-Za-z]+\s*[:\-]", s): f.append("__lead_label")
    if DATE_RE.search(s): f.append("__date")
    if t: f.append("__first_" + t[0])
    return f

def extract_due(text):
    m = DATE_RE.search(text)
    if not m: return None
    d = m.group(1).lower()
    d = re.sub(r"end of the ", "end of ", d)
    if d in DAY_FULL: d = DAY_FULL[d]
    m2 = re.match(r"(%s)[a-z]* (\d{1,2})" % MONTHS, d)
    if m2: d = m2.group(1)[:3] + " " + m2.group(2)
    return d

def extract_owner(text):
    s = text.strip()
    m = re.search(r"@([A-Za-z]+)", s)
    if m: return m.group(1).lower()
    if re.match(r"^(i'll|i will|ill|i'm going to)\b", s, re.I): return "author"
    m = re.search(r"\(([A-Z][a-z]+),", s)
    if m: return m.group(1).lower()
    m = re.search(r"\bowner:? ([A-Za-z]+)", s, re.I)
    if m: return m.group(1).lower()
    t = re.findall(r"[A-Za-z']+|[:\-]", s)
    if len(t) >= 2 and t[0].lower() not in OWNER_BLOCK and t[0].lower() not in STOP:
        if t[1] in (":", "-") or t[1].lower() in OWNER_VERBS: return t[0].lower()
    return None

class Model:
    def __init__(self, path_model, path_vecs):
        m = json.load(open(path_model))
        self.vocab = {w: i for i, w in enumerate(m["vocab"])}; self.idf = np.array(m["idf"])
        self.coef = np.array(m["coef"]); self.bias = np.array(m["bias"]); self.labels = m["labels"]
        self.emb_scale = m["emb_scale"]; self.theme_threshold = m["theme_threshold"]
        v = np.load(path_vecs); self.words = {w: i for i, w in enumerate(v["words"])}; self.vecs = v["vecs"]
        self.widf = dict(zip(m["widf_words"], m["widf"]))

    def embed(self, text):
        ws = [w for w in tokens(text) if w not in STOP and w in self.words and not w.isdigit()]
        if not ws: return np.zeros(self.vecs.shape[1])
        wts = np.array([self.widf.get(w, 8.0) for w in ws])
        v = (self.vecs[[self.words[w] for w in ws]] * wts[:, None]).sum(0)
        n = np.linalg.norm(v); return v / n if n else v

    def featurize(self, text):
        x = np.zeros(len(self.vocab))
        for f in features(text):
            if f in self.vocab: x[self.vocab[f]] += 1
        x = np.where(x > 0, 1 + np.log(np.maximum(x, 1)), 0) * self.idf
        n = np.linalg.norm(x); x = x / n if n else x
        return np.concatenate([x, self.embed(text) * self.emb_scale])

    def classify(self, text):
        z = self.coef @ self.featurize(text) + self.bias
        p = np.exp(z - z.max()); p /= p.sum(); k = int(p.argmax())
        return self.labels[k], float(p[k])

    def cluster(self, texts):
        """Average-linkage agglomerative clustering on meaning vectors; stops below a learned similarity threshold."""
        E = np.array([self.embed(t) for t in texts]); S = E @ E.T
        clusters = [[i] for i in range(len(texts))]
        while len(clusters) > 1:
            best, pair = -9, None
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    s = S[np.ix_(clusters[a], clusters[b])].mean()
                    if s > best: best, pair = s, (a, b)
            if best < self.theme_threshold: break
            a, b = pair; clusters[a] += clusters[b]; del clusters[b]
        lab = [0] * len(texts)
        for k, c in enumerate(clusters):
            for i in c: lab[i] = k
        return lab

    def name_theme(self, texts):
        cnt = {}
        for t in texts:
            for w in set(tokens(t)):
                if w in STOP or w.isdigit() or len(w) < 3 or w in self.idf_blacklist: continue
                cnt[w] = cnt.get(w, 0) + self.widf.get(w, 8.0)
        top = sorted(cnt, key=lambda w: -cnt[w])[:2]
        return " / ".join(w.capitalize() for w in top) or "Other"
    idf_blacklist = set("users people team customers want need know think like good really".split())

    def run(self, texts):
        """Full board -> themes, typed notes, action items with owner and due date."""
        out = []
        for t in texts:
            typ, conf = self.classify(t)
            out.append({"text": t, "type": typ, "confidence": round(conf, 2),
                        "owner": extract_owner(t) if typ == "action" else None,
                        "due": extract_due(t) if typ == "action" else None})
        lab = self.cluster(texts)
        themes = {}
        for n, l in zip(out, lab): themes.setdefault(l, []).append(n)
        return [{"theme": self.name_theme([n["text"] for n in ns]), "notes": ns} for ns in themes.values()]
