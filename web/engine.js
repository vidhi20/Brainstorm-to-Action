// Brainstorm to Action: browser engine (port of b2a/pipeline.py; same features, same model weights)
const B2A = (() => {
  const STOP = new Set(`a an the and or but if of to in on at for from by with without about into over under is are was were be been being
it its this that these those we our us you your they their them i me my he she his her as so than too very can could should would will
just do does did not no yes all any some more most less much many each per vs via up down out off again then there here when where why
how what which who whom whose also only even still maybe let lets let's make get go use using new next last week weeks day days time`.split(/\s+/));
  const DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun";
  const MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec";
  const DATE_RE = new RegExp(`\\b(eod (?:${DAYS})|end of (?:the )?month|end of (?:the )?week|next (?:week|sprint|release|month|quarter)|tomorrow|today|eod|q[1-4]|\\d{1,2}/\\d{1,2}|(?:${MONTHS})[a-z]* \\d{1,2}|(?:${DAYS}))\\b`, "i");
  const DAY_FULL = {mon:"monday",tue:"tuesday",tues:"tuesday",wed:"wednesday",thu:"thursday",thur:"thursday",thurs:"thursday",fri:"friday",sat:"saturday",sun:"sunday"};
  const OWNER_BLOCK = new Set(`set create estimate profile todo next follow action ai we team everyone someone send add build write draft review fix update
test check run ask book file pull share make clean document measure prototype simplify audit redesign decided agreed final decision
consensus approved going i ill i'll the a`.split(/\s+/));
  const OWNER_VERBS = new Set("to will owns own drafts books sets builds pulls mocks writes sends runs leads handles creates checks updates reviews".split(" "));
  const tokens = t => t.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || [];
  function features(text) {
    const t = tokens(text), f = [...t];
    for (let i = 0; i + 1 < t.length; i++) f.push(t[i] + "_" + t[i + 1]);
    const s = text.trim();
    if (s.includes("?")) f.push("__qmark");
    if (s.includes("@")) f.push("__at");
    if (/^[A-Za-z]+\s*[:\-]/.test(s)) f.push("__lead_label");
    if (DATE_RE.test(s)) f.push("__date");
    if (t.length) f.push("__first_" + t[0]);
    return f;
  }
  function extractDue(text) {
    const m = text.match(DATE_RE); if (!m) return null;
    let d = m[1].toLowerCase().replace("end of the ", "end of ");
    if (DAY_FULL[d]) d = DAY_FULL[d];
    const m2 = d.match(new RegExp(`^(${MONTHS})[a-z]* (\\d{1,2})`));
    if (m2) d = m2[1].slice(0, 3) + " " + m2[2];
    return d;
  }
  function extractOwner(text) {
    const s = text.trim(); let m;
    if ((m = s.match(/@([A-Za-z]+)/))) return m[1].toLowerCase();
    if (/^(i'll|i will|ill|i'm going to)\b/i.test(s)) return "author";
    if ((m = s.match(/\(([A-Z][a-z]+),/))) return m[1].toLowerCase();
    if ((m = s.match(/\bowner:? ([A-Za-z]+)/i))) return m[1].toLowerCase();
    const t = s.match(/[A-Za-z']+|[:\-]/g) || [];
    if (t.length >= 2 && !OWNER_BLOCK.has(t[0].toLowerCase()) && !STOP.has(t[0].toLowerCase()))
      if (t[1] === ":" || t[1] === "-" || OWNER_VERBS.has(t[1].toLowerCase())) return t[0].toLowerCase();
    return null;
  }
  let M = null;
  function load(model) {
    const bin = atob(model.vecs_b64), q = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) q[i] = (bin.charCodeAt(i) << 24) >> 24;
    M = { ...model, q, vocabIdx: new Map(model.vocab.map((w, i) => [w, i])), wordIdx: new Map(model.words.map((w, i) => [w, i])) };
  }
  function embed(text) {
    const d = M.dim, v = new Float64Array(d);
    for (const w of tokens(text)) {
      if (STOP.has(w) || /^\d+$/.test(w) || !M.wordIdx.has(w)) continue;
      const i = M.wordIdx.get(w), wt = M.widf[i];
      for (let k = 0; k < d; k++) v[k] += M.q[i * d + k] * M.vec_scale * wt;
    }
    const n = Math.hypot(...v); if (n) for (let k = 0; k < d; k++) v[k] /= n;
    return v;
  }
  function classify(text) {
    const V = M.vocab.length, x = new Map();
    for (const f of features(text)) if (M.vocabIdx.has(f)) { const i = M.vocabIdx.get(f); x.set(i, (x.get(i) || 0) + 1); }
    let n = 0; for (const [i, c] of x) { const val = (1 + Math.log(c)) * M.idf[i]; x.set(i, val); n += val * val; }
    n = Math.sqrt(n); const e = embed(text);
    const z = M.labels.map((_, k) => {
      let s = M.bias[k]; const row = M.coef[k];
      for (const [i, val] of x) s += row[i] * (n ? val / n : 0);
      for (let j = 0; j < M.dim; j++) s += row[V + j] * e[j] * M.emb_scale;
      return s;
    });
    const mx = Math.max(...z), p = z.map(s => Math.exp(s - mx)), sum = p.reduce((a, b) => a + b);
    const k = p.indexOf(Math.max(...p));
    return { type: M.labels[k], confidence: p[k] / sum };
  }
  function run(lines) {
    return lines.map(t => t.trim()).filter(Boolean).map(text => {
      const c = classify(text);
      return { text, ...c, owner: c.type === "action" ? extractOwner(text) : null, due: c.type === "action" ? extractDue(text) : null };
    });
  }
  return { load, run, classify, extractOwner, extractDue };
})();
if (typeof module !== "undefined") module.exports = B2A;
