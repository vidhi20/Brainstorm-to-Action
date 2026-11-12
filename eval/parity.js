// Check the browser engine matches the Python evaluation on the held-out test set
const B2A = require("../web/engine.js"), fs = require("fs");
global.atob = s => Buffer.from(s, "base64").toString("binary");
B2A.load(JSON.parse(fs.readFileSync("web/model.web.json")));
const notes = JSON.parse(fs.readFileSync("data/test_boards.json")).flatMap(b => b.notes);
let ok = 0, act = 0, actHit = 0, both = 0;
for (const n of notes) {
  const r = B2A.run([n.text])[0]; ok += r.type === n.type;
  if (n.type === "action") { act++; if (r.type === "action") { actHit++; if (r.owner === n.owner && r.due === n.due) both++; } }
}
console.log(`browser engine: type accuracy ${(ok / notes.length).toFixed(3)}, action recall ${(actHit / act).toFixed(3)}, owner+due e2e ${(both / act).toFixed(3)}`);
