"""Generate the TRAINING boards. The test set (data/test_boards.json) is hand-written separately
and never produced by this script, so test phrasing is unseen at training time."""
import json, random
random.seed(7)
THEMES = {
 "onboarding": ["the signup flow","the welcome email","the product tour","the first-run checklist","account setup","SSO login on day one"],
 "pricing": ["the pricing page","annual plans","the free tier limits","seat-based billing","the upgrade prompt","discount codes"],
 "performance": ["page load time","the export job","search latency","large file uploads","the sync engine","mobile startup time"],
 "support": ["the help center","ticket backlog","live chat","the refund process","support macros","escalation paths"],
 "mobile": ["the iOS app","push notifications","offline mode","the Android widget","tablet layout","app store reviews"],
 "analytics": ["the KPI dashboard","event tracking","weekly active users","the funnel report","data freshness","cohort retention"],
 "marketing": ["the launch blog post","the webinar series","paid search ads","the newsletter","case studies","social posts"],
 "hiring": ["the interview loop","job descriptions","referral bonuses","the take-home task","new hire onboarding","recruiter sync"],
 "security": ["two-factor auth","the audit log","password reset","admin permissions","SOC 2 evidence","data retention"],
 "collaboration": ["shared templates","comment threads","real-time cursors","the share dialog","version history","guest access"],
 "ai_features": ["AI summaries","the prompt box","auto-generated diagrams","smart suggestions","AI usage limits","answer accuracy"],
 "events": ["the offsite agenda","venue booking","the team dinner","travel budget","the keynote slot","swag order"],
 "sales": ["the demo script","enterprise trials","the CRM pipeline","sales enablement deck","renewal calls","procurement questions"],
 "content": ["the style guide","blog editorial calendar","video tutorials","the glossary","translation queue","SEO keywords"],
 "infrastructure": ["the staging environment","CI build times","database backups","the on-call rotation","cloud costs","error alerts"],
}
NAMES=["Priya","Marcus","Elena","Jordan","Wei","Sam","Aisha","Tom","Lucia","Ben","Nora","Raj","Kim","Diego","Hana","Olu"]
DATES=["Friday","Monday","next week","EOD Thursday","tomorrow","end of month","next sprint","11/20","Dec 5","Q1"]
VERBS=["audit","fix","draft","review","redesign","test","update","clean up","document","prototype","measure","simplify"]
T = {
 "action":[ "{name} to {verb} {obj} by {date}", "AI: {name} will {verb} {obj}", "@{name} {verb} {obj} by {date}",
   "Follow up: {verb} {obj} ({name}, {date})", "{verb} {obj} - owner {name}", "TODO: {verb} {obj}", "Next step: {verb} {obj} before {date}",
   "I'll {verb} {obj} by {date}", "{name}: {verb} {obj}", "Action - {verb} {obj}, due {date}", "{name} owns {verb}ing {obj}"],
 "idea":[ "What if we {verb} {obj}?", "Idea: bundle {obj} with {obj2}", "Try a lighter version of {obj}", "Could add a shortcut to {obj}",
   "Make {obj} part of the premium plan", "Maybe merge {obj} and {obj2}", "Let's explore a new take on {obj}", "Gamify {obj}",
   "Personalize {obj} per team", "A template gallery for {obj} would be cool", "Use AI to suggest {obj}"],
 "problem":[ "{obj} is confusing", "Users drop off at {obj}", "Too many steps in {obj}", "{obj} keeps breaking", "No clear owner for {obj}",
   "{obj} is slow", "Customers complain about {obj}", "{obj} takes forever", "Nobody can find {obj}", "{obj} is out of date",
   "We keep missing deadlines on {obj}", "{obj} errors spiked last week"],
 "question":[ "Why do people skip {obj}?", "Who owns {obj}?", "How do we measure success for {obj}?", "Is {obj} in scope for this quarter?",
   "Do we have data on {obj}?", "Can legal sign off on {obj}?", "What's the budget for {obj}?", "Should {obj} be opt-in?",
   "When did {obj} last change?", "Does {obj} work for enterprise accounts?"],
 "decision":[ "Decided: {obj} ships in v1", "Agreed to drop {obj} for now", "Going with {obj} over {obj2}", "We will keep {obj} as is",
   "Final call: {obj} moves to next quarter", "Approved: budget for {obj}", "Consensus - pause {obj}", "{obj} is locked for launch",
   "Team voted to prioritize {obj}", "Decision: sunset {obj}"],
}
def fill(tpl, theme_objs):
    obj, obj2 = random.sample(theme_objs, 2)
    name, date, verb = random.choice(NAMES), random.choice(DATES), random.choice(VERBS)
    txt = tpl.format(obj=obj, obj2=obj2, name=name, date=date, verb=verb)
    if txt.startswith("the "): txt = "The " + txt[4:]
    owner = None; due = None
    if "{name}" in tpl: owner = name.lower()
    if tpl.startswith("I'll"): owner = "author"
    if "{date}" in tpl: due = date.lower()
    if random.random() < 0.15: txt = txt.lower()
    if random.random() < 0.08: txt = txt.rstrip("?.")
    return txt, owner, due
boards=[]
for b in range(220):
    themes = random.sample(list(THEMES), random.choice([3,3,4]))
    notes=[]
    for th in themes:
        for _ in range(random.randint(4,6)):
            typ = random.choices(list(T), weights=[3,3,3,2,1.5])[0]
            txt, owner, due = fill(random.choice(T[typ]), THEMES[th])
            notes.append({"text":txt,"type":typ,"theme":th,
                          "owner": owner if typ=="action" else None, "due": due if typ=="action" else None})
    random.shuffle(notes)
    boards.append({"board":f"train-{b:03d}","notes":notes})
json.dump(boards, open("data/train_boards.json","w"), indent=1)
print(len(boards), sum(len(b["notes"]) for b in boards))
