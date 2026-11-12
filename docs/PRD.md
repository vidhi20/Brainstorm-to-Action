# PRD: Brainstorm to Action

**Author:** Vidhi Parekh  |  **Status:** v1 shipped as a browser prototype  |  **Last updated:** Nov 2026

## 1. Problem
Brainstorms and retros on digital whiteboards produce dozens of sticky notes, but the output that matters (who does what, by when, and what was decided) is buried among ideas and complaints. Someone has to re-read the board after the meeting and type up the follow-ups. When nobody does, action items are lost and decisions get re-debated.

**Problem statement:** Teams leave brainstorms without a clear, owned action list, because sorting sticky notes into tasks, decisions and questions is manual work that happens after the meeting, if at all.

## 2. Users
- **Primary: the facilitator** (PM, designer, team lead) who runs the session and is on the hook for follow-up.
- **Secondary: participants** who need to know what they own and what was decided.

## 3. Hypotheses to validate
- H1: Facilitators spend meaningful time after each session turning notes into follow-ups. (Validate with 8 to 10 facilitator interviews.)
- H2: A meaningful share of action items on a board have no owner or no date. (Validate by auditing 20 real exported boards.)
- H3: If unowned items are flagged *during* the meeting, teams assign them on the spot.

## 4. Solution (v1)
One click turns a board into: an action list with owner and due date, flags for missing owners and dates, and grouped decisions, open questions, problems and ideas. Copy the action list as a checklist.

**Key product decisions**
- **Runs in the browser, no LLM call.** $0 per board, no data leaves the device (important for enterprise boards), and results in milliseconds. Trade-off: lower accuracy on ambiguous notes than a large language model.
- **Action items first.** The eval showed the model is strongest exactly where the value is: action items (F1 0.85) and owner and date (25 of 28 fully right).
- **Theme grouping cut from v1.** Automatic topic clustering averaged ARI 0.02 on held-out boards; showing wrong groups would erode trust. Keep human grouping; revisit in v2.
- **Confidence shown on every note** so users can spot and fix low-confidence calls.

**Non-goals (v1):** writing to Jira or calendars, real-time co-editing, summarizing the meeting.

## 5. Success metrics
- **North star:** share of action items that have an owner and a due date within 24 hours of the session.
- **Adoption:** % of boards where the action list is copied or exported; repeat use per facilitator per month.
- **Quality guardrails (offline eval gate before every model release):** action recall at least 85%, action precision at least 80%, owner and date both right at least 85%.
- **Trust guardrail:** % of notes users re-label by hand (target under 15%).

## 6. Results so far (offline, held-out test set of 124 notes, 8 boards)
Action recall 89.3% vs 35.7% for keyword rules; action precision 80.6%; owner and date both right 89.3%; overall 5-type accuracy 67.7% vs 54.0%; about 3 ms for the first 16-note board in the browser, under 0.1 ms once warmed up. Weakest: idea vs decision (F1 0.44 and 0.45).

## 7. Roadmap
- **v1.1:** let users correct a label in one click, and log corrections as new training data.
- **v2:** optional LLM mode for idea vs decision and for theme grouping, gated on the same eval set (must beat v1 on macro F1 and keep action recall at least 85%), with cost per board tracked.
- **v2:** push action items to task trackers; remind owners before due dates.
- **Validation:** run H1 to H3 with real facilitators and real exported boards before investing in v2.

## 8. Risks
- Synthetic and author-written eval data may overstate accuracy on real boards. Mitigation: real-board audit (H2) doubles as a new test set.
- Owner rules miss unusual phrasing ("ping Sam about it"). Mitigation: correction logging (v1.1).
