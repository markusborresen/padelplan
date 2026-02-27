"""
Padel scheduler (1 bane) for 4–8 spillere.

Mål:
- Når matematisk mulig: alle er på lag med alle nøyaktig én gang + likt antall kamper
- Ellers: likt antall kamper + maksimer unike lagkamerater (og få repeats)

Output (i samme mappe som scriptet kjøres fra):
- padelplan_YYYY-MM-DD.html  (mobilvennlig/kompakt + live poeng + husker valg via localStorage)
- padelplan_YYYY-MM-DD.csv
- index.html (kopi av dagens HTML, praktisk for GitHub Pages)

Poeng:
- Vinnerlagets to spillere får +1 hver.
- Valg av vinner per kamp (A/B) lagres i nettleseren (localStorage) per PLAN_ID (dato).
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import shutil
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

# -----------------------------
# Konfig
# -----------------------------
RANDOM_SEED = 42

SEARCH_RESTARTS = 180       # flere = bedre, men tregere
LOCAL_STEPS = 3500          # flere = bedre, men tregere
CANDIDATE_SAMPLE = 1200     # hvis mange kandidater, sampler vi for fart

# Vekter (lavere score = bedre)
W_PLAY_BALANCE = 10.0       # (skal i praksis være 0 hvis vi velger M riktig)
W_TEAMMATE_MISSING = 25.0   # straff for teammate-par som ikke forekommer (viktig)
W_TEAMMATE_REPEAT = 6.0     # straff for å være på lag igjen
W_OPP_REPEAT = 2.0          # straff for å møte samme motstander igjen
W_CONSEC_REST = 1.2         # straff for hvile mange kamper på rad

# Når perfekt løsning er mulig: press hardt mot "nøyaktig 1 gang per teammate-par"
W_PERFECT_DEVIATION = 40.0  # straff for |count - 1| per teammate-par i perfekt-modus

# Filnavn / plan-id (brukes også som localStorage-nøkkel i HTML)
TODAY = date.today().strftime("%Y-%m-%d")
PLAN_ID = TODAY  # kan endres om du vil skille mandag/tirsdag
OUTPUT_HTML = f"padelplan_{TODAY}.html"
OUTPUT_CSV = f"padelplan_{TODAY}.csv"
INDEX_HTML = "index.html"


# -----------------------------
# Typer
# -----------------------------
Player = str
Team = Tuple[Player, Player]


@dataclass(frozen=True)
class Match:
    a: Team
    b: Team

    def players(self) -> Tuple[Player, Player, Player, Player]:
        return (self.a[0], self.a[1], self.b[0], self.b[1])

    def resting(self, all_players: List[Player]) -> List[Player]:
        in_match = set(self.players())
        return [p for p in all_players if p not in in_match]


# -----------------------------
# Hjelpere
# -----------------------------
def pair_key(x: Player, y: Player) -> Tuple[Player, Player]:
    return (x, y) if x < y else (y, x)


def normalize_team(t: Tuple[Player, Player]) -> Team:
    return tuple(sorted(t))  # type: ignore


def normalize_match(t1: Team, t2: Team) -> Match:
    a = normalize_team(t1)
    b = normalize_team(t2)
    return Match(a=min(a, b), b=max(a, b))


# -----------------------------
# Kandidatmatcher (alle 2v2 fra alle 4-kombinasjoner)
# -----------------------------
def all_team_partitions_of_four(p4: Tuple[Player, Player, Player, Player]) -> List[Match]:
    # For 4 spillere finnes 3 unike 2v2-partisjoner
    p = list(p4)
    pairings = [
        ((p[0], p[1]), (p[2], p[3])),
        ((p[0], p[2]), (p[1], p[3])),
        ((p[0], p[3]), (p[1], p[2])),
    ]
    seen = set()
    matches: List[Match] = []
    for t1, t2 in pairings:
        m = normalize_match(t1, t2)
        key = (m.a, m.b)
        if key not in seen:
            seen.add(key)
            matches.append(m)
    return matches


def generate_candidate_matches(players: List[Player]) -> List[Match]:
    candidates: List[Match] = []
    for p4 in itertools.combinations(players, 4):
        candidates.extend(all_team_partitions_of_four(p4))
    uniq: Dict[Tuple[Team, Team], Match] = {}
    for m in candidates:
        uniq[(m.a, m.b)] = m
    return list(uniq.values())


# -----------------------------
# Hvor mange kamper M velger vi?
# - Må gi likt antall kamper per spiller: 4M % n == 0
# - Hvis "perfekt" er mulig: M = n(n-1)/4  (alle teammate-par nøyaktig 1 gang)
# - Ellers: minst mulig M som også prøver å dekke mange teammate-par: ceil(C(n,2)/2)
# -----------------------------
def perfect_possible(n: int) -> bool:
    return (n * (n - 1)) % 4 == 0


def choose_match_count(n: int) -> Tuple[int, bool]:
    """
    Returnerer (M, perfect_mode)
    perfect_mode True betyr at det matematisk finnes plan der alle teammate-par er nøyaktig 1 gang
    OG alle spiller like mange kamper. (For 4,5,8 innen 4–8)
    """
    if perfect_possible(n):
        M = (n * (n - 1)) // 4
        return M, True

    min_for_teammates = math.ceil((n * (n - 1) / 2) / 2)  # ceil(C(n,2)/2)

    base = n // math.gcd(n, 4)  # minste M som gir 4M % n == 0 er multiplum av base
    M = max(base, min_for_teammates)
    if M % base != 0:
        M += (base - (M % base))

    return M, False


# -----------------------------
# Scoring (lavere er bedre)
# -----------------------------
def score_schedule(schedule: List[Match], players: List[Player], perfect_mode: bool) -> float:
    n = len(players)
    teammate_counts: Dict[Tuple[Player, Player], int] = {}
    opp_counts: Dict[Tuple[Player, Player], int] = {}
    plays = {p: 0 for p in players}

    rest_streak = {p: 0 for p in players}
    rest_streak_pen = 0.0

    for m in schedule:
        in_match = set(m.players())

        for p in players:
            if p in in_match:
                plays[p] += 1
                rest_streak[p] = 0
            else:
                rest_streak[p] += 1
                if rest_streak[p] >= 2:
                    rest_streak_pen += (rest_streak[p] - 1)

        a1, a2 = m.a
        b1, b2 = m.b
        teammate_counts[pair_key(a1, a2)] = teammate_counts.get(pair_key(a1, a2), 0) + 1
        teammate_counts[pair_key(b1, b2)] = teammate_counts.get(pair_key(b1, b2), 0) + 1

        for x in m.a:
            for y in m.b:
                opp_counts[pair_key(x, y)] = opp_counts.get(pair_key(x, y), 0) + 1

    # Spillbalanse (bør bli ~0 hvis M valgt riktig)
    play_vals = list(plays.values())
    mean_play = sum(play_vals) / n
    var_play = sum((v - mean_play) ** 2 for v in play_vals) / n

    # Teammate coverage / missing + repeats
    all_pairs = [pair_key(players[i], players[j]) for i in range(n) for j in range(i + 1, n)]
    missing = 0
    deviation_from_one = 0
    repeats = 0

    for pk in all_pairs:
        c = teammate_counts.get(pk, 0)
        if c == 0:
            missing += 1
        if perfect_mode:
            deviation_from_one += abs(c - 1)
        repeats += max(0, c - 1)

    opp_repeats = sum(max(0, c - 1) for c in opp_counts.values())

    total = (
        W_PLAY_BALANCE * var_play
        + W_TEAMMATE_MISSING * (missing ** 2)
        + W_TEAMMATE_REPEAT * repeats
        + W_OPP_REPEAT * opp_repeats
        + W_CONSEC_REST * rest_streak_pen
    )

    if perfect_mode:
        total += W_PERFECT_DEVIATION * deviation_from_one

    return total


# -----------------------------
# Lokal søk (random restart + hillclimb)
# -----------------------------
def improve_schedule(
    init: List[Match],
    candidates: List[Match],
    players: List[Player],
    perfect_mode: bool,
    steps: int,
    rng: random.Random
) -> List[Match]:
    best = init[:]
    best_score = score_schedule(best, players, perfect_mode)

    for _ in range(steps):
        new = best[:]
        i = rng.randrange(len(new))
        new[i] = rng.choice(candidates)

        s = score_schedule(new, players, perfect_mode)
        if s < best_score:
            best, best_score = new, s

    return best


def build_schedule(players: List[Player]) -> Tuple[List[Match], int, bool]:
    n = len(players)
    M, perfect_mode = choose_match_count(n)

    rng = random.Random(RANDOM_SEED)

    candidates = generate_candidate_matches(players)
    if len(candidates) > CANDIDATE_SAMPLE:
        candidates = rng.sample(candidates, CANDIDATE_SAMPLE)

    best_schedule: List[Match] = []
    best_score = float("inf")

    for _ in range(SEARCH_RESTARTS):
        init = [rng.choice(candidates) for _ in range(M)]
        improved = improve_schedule(init, candidates, players, perfect_mode, LOCAL_STEPS, rng)
        s = score_schedule(improved, players, perfect_mode)
        if s < best_score:
            best_schedule, best_score = improved, s

    return best_schedule, M, perfect_mode


# -----------------------------
# Output: CSV + HTML (uten tid) + localStorage persistence
# -----------------------------
def export_csv(schedule: List[Match], players: List[Player], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Kamp", "Lag A", "Lag B", "Hviler"])
        for i, m in enumerate(schedule, 1):
            rest = ", ".join(m.resting(players)) if len(players) > 4 else "-"
            w.writerow([i, f"{m.a[0]} & {m.a[1]}", f"{m.b[0]} & {m.b[1]}", rest])


def export_html(
    schedule: List[Match],
    players: List[Player],
    perfect_mode: bool,
    plan_id: str,
    path: str
) -> None:
    rows = []
    for i, m in enumerate(schedule, 1):
        rest = ", ".join(m.resting(players)) if len(players) > 4 else "-"
        rows.append(f"""
        <tr>
          <td>{i}</td>
          <td><b>{m.a[0]}</b> &amp; <b>{m.a[1]}</b></td>
          <td><b>{m.b[0]}</b> &amp; <b>{m.b[1]}</b></td>
          <td>{rest}</td>
          <td>
            <label><input type="radio" name="w{i}" value="A"> A</label>
            <label style="margin-left:10px;"><input type="radio" name="w{i}" value="B"> B</label>
          </td>
        </tr>
        """)

    js_players = "[" + ",".join(repr(p) for p in players) + "]"
    js_matches = "[" + ",".join(
        f"{{a:{list(m.a)!r}, b:{list(m.b)!r}}}" for m in schedule
    ) + "]"

    mode_text = "PERFEKT" if perfect_mode else "BEST MULIG (perfekt er matematisk umulig for dette antallet spillere)"
    storage_key = f"padelplan_winners_v1_{plan_id}"

    # Mobil-kompakt CSS:
    css = """
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 12px; }
  .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; max-width: 980px; }

  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px 6px; border-bottom: 1px solid #eee; vertical-align: top; }
  th { text-align: left; font-size: 13px; }
  td { font-size: 14px; }

  h2 { font-size: 18px; margin: 0 0 6px 0; }
  h3 { font-size: 16px; margin: 12px 0 6px 0; }

  .muted { color: #666; font-size: 12px; line-height: 1.35; }
  .scoregrid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin-top: 10px; }
  .pill { display: inline-block; padding: 2px 7px; border: 1px solid #ddd; border-radius: 999px; font-size: 12px; }

  .tag { display:inline-block; padding: 3px 8px; border-radius: 999px; border:1px solid #ddd; font-size: 11px; }

  .btn {
    display:inline-block; padding: 7px 10px; border:1px solid #ddd; border-radius:10px;
    background:#fff; cursor:pointer; font-size:13px;
  }
  .btn:active { transform: translateY(1px); }

  /* Mobil: enda litt tettere */
  @media (max-width: 520px) {
    body { margin: 8px; }
    .card { padding: 10px; border-radius: 10px; }

    th, td { padding: 6px 4px; }
    th { font-size: 12px; }
    td { font-size: 13px; }

    h2 { font-size: 16px; }
    h3 { font-size: 15px; }

    .scoregrid { grid-template-columns: 1fr; } /* 1 kolonne på mobil */
    .btn { padding: 6px 9px; font-size: 12.5px; }
  }

  @media print {
    body { margin: 0; }
    .muted, .btn { display: none; }
    .card { border: none; }
  }
"""

    html = f"""<!doctype html>
<html lang="no">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Padelplan {plan_id}</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="card">
    <h2>Padelplan ({len(players)} spillere, {len(schedule)} kamper)</h2>

    <div class="muted" style="margin-bottom:10px;">
      <span class="tag">{mode_text}</span><br/>
      Valg av vinner (A/B) lagres på denne enheten. Plan-ID: <b>{plan_id}</b>
    </div>

    <div style="display:flex; gap:10px; margin: 8px 0 14px 0;">
      <button class="btn" onclick="window.resetPadel()">Nullstill poeng</button>
    </div>

    <table>
      <thead>
        <tr>
          <th>Kamp</th>
          <th>Lag A</th>
          <th>Lag B</th>
          <th>Hviler</th>
          <th>Vinner</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>

    <h3>Poeng (vinnerlag = 1 poeng per spiller)</h3>
    <div id="scores" class="scoregrid"></div>

    <div class="muted" style="margin-top:10px;">
      Tips: Hvis du åpner samme URL på en annen telefon, vil den ikke se valgene dine (lagres lokalt per enhet).
    </div>
  </div>

<script>
const players = {js_players};
const matches = {js_matches};
const STORAGE_KEY = {storage_key!r};

function saveWinners() {{
  const winners = {{}};
  for (let i = 0; i < matches.length; i++) {{
    const pick = document.querySelector(`input[name="w${{i+1}}"]:checked`);
    winners[i+1] = pick ? pick.value : null;
  }}
  localStorage.setItem(STORAGE_KEY, JSON.stringify(winners));
}}

function loadWinners() {{
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  let winners;
  try {{
    winners = JSON.parse(raw);
  }} catch {{
    return;
  }}
  for (let i = 0; i < matches.length; i++) {{
    const v = winners[i+1];
    if (v !== "A" && v !== "B") continue;
    const el = document.querySelector(`input[name="w${{i+1}}"][value="${{v}}"]`);
    if (el) el.checked = true;
  }}
}}

function computeScores() {{
  const scores = Object.fromEntries(players.map(p => [p, 0]));
  for (let i = 0; i < matches.length; i++) {{
    const pick = document.querySelector(`input[name="w${{i+1}}"]:checked`);
    if (!pick) continue;
    const m = matches[i];
    const team = pick.value === "A" ? m.a : m.b;
    scores[team[0]] += 1;
    scores[team[1]] += 1;
  }}
  return scores;
}}

function renderScores() {{
  const scores = computeScores();
  const entries = Object.entries(scores).sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0]));
  const el = document.getElementById("scores");
  el.innerHTML = entries.map(([p, s]) => `<div><span class="pill">${{s}}</span> ${{p}}</div>`).join("");
}}

document.addEventListener("change", (e) => {{
  if (e.target && e.target.name && e.target.name.startsWith("w")) {{
    saveWinners();
    renderScores();
  }}
}});

window.resetPadel = function() {{
  localStorage.removeItem(STORAGE_KEY);
  for (let i = 0; i < matches.length; i++) {{
    document.querySelectorAll(`input[name="w${{i+1}}"]`).forEach(x => x.checked = false);
  }}
  renderScores();
}};

loadWinners();
renderScores();
</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------
# Rapport (diagnostikk)
# -----------------------------
def report(schedule: List[Match], players: List[Player], perfect_mode: bool) -> None:
    n = len(players)
    plays = {p: 0 for p in players}
    teammate_counts: Dict[Tuple[Player, Player], int] = {}

    for m in schedule:
        for p in m.players():
            plays[p] += 1
        teammate_counts[pair_key(m.a[0], m.a[1])] = teammate_counts.get(pair_key(m.a[0], m.a[1]), 0) + 1
        teammate_counts[pair_key(m.b[0], m.b[1])] = teammate_counts.get(pair_key(m.b[0], m.b[1]), 0) + 1

    all_pairs = [pair_key(players[i], players[j]) for i in range(n) for j in range(i + 1, n)]
    missing = sum(1 for pk in all_pairs if teammate_counts.get(pk, 0) == 0)
    repeats = sum(max(0, teammate_counts.get(pk, 0) - 1) for pk in all_pairs)

    print("\nKamper per spiller:")
    for p, c in sorted(plays.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {p}: {c}")

    print(f"\nTeammate-dekning: {len(all_pairs) - missing}/{len(all_pairs)} par")
    print(f"Teammate repeats totalt: {repeats}")
    if perfect_mode:
        print("Perfekt-modus: forsøker nøyaktig 1 gang per teammate-par.")


# -----------------------------
# Main
# -----------------------------
def main():
    print("Skriv inn spillere (tom linje for å avslutte).")
    players: List[Player] = []
    while True:
        name = input(f"Spiller {len(players)+1}: ").strip()
        if not name:
            break
        players.append(name)

    if len(players) < 4 or len(players) > 8:
        raise SystemExit("Du må ha mellom 4 og 8 spillere.")

    schedule, M, perfect_mode = build_schedule(players)

    export_csv(schedule, players, OUTPUT_CSV)
    export_html(schedule, players, perfect_mode, PLAN_ID, OUTPUT_HTML)

    # Lag også index.html for GitHub Pages (alltid "siste plan")
    shutil.copyfile(OUTPUT_HTML, INDEX_HTML)

    print("\nFerdig!")
    print(f"- CSV:   {OUTPUT_CSV}")
    print(f"- HTML:  {OUTPUT_HTML}")
    print(f"- Index: {INDEX_HTML} (kopi av dagens plan)")
    print(f"- Kamper: {M}")
    print(f"- Plan-ID (lagring av poeng): {PLAN_ID}")

    if not perfect_mode:
        n = len(players)
        print(
            f"\nMerk: For {n} spillere er det matematisk umulig å få 'alle på lag med alle én gang' "
            "samtidig som alle spiller like mange kamper på 1 bane. "
            "Denne planen er derfor en best-mulig tilnærming med lik spilletid og maks teammate-variasjon."
        )

    report(schedule, players, perfect_mode)


if __name__ == "__main__":
    main()