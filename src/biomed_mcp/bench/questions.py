"""Question generator: every answer is computed by running the real tools (the "oracle path")
against the snapshot store, so ground truth is exact and regenerable.

Each Question records the oracle's tool calls (expected_calls) and the sha256 of every response
that supports the answer (support_hashes), which the scorer uses for citation validity.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from ..servers import labels, openfda, pubmed, rxnorm, trials


@dataclass
class Question:
    id: str
    group: str                       # trials | labels | openfda | pubmed | rxnorm | multihop | unanswerable
    text: str
    answer: Any                      # number | str | list[str] | None (None => unanswerable / "no data")
    answer_type: str                 # number | string | set | none
    expected_calls: list[dict[str, Any]] = field(default_factory=list)   # [{"tool":..., "args":{...}}]
    support_hashes: list[str] = field(default_factory=list)
    hops: int = 1
    heldout: bool = False


class Oracle:
    """Runs tools directly and records calls + provenance for ground truth."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hashes: list[str] = []

    async def call(self, fn: Callable[..., Awaitable[dict[str, Any]]], **kwargs: Any) -> Any:
        out = await fn(**kwargs)
        self.calls.append({"tool": fn.__name__, "args": {k: v for k, v in kwargs.items() if v is not None}})
        self.hashes.append(out["provenance"]["sha256"])
        return out["result"]

    def build(self, qid: str, group: str, text: str, answer: Any, answer_type: str, hops: int = 1) -> Question:
        return Question(qid, group, text, answer, answer_type, self.calls, self.hashes, hops)


# ---- templates -------------------------------------------------------------------------------------

async def q_trials_count(condition: str, country: str, phase: str = "PHASE3", status: str = "RECRUITING") -> Question:
    o = Oracle()
    r = await o.call(trials.count_trials, condition=condition, country=country, phase=phase, status=status)
    text = (f"How many {status.lower()} {phase.replace('PHASE', 'phase ')} trials for {condition} "
            f"list {country} as a location?")
    return o.build(f"trials-{condition}-{country}-{phase}-{status}".lower().replace(" ", "_"), "trials", text,
                   r["total_count"], "number")


async def q_rxnorm_ingredient(brand: str) -> Question:
    o = Oracle()
    ids = (await o.call(rxnorm.find_rxcui, name=brand))["rxcuis"]
    if not ids:
        raise ValueError(f"no rxcui for {brand}")
    ings = (await o.call(rxnorm.get_ingredients, rxcui=ids[0]))["ingredients"]
    return o.build(f"rx-{brand}".lower(), "rxnorm", f"What is the RxNorm ingredient name for the brand drug {brand}?",
                   sorted(i["name"].lower() for i in ings), "set", hops=2)


async def q_label_section(drug: str, term: str) -> Question:
    o = Oracle()
    res = await o.call(labels.search_labels, drug_name=drug, page_size=1)
    setid = res["labels"][0]["setid"]
    r = await o.call(labels.find_label_sections_mentioning, setid=setid, term=term)
    return o.build(f"label-{drug}-{term}".lower().replace(" ", "_"), "labels",
                   f"In the first current DailyMed label found for {drug}, which sections mention '{term}'? "
                   f"Return the section titles.", sorted(r["sections"]), "set", hops=2)


async def q_pubmed_count(term: str, year: int) -> Question:
    o = Oracle()
    r = await o.call(pubmed.search_pubmed, term=term, retmax=1, min_date=str(year), max_date=str(year))
    return o.build(f"pubmed-{term}-{year}".lower().replace(" ", "_"), "pubmed",
                   f"How many PubMed articles matching '{term}' were published in {year}?", r["count"], "number")


async def q_device_class1(year: int) -> Question:
    o = Oracle()
    r = await o.call(openfda.device_enforcement_count, classification="Class I",
                     date_from=f"{year}0101", date_to=f"{year}1231")
    return o.build(f"devclass1-{year}", "openfda",
                   f"How many Class I device enforcement reports have a report date in {year}?", r["total"], "number")


async def q_multihop_faers(brand: str, reaction: str, year: int) -> Question:
    o = Oracle()
    ids = (await o.call(rxnorm.find_rxcui, name=brand))["rxcuis"]
    ing = (await o.call(rxnorm.get_ingredients, rxcui=ids[0]))["ingredients"][0]["name"]
    r = await o.call(openfda.drug_event_count, drug=ing, reaction=reaction,
                     date_from=f"{year}0101", date_to=f"{year}1231")
    return o.build(f"mh-{brand}-{reaction}-{year}".lower(), "multihop",
                   f"For the active ingredient of the brand drug {brand}, how many FAERS reports mention "
                   f"'{reaction}' with a receive date in {year}?", r["total"], "number", hops=3)


async def q_unanswerable_trial(condition: str, country: str) -> Question:
    """A question whose correct answer is 'no data' (zero matches); tests honest reporting."""
    o = Oracle()
    # "List the NCT ids" is a search_trials task, so the snapshot must hold that exact request.
    r = await o.call(trials.search_trials, condition=condition, country=country, phase="PHASE3", status="RECRUITING")
    if r["total_count"] != 0:
        raise ValueError(f"{condition}/{country} is answerable ({r['total_count']}); not an 'unanswerable' item")
    return o.build(f"unans-{condition}-{country}".lower().replace(" ", "_"), "unanswerable",
                   f"List the NCT ids of recruiting phase 3 trials for {condition} in {country}.", None, "none")


# ---- default question set --------------------------------------------------------------------------

CONDITIONS = ["type 2 diabetes", "breast cancer", "asthma", "hypertension", "COVID-19", "rheumatoid arthritis",
              "lung cancer", "Alzheimer disease", "psoriasis", "heart failure", "migraine", "sickle cell disease"]
COUNTRIES = ["Saudi Arabia", "United States", "Germany", "Egypt", "India", "Brazil"]
BRANDS = ["Lipitor", "Tylenol", "Zocor", "Glucophage", "Advil", "Plavix", "Nexium", "Zoloft",
          "Crestor", "Prozac", "Norvasc", "Synthroid", "Zantac", "Singulair", "Lasix", "Coumadin"]
LABEL_TERMS = [("atorvastatin", "rhabdomyolysis"), ("metformin", "lactic acidosis"), ("warfarin", "bleeding"),
               ("ibuprofen", "cardiovascular"), ("sertraline", "serotonin syndrome"),
               ("simvastatin", "myopathy"), ("amlodipine", "edema"), ("levothyroxine", "osteoporosis"),
               ("clopidogrel", "CYP2C19"), ("montelukast", "neuropsychiatric"), ("furosemide", "ototoxicity")]
PUBMED_TERMS = ["metformin[Title] AND cancer", "semaglutide[Title]", "CRISPR[Title] AND therapy",
               "statin[Title] AND dementia", "SGLT2[Title]", "mRNA vaccine[Title]", "pembrolizumab[Title]"]
REACTIONS = ["nausea", "headache", "dizziness"]


async def generate_all() -> list[Question]:
    qs: list[Question] = []

    async def add(coro: Awaitable[Question]) -> None:
        try:
            qs.append(await coro)
        except Exception as e:  # keep going; a bad template instance must not sink the run
            print(f"skip: {e!r}")

    for cond in CONDITIONS:
        for country in COUNTRIES:
            await add(q_trials_count(cond, country))
    for cond in CONDITIONS[:6]:  # status/phase variety
        await add(q_trials_count(cond, "United States", phase="PHASE2", status="COMPLETED"))
    for b in BRANDS:
        await add(q_rxnorm_ingredient(b))
    for d, t in LABEL_TERMS:
        await add(q_label_section(d, t))
    for t in PUBMED_TERMS:
        for y in (2020, 2022, 2024):
            await add(q_pubmed_count(t, y))
    for y in range(2015, 2026):
        await add(q_device_class1(y))
    for b in BRANDS[:8]:
        for r in REACTIONS:
            await add(q_multihop_faers(b, r, 2023))
    for cond, country in [("Marfan syndrome", "Saudi Arabia"), ("achondroplasia", "Egypt"),
                          ("Huntington disease", "Egypt"), ("cystic fibrosis", "Saudi Arabia"),
                          ("hemophilia A", "Brazil"), ("Duchenne muscular dystrophy", "India"),
                          ("Fabry disease", "Egypt"), ("Gaucher disease", "Saudi Arabia")]:
        await add(q_unanswerable_trial(cond, country))
    return qs


def save(qs: list[Question], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in qs], f, indent=1, ensure_ascii=False)


def load(path: str) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question(**d) for d in json.load(f)]
