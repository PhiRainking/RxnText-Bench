# -*- coding: utf-8 -*-
"""Reaction-condition extraction benchmark —— demo pipeline(纯标准库)。

用法:
    python pipeline.py                 # 规则 baseline
    python pipeline.py --n 10          # 截断
    (设 DEEPSEEK_API_KEY 后自动加跑 LLM 对比; LLM_MODEL 可覆盖默认 deepseek-chat)

输出:
    out/results.json   逐条预测 + 逐字段分数 + 汇总
    out/fig1.svg       柱状图(论文 Figure 的 demo 版)
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

from seed_reactions import SEED_REACTIONS, SYNONYMS, REV_MAP

# 噪声渲染 + 同义词归一化:别名→规范金标
CANON_ALIAS = {}
for _d in SYNONYMS.values():
    for _alias, _canon in _d.items():
        CANON_ALIAS.setdefault(_canon, []).append(_alias)


def chem_norm(s):
    return re.sub(r"[\s()\[\]]", "", (s or "").lower())


REV_SUB = {chem_norm(k): v for k, v in REV_MAP.items()}


def canonize(x):
    """别名→规范名;LLM 常输出 'PhMe (toluene)' 带括号注释,剥括号取规范名。"""
    x = str(x).strip()
    k = chem_norm(x)
    if k in REV_SUB:
        return REV_SUB[k]
    if "(" in x or "（" in x:
        parts = [p for p in re.split(r"[()（）]", x) if p.strip()]
        hits = [REV_SUB[chem_norm(p)] for p in parts if chem_norm(p) in REV_SUB]
        if hits:
            return hits[0]
    return x

FIELDS = ["reactants", "reagents", "catalyst", "solvent",
          "temperature_c", "time_min", "yield_pct", "product"]

# ---------- 归一化 ----------
def sub_norm(s):            # 去空格(子串匹配用)
    return re.sub(r"\s+", "", (s or "").lower())

def toks(s):                # 词元集合(IoU 用)
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def tok_union(items):
    u = set()
    for it in items or []:
        u |= toks(it)
    return u

# ---------- 规则抽取器 ----------
SOLVENTS = ["water", "methanol", "ethanol", "dichloromethane", "chloroform",
            "tetrahydrofuran", "toluene", "diethyl ether", "ethyl acetate",
            "dimethylformamide", "acetonitrile", "hexane", "dioxane", "xylene",
            "acetone", "tert-butanol", "2-propanol", "1,4-dioxane"]
CAT_KEYS = ["pd(pph3)4", "pd/c", "pd(oac)2", "pd2(dba)3", "pdcl2(pph3)2",
            "dmap", "cuso4"]
REAG_KEYS = ["potassium carbonate", "sodium hydroxide", "sodium borohydride",
             "lithium aluminum hydride", "sulfuric acid", "triethylamine",
             "trifluoroacetic acid", "imidazole", "mcpba", "pcc", "piperidine",
             "n-butyllithium", "sodium ascorbate", "copper(i) iodide", "xantphos",
             "potassium tert-butoxide", "edci", "hobt", "dmp"]


def extract_regex(text):
    t = text.lower()
    t2 = sub_norm(text)
    out = {"reactants": [], "reagents": [], "catalyst": None,
           "solvent": None, "temperature_c": None, "time_min": None,
           "yield_pct": None, "product": None}

    # solvent:出现位置最靠前的已知溶剂
    hits = [(t2.find(sub_norm(s)), s) for s in SOLVENTS if sub_norm(s) in t2]
    if hits:
        out["solvent"] = min(hits)[1]

    # temperature
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[°o]\s*c", t)
    if m:
        out["temperature_c"] = float(m.group(1))

    # time
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|h|minutes?|min)\b", t)
    if m:
        v = float(m.group(1))
        out["time_min"] = v * 60 if m.group(2).startswith("h") else v

    # yield
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        out["yield_pct"] = float(m.group(1))

    # catalyst
    for k in CAT_KEYS:
        if k in t2:
            out["catalyst"] = k
            break

    # reagents(关键词命中,剔除已被认作催化剂的)
    out["reagents"] = [k for k in REAG_KEYS if k in t2 and k != out["catalyst"]]

    # reactants:主语短语
    cap = None
    m = re.search(r"(?:solution|mixture|suspension) of (.+?)(?: in | was | were )", t)
    if m:
        cap = m.group(1)
    if cap:
        cands = [c.strip() for c in re.split(r"\s+and\s+|\s*,\s*", cap) if c.strip()]
        banned = {sub_norm(x) for x in out["reagents"] + [out["catalyst"], out["solvent"]]}
        out["reactants"] = [c for c in cands if sub_norm(c) not in banned]

    # product:头部 "X (92%):" 或 "affording/gave/... X"
    m = re.match(r"^\s*([^:()]+?)(?:\s*\(\s*\d+(?:\.\d+)?\s*%\))?\s*:", t)
    if m and len(m.group(1)) < 120:
        out["product"] = m.group(1).strip()
    else:
        m = re.search(r"(?:affording|gave|yielded|isolated|obtained)\s+([^,()]+?)(?:\s+in\s+\d|$)", t)
        if m:
            out["product"] = m.group(1).strip().strip("the ")
    return out

# ---------- LLM 抽取器 ----------
LLM_ALIASES = {
    "reactants": ["reactants", "reactant", "starting_materials"],
    "product": ["product", "products"],
    "reagents": ["reagents", "reagent", "additives"],
    "catalyst": ["catalyst"],
    "solvent": ["solvent", "solvents"],
    "temperature_c": ["temperature_c", "temperature", "temp_c", "temp"],
    "time_min": ["time_min", "time_minutes", "time"],
    "yield_pct": ["yield_pct", "yield_percent", "yield"],
}

SYSTEM_PROMPT = (
    "You extract reaction information from chemistry sentences. "
    'Return ONLY JSON with exactly these keys: "reactants" (list of strings), '
    '"product" (string), "reagents" (list of non-catalytic additives; empty list if none), '
    '"catalyst" (string or null), "solvent" (string or null), '
    '"temperature_c" (number, or null if room temperature is stated), '
    '"time_min" (number in minutes), "yield_pct" (number or null). No commentary.'
)


def _first_num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def parse_llm_json(raw):
    raw = raw.strip()
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.S)
        raw = m.group(0) if m else raw
    d = json.loads(raw)
    out = {}
    for field, aliases in LLM_ALIASES.items():
        v = None
        for a in aliases:
            if a in d:
                v = d[a]
                break
        out[field] = v
    return out


def coerce(pred, field):
    """把 LLM 原始值转成与 gold 同构的规范形态。"""
    if field in ("reactants", "reagents"):
        if not pred:
            return []
        if isinstance(pred, str):
            return [canonize(x) for x in re.split(r",|;|\s+and\s+", pred) if x.strip()]
        return [canonize(x) for x in pred if str(x).strip()]
    if field == "product":
        return str(pred).strip() if pred else None
    if field in ("catalyst", "solvent"):
        return canonize(pred) if pred else None
    # 数值域
    if pred is None or pred == "":
        return None
    s = str(pred)
    if field == "temperature_c":
        return _first_num(s)
    if field == "time_min":
        v = _first_num(s)
        if v is not None and ("h" in s.lower() and "min" not in s.lower()):
            v *= 60
        return v
    if field == "yield_pct":
        return _first_num(s)
    return None


def _chat_once(text, model, api_key, base_url, json_mode):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def llm_extract(text, model, api_key, base_url):
    """OpenAI 兼容 chat completion;json 模式失败自动降级重试。返回 (parsed|None, usage|None)。"""
    json_mode = True
    last_err = None
    for attempt in range(3):
        try:
            resp = _chat_once(text, model, api_key, base_url, json_mode)
            content = resp["choices"][0]["message"]["content"]
            try:
                return parse_llm_json(content), resp.get("usage", {})
            except Exception:
                last_err = "json parse failure"
                json_mode = False
                continue
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:150]!r}"
            if e.code == 400 and json_mode:
                json_mode = False
                continue
            if e.code in (429, 500, 502, 503):
                time.sleep(4 * (attempt + 1))
                continue
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(3)
    print(f"    [llm fail] {last_err}")
    return None, None


# ---------- 语料渲染(合成 gold 文本) ----------
def fmt_temp(t, rng):
    if t is None:
        return rng.choice(["at room temperature", "at ambient temperature"])
    ti = int(t) if float(t).is_integer() else t
    return rng.choice([f"at {ti} °C", f"at {ti}°C", f"heated at {ti} °C"])


def fmt_time(mins, rng):
    if mins % 60 == 0:
        h = mins // 60
        return rng.choice([f"for {h} h", f"for {h} hours"])
    if mins < 60:
        return f"for {mins} min"
    return rng.choice([f"for {mins} min", f"for {mins / 60:.1f} h"])


def fmt_yield(y, rng):
    if y is None:
        return "", ""
    yi = int(y) if float(y).is_integer() else y
    return f" in {yi}% yield", f"({yi}%)"


def stir_verb(tmp):
    return "" if tmp.startswith("heated") else "stirred "


def apply_noise(s, rxn, rng):
    """现实噪声:50% 溶剂/催化剂/试剂换缩写或长名;≥10h 反应 35% 改 'overnight'。"""
    if rxn.get("solvent") and rng.random() < 0.5:
        aliases = CANON_ALIAS.get(rxn["solvent"])
        if aliases:
            s = s.replace(rxn["solvent"], rng.choice(aliases))
    if rxn.get("catalyst") and rng.random() < 0.5:
        aliases = CANON_ALIAS.get(rxn["catalyst"])
        if aliases:
            s = s.replace(rxn["catalyst"], rng.choice(aliases))
    for rg in rxn.get("reagents") or []:
        if rng.random() < 0.5:
            aliases = CANON_ALIAS.get(rg)
            if aliases:
                s = s.replace(rg, rng.choice(aliases))
    if rxn.get("time_min", 0) >= 600 and rng.random() < 0.35:
        m = re.search(r"(for|over) \d+(?:\.\d+)? (?:h|hours|min)\b", s)
        if m:
            s = s[:m.start()] + "overnight" + s[m.end():]
    return s


def render(rxn, rng):
    rp = " and ".join(rxn["reactants"])
    rg = ", ".join(rxn["reagents"]) if rxn["reagents"] else ""
    cat = rxn["catalyst"]
    sol = rxn["solvent"]
    tmp = fmt_temp(rxn["temperature_c"], rng)
    dur = fmt_time(rxn["time_min"], rng)
    yph_a, yph_b = fmt_yield(rxn["yield_pct"], rng)
    prod = rxn["product"]

    cp = f" in the presence of {cat} as catalyst" if cat else ""
    if sol:
        sp = f" in {sol}"
    else:
        sp = ""
    k = rng.randint(0, 2)

    if k == 0:
        if rg:
            rgs = f"{rg}{f' and {cat}' if cat else ''}"
            s = (f"To a stirred solution of {rp}{sp} was added {rgs}, and the "
                 f"mixture was {stir_verb(tmp)}{tmp} {dur}, affording {prod}{yph_a}.")
        else:
            s = (f"A solution of {rp}{sp}{cp} was {stir_verb(tmp)}{tmp} {dur}, "
                 f"affording {prod}{yph_a}.")
    elif k == 1:
        parts = [p for p in [rp, rg, cat] if p]
        head = f"{prod}{(' ' + yph_b) if yph_b else ''}:"
        s = (f"{head} A mixture of {', '.join(parts)}{sp} was {stir_verb(tmp)}{tmp} "
             f"{dur}; after workup, {prod} was isolated.")
    else:
        if rg:
            s = (f"A solution of {rp}{sp} containing {rg}{cp} was {stir_verb(tmp)}{tmp} "
                 f"{dur}; after aqueous workup, {prod} was obtained{yph_a}.")
        else:
            s = (f"A solution of {rp}{sp}{cp} was {stir_verb(tmp)}{tmp} {dur}; after "
                 f"aqueous workup, {prod} was obtained{yph_a}.")
    return apply_noise(s, rxn, rng)


# ---------- 评测 ----------
def names_match(g_items, p_items):
    g = tok_union(g_items if isinstance(g_items, list) else [g_items])
    p = tok_union(p_items if isinstance(p_items, list) else [p_items])
    if not g and not p:
        return True
    if not g or not p:
        return False
    return len(g & p) / len(g | p) >= 0.75


def num_match(g, p, kind):
    if g is None and p is None:
        return True
    if g is None or p is None:
        return False
    if kind == "temperature_c":
        return abs(g - p) <= 2.0
    if kind == "time_min":
        return abs(g - p) <= max(5.0, 0.05 * g)
    return abs(g - p) <= 2.0  # yield_pct


def score_instance(gold, pred):
    res = {}
    for f in FIELDS:
        g, p = gold[f], pred[f]
        if f in ("temperature_c", "time_min", "yield_pct"):
            res[f] = num_match(g, p, f)
        else:
            res[f] = names_match(g, p)
    return res


# ---------- 汇总与输出 ----------
def acc_table(scores, extractor_names):
    rows = {}
    for name in extractor_names:
        rows[name] = {f: [] for f in FIELDS}
    for sc in scores:
        for name in extractor_names:
            for f in FIELDS:
                rows[name][f].append(sc[name][f])
    out = {}
    for name in extractor_names:
        out[name] = {}
        for f in FIELDS:
            vals = rows[name][f]
            out[name][f] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def write_svg(path, table, extractor_names, n_inst=0):
    colors = {"regex": "#4C72B0", "llm": "#DD8452"}
    rows = [f for f in FIELDS] + ["OVERALL"]
    n = len(rows)
    bw, bh = 640, 34
    hh = 46
    H = 90 + n * hh
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{bw}" height="{H}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="20" y="30" font-family="Arial" font-size="16" font-weight="bold">'
             f'Field-level extraction accuracy on RxnText-Bench ({n_inst} instances)</text>']
    overall = {}
    for name in extractor_names:
        v = [table[name][f] for f in FIELDS]
        overall[name] = sum(v) / len(v) if v else None
    x0, xmax = 200, 600
    for i, f in enumerate(rows):
        y = 60 + i * hh
        parts.append(f'<text x="{x0 - 10}" y="{y + 14}" font-family="Arial" font-size="13" '
                     f'text-anchor="end">{f}</text>')
        x = x0
        for name in extractor_names:
            v = table[name][f] if f != "OVERALL" else overall[name]
            v = v if v is not None else 0.0
            w = max(2.0, (xmax - x0) * v)
            parts.append(f'<rect x="{x}" y="{y}" width="{w:.1f}" height="20" '
                         f'fill="{colors[name]}" rx="3"/>')
            parts.append(f'<text x="{x + w + 4}" y="{y + 14}" font-family="Arial" '
                         f'font-size="12">{v:.2f}</text>')
            x += (xmax - x0) / max(1, len(extractor_names)) + 20
    ly = H - 50
    lx = 200
    for name in extractor_names:
        parts.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx + 22}" y="{ly + 13}" font-family="Arial" font-size="13">{name}</text>')
        lx += 90
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="instance 数上限")
    ap.add_argument("--variants", type=int, default=1, help="每条反应渲染变体数")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    workers = int(os.environ.get("LLM_WORKERS", "6"))
    use_llm = bool(api_key) and not args.no_llm

    # 合成语料
    instances = []
    for i, rxn in enumerate(SEED_REACTIONS):
        for v in range(args.variants):
            rng = random.Random(2026 + 1000 * i + v)
            instances.append({"id": rxn["id"], "gold": rxn, "text": render(rxn, rng)})
    if args.n:
        instances = instances[: args.n]

    # gold 规范化(空串→None)
    def gold_of(rxn):
        g = dict(rxn)
        for f in ("catalyst", "solvent", "product"):
            g[f] = g[f] or None
        return g

    print(f"== demo pipeline | instances={len(instances)} | llm={'on (' + model + ')' if use_llm else 'off'}")
    print()

    llm_results = {}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    if use_llm:
        from concurrent.futures import ThreadPoolExecutor

        def job(item):
            try:
                raw, usage = llm_extract(item["text"], model, api_key, base_url)
                return raw, usage, None
            except Exception as e:
                return None, None, str(e)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(job, it): i for i, it in enumerate(instances)}
            for fut in futs:
                i = futs[fut]
                raw, usage, err = fut.result()
                if err:
                    print(f"  [llm error {instances[i]['id']}] {err}")
                llm_results[i] = raw
                if usage:
                    usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                    usage_total["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        n_ok = sum(1 for r in llm_results.values() if r is not None)
        print(f"  llm calls ok: {n_ok}/{len(instances)} | "
              f"tokens in={usage_total['prompt_tokens']} out={usage_total['completion_tokens']}")

    scores, rows = [], []
    for i, it in enumerate(instances):
        g = gold_of(it["gold"])
        pr = extract_regex(it["text"])
        row = {"id": it["id"], "text": it["text"],
               "pred_regex": pr, "sc_regex": score_instance(g, pr)}
        if use_llm:
            raw = llm_results.get(i)
            if raw is not None:
                pl = {f: coerce(raw.get(f), f) for f in FIELDS}
                row["pred_llm"] = pl
                row["sc_llm"] = score_instance(g, pl)
            else:
                row["llm_error"] = "failed"
        rows.append(row)
        sc = {"regex": row["sc_regex"]}
        if use_llm and "sc_llm" in row:
            sc["llm"] = row["sc_llm"]
        scores.append(sc)

    names = ["regex"] + (["llm"] if use_llm else [])
    table = acc_table(scores, names)

    # 打印
    hdr = "field".ljust(14) + "".join(x.rjust(12) for x in names)
    print(hdr)
    print("-" * len(hdr))
    for f in FIELDS:
        print(f.ljust(14) + "".join(f"{table[x][f]:.3f}".rjust(12) for x in names))
    ov = [sum(table[x][f] for f in FIELDS) / len(FIELDS) for x in names]
    print("-" * len(hdr))
    print("OVERALL".ljust(14) + "".join(f"{v:.3f}".rjust(12) for v in ov))

    # 落盘
    os.makedirs("out", exist_ok=True)
    meta = {"instances": len(instances), "extractors": names,
            "llm_model": model if use_llm else None,
            "llm_base": base_url if use_llm else None,
            "tokens": usage_total if use_llm else None}
    with open("out/results.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "table": table, "rows": rows},
                  fh, ensure_ascii=False, indent=1)
    write_svg("out/fig1.svg", table, names, len(instances))
    print()
    print(f"wrote out/results.json, out/fig1.svg")
    if not use_llm:
        print("LLM mode skipped: set DEEPSEEK_API_KEY (and optionally LLM_MODEL) "
              "to compare LLM extraction.")


if __name__ == "__main__":
    sys.exit(main())
