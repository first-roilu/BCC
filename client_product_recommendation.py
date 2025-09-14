"""

#
По мимо библиотек нужно установить локальную ИИ mistral через ollama для генерации пуш-уведомлений
#

Запуск скрипта осуществляется через консоль комманда для запуска (пример): python client_product_recommendation.py --clients_dir ./clients --output recommendations.csv --fx_rates fx_rates.json

Параметр --clients_dir отвечает за путь к файлам информации о клиентах.

Параметр --output отвечает за путь сохранения, названия, формата файла.

Параметр --fx_rates отвечает за загрузку файла с курсами валют примерно такого содержания

            {
              "USD": 480.25,
              "EUR": 512.10,
              "RUB": 5.25
            }


"""

import requests
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import math
import numpy as np
import pandas as pd


# ----------------------------- ПАРАМЕТРЫ -----------------------------
PARAMS = {
    "months_window": 3,

    "premium": {
        "base_rate": 0.02,
        "tier_1_min_balance": 1_000_000,
        "tier_1_max_balance": 6_000_000,
        "tier_1_rate": 0.03,
        "tier_2_min_balance": 6_000_000,
        "tier_2_rate": 0.04,
        "special_categories_rate": 0.04,
        "monthly_cashback_limit": 100_000,
    },

    "travel": {
        "rate": 0.04,
        "categories": ["Путешествия", "Такси", "Отели"],
    },

    "credit_card": {
        "fav_rate": 0.10,
        "fav_categories_count": 3,
        "online_categories": ["Играем дома", "Смотрим дома", "Кино"],
    },

    "fx": {"assumed_spread_saving": 0.005},

    "deposits": {
        "multicurrency_rate": 0.1450,
        "saver_rate": 0.1650,
        "accumulation_rate": 0.1550,
    },

    "investments": {"expected_first_year_return": 0.06},

    "cash_loan": {"benefit_rate_on_gap": 0.01, "gap_threshold_ratio": 0.5},

    "gold": {"expected_appreciation": 0.01},

    # сигнальные веса для compute_signals (могут быть подделаны)
    "signals_weight": {
        "age": 1.0,
        "status": 1.5,
        "top_categories": 2.0,
        "freq_categories": 1.5,
        "balance": 2.0,
        "currency_ops": 1.5,
        "transfers_profile": 1.0,
    },

    # финальные веса для score
    "score_weights": {
        "benefit": 0.65,  # нормализованный ожидаемый benefit
        "signals": 0.25,  # нормализованный сигнал
        "urgency": 0.07,  # бонус за срочность
        "current_penalty": 0.03,  # штраф если уже есть такой продукт
    },

    # urgency bonuses
    "urgency_bonus": {"immediate": 0.08, "future": 0.02, "never": -0.2},

    # минимальная пороговая выгода чтобы продукт считался релевантным
    "min_expected_benefit_threshold": 50.0,
}


# ----------------------------- Утилиты -----------------------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                return pd.DataFrame()


def convert_to_kzt(amount: float, currency: str, fx_rates: Optional[Dict[str, float]]) -> Optional[float]:
    if pd.isna(amount):
        return None
    if not currency or currency == "KZT":
        try:
            return float(amount)
        except Exception:
            return None
    if not fx_rates:
        return None
    rate = fx_rates.get(currency)
    if rate is None:
        return None
    try:
        return float(amount) * float(rate)
    except Exception:
        return None

def generate_push_with_mistral(profile, product, reason, urgency):
    prompt = (
        "Составь короткое push-уведомление для клиента банка. Не больше 180-220 символов. Один восклицательный максимум (и только по делу). "
        "Даты формата дд.мм.гггг или же дд.месяц_прописью.гггг - где уместно. Числа: дробная часть — запятая; разряды — пробелы. "
        "Валюта: единый формат по каналу (в интерфейсе — символ; в SMS — «тг»); разряд и знак валюты отделяем пробелом например не 2490₸ а 2 490 ₸. "
        "Ссылки/кнопки: глагол действия — «Открыть», «Настроить», «Посмотреть». Никаких «крикливых» обещаний/давления; не злоупотреблять триггерами дефицита. "
        "Нужно по итогу что бы сообщение было примерно таким (будет зависить от рекомендуемого продукта) Карта путешествий «{name}, в {month} у вас много "
        "поездок/такси. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?» Премиальная карта  «{name}, у вас стабильно крупный остаток "
        "и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас.» Кредитная карта «{name}, ваши топ-категории "
        "— {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту.» FX/мультивалютный продукт: «{name}, "
        "вы часто платите в {fx_curr}. В приложении выгодный обмен и авто-покупка по целевому курсу. Настроить обмен.» Вклады (сберегательный/накопительный) "
        "«{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.» Инвестиции «{name}, "
        "попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт.» Кредит наличными (только при явной потребности) «{name}, если "
        "нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит.». Вот тебе несколько примеров отталкиваясь "
        "от них уже генерируешь нормальное Пуш-уведомление смотри на примеры и не добавляй лишней личной информации и без глупостей/шуточек что будет дальше этих "
        "слов уже данные по которым ты генерируешь Пуш-сообщение\n"
        f"Профиль: {profile}\n"
        f"Рекомендуемый продукт: {product}\n"
        f"Причина: {reason}\n"
        f"Срочность: {urgency}\n"
        f"Тон: дружелюбный, CTA в конце."
    )

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False},
        timeout=90,
    )
    try:
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.JSONDecodeError:
        text = []
        for line in resp.text.splitlines():
            if line.strip().startswith("{"):
                try:
                    part = json.loads(line)
                    if "response" in part:
                        text.append(part["response"])
                except json.JSONDecodeError:
                    pass
        return "".join(text).strip()


# ----------------------------- Фичи клиента -----------------------------

def compute_client_features(profile: Dict, tx_df: pd.DataFrame, tr_df: pd.DataFrame, fx_rates: Optional[Dict[str, float]]):
    months = PARAMS["months_window"]
    tx = tx_df.copy()
    tr = tr_df.copy()

    for df in (tx, tr):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    tx["amount_kzt"] = tx.apply(lambda r: convert_to_kzt(r.get("amount"), r.get("currency"), fx_rates), axis=1)
    tx_spend = tx[tx["amount_kzt"].notna()].copy()

    if not tx_spend.empty and "category" in tx_spend.columns:
        spend_by_cat = tx_spend.groupby("category")["amount_kzt"].sum().rename("sum_kzt").reset_index()
        freq_by_cat = tx_spend.groupby("category")["amount_kzt"].count().rename("count").reset_index()
    else:
        spend_by_cat = pd.DataFrame(columns=["category", "sum_kzt"])
        freq_by_cat = pd.DataFrame(columns=["category", "count"])

    total_spend_3m = float(tx_spend["amount_kzt"].sum()) if not tx_spend.empty else 0.0
    avg_monthly_spend = total_spend_3m / months

    top_categories_by_sum = spend_by_cat.sort_values("sum_kzt", ascending=False).head(5).to_dict(orient="records")
    top_categories_by_freq = freq_by_cat.sort_values("count", ascending=False).head(5).to_dict(orient="records")

    tr["amount_kzt"] = tr.apply(lambda r: convert_to_kzt(r.get("amount"), r.get("currency"), fx_rates), axis=1)

    fx_ops = tr[tr.get("currency") != "KZT"] if not tr.empty else pd.DataFrame()
    total_fx_volume_kzt = float(fx_ops["amount_kzt"].abs().sum()) if not fx_ops.empty and "amount_kzt" in fx_ops.columns else 0.0

    salary_in_df = tr[(tr.get("type") == "salary_in") & (tr.get("amount_kzt").notna())] if not tr.empty else pd.DataFrame()
    salary_count = int(salary_in_df.shape[0])
    salary_sum_kzt = float(salary_in_df["amount_kzt"].sum()) if not salary_in_df.empty else 0.0
    salary_avg_monthly = salary_sum_kzt / months if salary_count > 0 else 0.0

    cashback_count = int(tr[tr.get("type") == "cashback_in"].shape[0]) if not tr.empty else 0
    refund_count = int(tr[tr.get("type") == "refund_in"].shape[0]) if not tr.empty else 0

    important_types = [
        "salary_in", "p2p_out", "card_out", "atm_withdrawal", "loan_payment_out",
        "invest_in", "invest_out", "deposit_topup_out", "gold_buy_out", "gold_sell_in",
        "installment_payment_out"
    ]
    transfer_counts = {t: int(tr[tr.get("type") == t].shape[0]) if not tr.empty else 0 for t in important_types}

    combined = pd.concat([
        tx_spend[["date", "amount_kzt"]].assign(kind="tx") if not tx_spend.empty else pd.DataFrame(columns=["date", "amount_kzt", "kind"]),
        tr[["date", "amount_kzt"]] if not tr.empty else pd.DataFrame(columns=["date", "amount_kzt"]),
    ], ignore_index=True)

    combined = combined.dropna(subset=["date"]) if not combined.empty else combined
    if not combined.empty and "date" in combined.columns:
        combined["month"] = combined["date"].dt.to_period("M")
        monthly_out = combined.groupby("month")["amount_kzt"].sum()
        max_monthly_outflow = float(monthly_out.max()) if not monthly_out.empty else 0.0
        avg_monthly_outflow = float(monthly_out.mean()) if not monthly_out.empty else 0.0
    else:
        max_monthly_outflow = 0.0
        avg_monthly_outflow = 0.0

    travel_cats = PARAMS["travel"]["categories"]
    travel_sum_3m = 0.0
    if not spend_by_cat.empty and "category" in spend_by_cat.columns:
        travel_sum_3m = float(spend_by_cat.loc[spend_by_cat["category"].isin(travel_cats), "sum_kzt"].sum())
    travel_share = (travel_sum_3m / total_spend_3m) if total_spend_3m > 0 else 0.0

    features = {
        "total_spend_3m_kzt": total_spend_3m,
        "avg_monthly_spend_kzt": avg_monthly_spend,
        "top_categories_by_sum": top_categories_by_sum,
        "top_categories_by_freq": top_categories_by_freq,
        "spend_by_cat_df": spend_by_cat,
        "freq_by_cat_df": freq_by_cat,
        "total_fx_volume_kzt": total_fx_volume_kzt,
        "transfer_counts": transfer_counts,
        "max_monthly_outflow": max_monthly_outflow,
        "avg_monthly_outflow": avg_monthly_outflow,
        "salary_count": salary_count,
        "salary_avg_monthly": salary_avg_monthly,
        "cashback_count": cashback_count,
        "refund_count": refund_count,
        "travel_share": travel_share,
    }

    return features


# ----------------------------- Вычисление ожидаемой выгоды -----------------------------

def expected_benefit_travel(features: Dict):
    cfg = PARAMS["travel"]
    spend_df = features.get("spend_by_cat_df")
    if spend_df.empty:
        return 0.0
    mask = spend_df["category"].isin(cfg["categories"]) if "category" in spend_df.columns else []
    travel_sum = float(spend_df.loc[mask, "sum_kzt"].sum()) if not spend_df.empty else 0.0
    monthly_travel = travel_sum / PARAMS["months_window"]
    return monthly_travel * cfg["rate"]


def expected_benefit_premium(profile: Dict, features: Dict):
    cfg = PARAMS["premium"]
    balance = float(profile.get("avg_monthly_balance_KZT") or 0)
    if cfg["tier_1_min_balance"] <= balance < cfg["tier_1_max_balance"]:
        base_rate = cfg["tier_1_rate"]
    elif balance >= cfg["tier_2_min_balance"]:
        base_rate = cfg["tier_2_rate"]
    else:
        base_rate = cfg["base_rate"]

    monthly_spend = features.get("avg_monthly_spend_kzt", 0.0)
    special_cats = ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"]
    special_sum = 0.0
    spend_df = features.get("spend_by_cat_df")
    if not spend_df.empty:
        special_sum = float(spend_df.loc[spend_df["category"].isin(special_cats), "sum_kzt"].sum()) / PARAMS["months_window"]
    rest = max(monthly_spend - special_sum, 0.0)
    cashback = special_sum * cfg["special_categories_rate"] + rest * base_rate
    cashback = min(cashback, cfg["monthly_cashback_limit"])

    if features.get("salary_count", 0) >= 1:
        cashback *= 1.05
    return cashback


def expected_benefit_credit_card(profile: Dict, features: Dict):
    cfg = PARAMS["credit_card"]
    spend_df = features.get("spend_by_cat_df")
    if spend_df.empty or spend_df.shape[0] == 0:
        return 0.0
    topN = spend_df.sort_values("sum_kzt", ascending=False).head(cfg["fav_categories_count"]) if not spend_df.empty else pd.DataFrame()
    topN_monthly = float(topN["sum_kzt"].sum()) / PARAMS["months_window"] if not topN.empty else 0.0
    online_sum = 0.0
    if not spend_df.empty:
        online_sum = float(spend_df.loc[spend_df["category"].isin(cfg["online_categories"]), "sum_kzt"].sum()) / PARAMS["months_window"]
    cashback = (topN_monthly + online_sum) * cfg["fav_rate"]
    return cashback


def expected_benefit_fx(profile: Dict, features: Dict):
    cfg = PARAMS["fx"]
    volume = features.get("total_fx_volume_kzt", 0.0)
    return volume * cfg["assumed_spread_saving"] / PARAMS["months_window"]


def expected_benefit_deposit(profile: Dict, features: Dict):
    bal = float(profile.get("avg_monthly_balance_KZT") or 0)
    monthly_multicurr = bal * PARAMS["deposits"]["multicurrency_rate"] / 12.0
    monthly_saver = bal * PARAMS["deposits"]["saver_rate"] / 12.0
    monthly_acc = bal * PARAMS["deposits"]["accumulation_rate"] / 12.0
    return max(monthly_multicurr, monthly_saver, monthly_acc)


def expected_benefit_investments(profile: Dict, features: Dict):
    bal = float(profile.get("avg_monthly_balance_KZT") or 0)
    investable = bal * 0.10
    monthly_return = investable * PARAMS["investments"]["expected_first_year_return"] / 12.0
    return monthly_return


def expected_benefit_cash_loan(profile: Dict, features: Dict):
    bal = float(profile.get("avg_monthly_balance_KZT") or 0)
    max_out = features.get("max_monthly_outflow", 0.0)
    if max_out > bal * (1 + PARAMS["cash_loan"]["gap_threshold_ratio"]):
        gap = max_out - bal
        return gap * PARAMS["cash_loan"]["benefit_rate_on_gap"] / PARAMS["months_window"]
    return 0.0


def expected_benefit_gold(profile: Dict, features: Dict):
    bal = float(profile.get("avg_monthly_balance_KZT") or 0)
    invest = bal * 0.05
    monthly = invest * PARAMS["gold"]["expected_appreciation"] / 12.0
    return monthly


# ----------------------------- Сигналы / Паттерны -----------------------------

def compute_eligibility_and_timing(profile: Dict, features: Dict, signals: Dict) -> Dict[str, Dict]:
    res = {}
    bal = float(profile.get("avg_monthly_balance_KZT") or 0)
    status = (profile.get("status") or "").lower()
    age = int(profile.get("age") or 0)

    top_sum = [c.get("category") for c in features.get("top_categories_by_sum", [])][:3]
    fx_vol = features.get("total_fx_volume_kzt", 0.0)
    transfer_counts = features.get("transfer_counts", {})
    avg_monthly_spend = features.get("avg_monthly_spend_kzt", 0.0)
    max_out = features.get("max_monthly_outflow", 0.0)

    def setp(prod, eligible, urgency, reason):
        res[prod] = {"eligible": eligible, "urgency": urgency, "reason": reason}

    travel_tags = PARAMS["travel"]["categories"]
    travel_signal = any(cat in travel_tags for cat in top_sum) or features.get("travel_share",0) > 0.12
    if travel_signal and bal >= 0:
        urgency = "immediate" if travel_signal and features.get("travel_share",0) > 0.2 else "future"
        setp("Карта для путешествий", True, urgency, "Высокие траты в travel-related категориях")
    else:
        setp("Карта для путешествий", False, "never", "Нет признаков путешествий")

    if status == "студент":
        setp("Премиальная карта", False, "never", "Не подходит студентам")
    elif bal >= PARAMS["premium"]["tier_1_min_balance"] or status == "премиальный клиент":
        setp("Премиальная карта", True, "immediate", "Высокий остаток или премиальный статус")
    elif bal >= 300_000:
        setp("Премиальная карта", True, "future", "Средний остаток — можно предложить позднее")
    else:
        setp("Премиальная карта", False, "never", "Недостаточный остаток")

    cc_need = False
    if transfer_counts.get("installment_payment_out", 0) > 0 or transfer_counts.get("card_out", 0) > 5:
        cc_need = True
    if any(cat in ["Развлечения", "Кино", "Играем дома", "Смотрим дома"] for cat in top_sum):
        cc_need = True
    if avg_monthly_spend > 50_000:
        cc_need = True
    if cc_need:
        setp("Кредитная карта", True, "immediate" if transfer_counts.get("installment_payment_out",0)>0 else "future", "Использует рассрочку/много расходов")
    else:
        setp("Кредитная карта", True, "future", "Может понадобиться при росте расходов")

    if fx_vol > 0 or features.get("total_fx_volume_kzt",0)>100_000:
        setp("Обмен валют", True, "immediate", "Активные FX операции")
    else:
        setp("Обмен валют", False, "never", "Нет операций в других валютах")

    if max_out > bal * (1 + PARAMS["cash_loan"]["gap_threshold_ratio"]):
        setp("Кредит наличными", True, "immediate", "Максимальный отток значительно превышает баланс")
    else:
        setp("Кредит наличными", False, "never", "Нет признаков острой нехватки ликвидности")

    if bal >= 100_000 and avg_monthly_spend < bal * 0.1:
        urgency = "immediate" if bal >= 500_000 else "future"
        setp("Депозит Сберегательный", True, urgency, "Высокий остаток и низкие текущие траты")
    else:
        setp("Депозит Сберегательный", False, "never", "Недостаточный остаток или большие регулярные траты")

    if fx_vol > 0 or features.get("total_fx_volume_kzt",0)>0:
        setp("Депозит Мультивалютный", True, "immediate", "Пользуется валютой")
    elif bal >= 200_000:
        setp("Депозит Мультивалютный", True, "future", "Можно предложить при желании диверсифицировать")
    else:
        setp("Депозит Мультивалютный", False, "never", "Низкий остаток и отсутствие валютных операций")

    if bal >= 50_000 and avg_monthly_spend < bal * 0.25:
        setp("Депозит Накопительный", True, "future", "Подходит для планомерных накоплений")
    else:
        setp("Депозит Накопительный", False, "never", "Не подходит для накопления сейчас")

    if features.get("transfer_counts",{}).get("invest_in",0)>0 or features.get("transfer_counts",{}).get("invest_out",0)>0 or (18 <= age <= 60 and bal >= 20_000):
        setp("Инвестиции", True, "immediate", "Есть интерес/история инвестиций или молодой инвест-ориентированный возраст")
    else:
        setp("Инвестиции", False, "future", "Можно предложить как опцию в будущем")

    if status == "студент" or age < 25 or bal < 200_000:
        setp("Золотые слитки", False, "never", "Неоптимально для студентов/молодых/низкий остаток")
    elif bal >= 500_000:
        setp("Золотые слитки", True, "future", "Хорошо для диверсификации при достаточном остатке")
    else:
        setp("Золотые слитки", False, "never", "Недостаточный остаток")

    return res


def compute_signals(profile: Dict, features: Dict) -> Dict[str, float]:
    signals = {}
    status = (profile.get("status") or "").lower()
    age = int(profile.get("age") or 0)
    balance = float(profile.get("avg_monthly_balance_KZT") or 0)

    top_sum = [c.get("category") for c in features.get("top_categories_by_sum", [])][:3]
    top_freq = [c.get("category") for c in features.get("top_categories_by_freq", [])][:3]
    total_fx = features.get("total_fx_volume_kzt", 0.0)

    salary_avg = features.get("salary_avg_monthly", 0.0)
    cashback_count = features.get("cashback_count", 0)

    travel_score = 0.0
    travel_cats = PARAMS["travel"]["categories"]
    if any(cat in travel_cats for cat in top_sum):
        travel_score += PARAMS["signals_weight"]["top_categories"]
    if any(cat in travel_cats for cat in top_freq):
        travel_score += PARAMS["signals_weight"]["freq_categories"]
    if status in ["премиальный клиент", "зарплатный клиент"]:
        travel_score += 0.5
    if salary_avg > 100_000 and features.get("travel_share",0) > 0.08:
        travel_score += 0.8
    signals["Карта для путешествий"] = travel_score

    premium_score = 0.0
    if balance >= 500_000:
        premium_score += PARAMS["signals_weight"]["balance"]
    if status == "премиальный клиент":
        premium_score += PARAMS["signals_weight"]["status"]
    if features.get("total_spend_3m_kzt", 0) > 200_000:
        premium_score += 1.0
    premium_score += min(cashback_count * 0.2, 1.0)
    signals["Премиальная карта"] = premium_score

    credit_score = 0.0
    transfer_counts = features.get("transfer_counts", {})
    if transfer_counts.get("installment_payment_out", 0) > 0 or transfer_counts.get("card_out", 0) > 5:
        credit_score += 1.5
    if any(cat in ["Развлечения", "Кино", "Играем дома", "Смотрим дома"] for cat in top_sum):
        credit_score += 1.0
    if features.get("avg_monthly_spend_kzt",0) > 50_000:
        credit_score += 0.5
    signals["Кредитная карта"] = credit_score

    fx_score = 0.0
    if total_fx > 0:
        fx_score += PARAMS["signals_weight"]["currency_ops"]
    signals["Обмен валют"] = fx_score

    loan_score = 0.0
    if features.get("max_monthly_outflow",0) > balance * 0.8:
        loan_score += 1.0
    signals["Кредит наличными"] = loan_score

    deposit_score = 0.0
    if balance > 50_000:
        deposit_score += 1.0
    if status in ["премиальный клиент"]:
        deposit_score += 0.5
    signals["Депозит Мультивалютный"] = deposit_score
    signals["Депозит Сберегательный"] = deposit_score
    signals["Депозит Накопительный"] = deposit_score

    invest_score = 0.0
    if transfer_counts.get("invest_in", 0) > 0 or transfer_counts.get("invest_out", 0) > 0:
        invest_score += 1.5
    if 18 <= age <= 45:
        invest_score += 0.5
    signals["Инвестиции"] = invest_score

    gold_score = 0.0
    if transfer_counts.get("gold_buy_out", 0) > 0 or transfer_counts.get("gold_sell_in", 0) > 0:
        gold_score += 1.0
    signals["Золотые слитки"] = gold_score

    return signals


# ----------------------------- Главная логика / финальная метрика -----------------------------

def safe_norm(vals):
    if not vals:
        return []
    mn = min(vals)
    mx = max(vals)
    if math.isclose(mn, mx):
        return [0.0 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]


def log_scale(x):
    return math.log1p(max(0.0, x))


def evaluate_client(profile: Dict, tx_df: pd.DataFrame, tr_df: pd.DataFrame, fx_rates: Optional[Dict[str, float]]):
    features = compute_client_features(profile, tx_df, tr_df, fx_rates)
    signals = compute_signals(profile, features)
    eligibility = compute_eligibility_and_timing(profile, features, signals)

    benefit_map = {
        "Карта для путешествий": expected_benefit_travel(features),
        "Премиальная карта": expected_benefit_premium(profile, features),
        "Кредитная карта": expected_benefit_credit_card(profile, features),
        "Обмен валют": expected_benefit_fx(profile, features),
        "Кредит наличными": expected_benefit_cash_loan(profile, features),
        "Депозит Мультивалютный": expected_benefit_deposit(profile, features),
        "Депозит Сберегательный": expected_benefit_deposit(profile, features),
        "Депозит Накопительный": expected_benefit_deposit(profile, features),
        "Инвестиции": expected_benefit_investments(profile, features),
        "Золотые слитки": expected_benefit_gold(profile, features),
    }

    candidates = []
    for product, ben in benefit_map.items():
        elig = eligibility.get(product, {"eligible": True, "urgency": "future", "reason": ""})
        candidates.append({
            "product": product,
            "expected_benefit_kzt": float(ben),
            "signal_raw": float(signals.get(product, 0.0)),
            "eligible": bool(elig.get("eligible", True)),
            "urgency": elig.get("urgency", "future"),
            "reason": elig.get("reason", ""),
        })

    eligible_candidates = [c for c in candidates if c["eligible"]]
    if not eligible_candidates:
        eligible_candidates = candidates
        for c in eligible_candidates:
            c["reason"] = c.get("reason") or "Нет явно подходящих продуктов — выбор по скору"

    bens_raw = [c["expected_benefit_kzt"] for c in eligible_candidates]
    bens_log = [log_scale(x) for x in bens_raw]
    sigs = [c["signal_raw"] for c in eligible_candidates]

    norm_b = safe_norm(bens_log)
    norm_s = safe_norm(sigs)

    w = PARAMS["score_weights"]
    urgency_bonus_map = PARAMS["urgency_bonus"]
    current = (profile.get("product") or profile.get("current_product") or "").strip()

    results = []
    for i, c in enumerate(eligible_candidates):
        nb = norm_b[i] if i < len(norm_b) else 0.0
        ns = norm_s[i] if i < len(norm_s) else 0.0
        urgency_b = urgency_bonus_map.get(c.get("urgency","future"), 0.0)
        cur_pen = 0.0
        if current and c["product"] == current:
            cur_pen = -PARAMS["score_weights"]["current_penalty"]

        final = w["benefit"] * nb + w["signals"] * ns + w["urgency"] * urgency_b + cur_pen

        if c["expected_benefit_kzt"] < PARAMS["min_expected_benefit_threshold"]:
            final -= 0.05

        results.append({
            "product": c["product"],
            "expected_benefit_kzt": c["expected_benefit_kzt"],
            "score": round(final, 6),
            "urgency": c["urgency"],
            "reason": c["reason"],
            "eligible": c["eligible"],
            "norm_b": round(nb, 4),
            "norm_s": round(ns, 4),
        })

    results_sorted = sorted(results, key=lambda r: (r["score"], r["expected_benefit_kzt"]), reverse=True)
    best = results_sorted[0] if results_sorted else {"product": None, "expected_benefit_kzt": 0.0, "score": 0.0, "urgency": "never", "reason": "нет кандидатов", "eligible": False}
    return best, results_sorted, features, signals


# ----------------------------- main -----------------------------

def main(clients_dir: Path, output: Path, fx_rates_path: Optional[Path]):
    fx_rates = None
    if fx_rates_path and fx_rates_path.exists():
        try:
            fx_rates = json.loads(fx_rates_path.read_text(encoding="utf-8"))
        except Exception:
            print("Не удалось загрузить fx_rates.json — операции в других валютах будут игнорироваться")
            fx_rates = None

    clients_file = clients_dir / "clients.csv"
    clients_df = read_csv_auto(clients_file)
    if clients_df.empty:
        print(f"Не найден или пуст файл профилей: {clients_file}")
        return

    out_rows = []
    detailed_rows = []

    for _, row in clients_df.iterrows():
        client_code = str(row.get("client_code") or row.get("client_id") or "").strip()
        if not client_code:
            continue
        profile = {c: row.get(c) for c in clients_df.columns}

        tx_file = clients_dir / f"client_{client_code}_transactions_3m.csv"
        tr_file = clients_dir / f"client_{client_code}_transfers_3m.csv"
        tx_df = read_csv_auto(tx_file)
        tr_df = read_csv_auto(tr_file)

        best, all_products, features, signals = evaluate_client(profile, tx_df, tr_df, fx_rates)

        push_placeholder = generate_push_with_mistral(
            profile=profile,
            product=best["product"],
            reason=best.get("reason", ""),
            urgency=best.get("urgency", "future"),
        )

        out_rows.append({
            "client_code": client_code,
            "product": best["product"],
            "push_notification": push_placeholder
        })

        detailed_rows.append({
            "client_code": client_code,
            "all_products": json.dumps(all_products, ensure_ascii=False),
            "features": json.dumps({k: (v if not isinstance(v, pd.DataFrame) else v.to_dict()) for k, v in features.items()}, ensure_ascii=False),
            "signals": json.dumps(signals, ensure_ascii=False),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output, index=False, encoding='utf-8')
    print(f"Готово. Рекомендации сохранены в {output}")

    details_path = output.parent / (output.stem + "_details.csv")
    pd.DataFrame(detailed_rows).to_csv(details_path, index=False, encoding='utf-8')
    print(f"Подробные данные сохранены в {details_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender: expected benefit per product per client (tuned)")
    parser.add_argument("--clients_dir", type=str, required=True, help="Папка с clients.csv и файлами транзакций/переводов")
    parser.add_argument("--output", type=str, required=False, default="recommendations.csv", help="CSV для результата")
    parser.add_argument("--fx_rates", type=str, required=False, help="JSON файл с курсами валют к KZT")
    args = parser.parse_args()

    main(Path(args.clients_dir), Path(args.output), Path(args.fx_rates) if args.fx_rates else None)
