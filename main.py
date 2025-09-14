import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

# ============================================================
#                           ФАЙЛЫ
# ============================================================

clients_file = "clients/clients.csv"
transfers_dir = "clients"
transactions_dir = "clients"

# ============================================================
#                КЛАСС ПРОДУКТОВ И РЕКОМЕНДАЦИЙ
# ============================================================

class ProductRecommender:

    # ----------------------------- ИНИЦИАЛИЗАЦИЯ -----------------------------
    def __init__(self):
        self.products = [
            'Карта для путешествий', 'Премиальная карта', 'Кредитная карта',
            'Обмен валют', 'Кредит наличными', 'Депозит мультивалютный',
            'Депозит сберегательный', 'Депозит накопительный', 'Инвестиции',
            'Золотые слитки'
        ]

    # ----------------------------- РАСЧЁТ БАЛЛОВ -----------------------------
    def calculate_product_scores(self, client_data, df_transfers, df_transactions):
        scores = {product: 0 for product in self.products}
        
        age = client_data.get('age', 30)
        status = client_data.get('status', 'Стандартный клиент')
        avg_balance = client_data.get('avg_monthly_balance_KZT', 0)
        city = client_data.get('city', '')
        
        transfer_counts = df_transfers['type'].value_counts()
        transfer_amounts = df_transfers.groupby('type')['amount'].sum()
        
        spending_by_category = pd.Series()
        total_spend = 0
        category_counts = pd.Series()
        top_categories = []
        
        if not df_transactions.empty and 'category' in df_transactions.columns:
            spending_by_category = df_transactions.groupby('category')['amount'].sum()
            total_spend = spending_by_category.sum()
            category_counts = df_transactions['category'].value_counts()
            
            if len(spending_by_category) > 0:
                top_categories = spending_by_category.nlargest(3).index.tolist()
        
        # 1. Карта для путешествий
        travel_categories = ['Путешествия', 'Отели', 'Такси', 'Авиабилеты', 'Транспорт']
        travel_spend = 0
        travel_count = 0
        
        if not spending_by_category.empty:
            travel_spend = sum(spending_by_category.get(cat, 0) for cat in travel_categories if cat in spending_by_category.index)
            travel_count = sum(category_counts.get(cat, 0) for cat in travel_categories if cat in category_counts.index)
        
        scores['Карта для путешествий'] = (
            min(travel_spend / 50000, 15) +
            min(travel_count / 5, 8) +
            (10 if any(cat in travel_categories for cat in top_categories) else 0) +
            (6 if 20 <= age <= 45 else 0)
        )
        
        # 2. Премиальная карта (увеличиваем вес)
        balance_score = 0
        if avg_balance >= 6000000:
            balance_score = 25
        elif avg_balance >= 3000000:
            balance_score = 20
        elif avg_balance >= 1000000:
            balance_score = 15
        elif avg_balance >= 500000:
            balance_score = 8
        
        premium_categories = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны', 'Спа и массаж', 'Люкс']
        premium_spend = 0
        if not spending_by_category.empty:
            premium_spend = sum(spending_by_category.get(cat, 0) for cat in premium_categories if cat in spending_by_category.index)
        
        # Активность переводов и снятий (ключевой фактор для премиальной карты)
        transfer_activity = (
            transfer_counts.get('atm_withdrawal', 0) * 3 +  # Снятия наличных
            transfer_counts.get('p2p_out', 0) * 2 +         # P2P переводы
            transfer_counts.get('card_out', 0) * 1.5 +      # Переводы на карты
            transfer_counts.get('utilities_out', 0) * 1.2   # Оплата услуг
        )
        
        scores['Премиальная карта'] = (
            balance_score +
            min(premium_spend / 80000, 15) +  # Увеличиваем вес премиальных трат
            min(transfer_activity / 20, 12) + # Увеличиваем вес активности
            (25 if status == 'Премиальный клиент' else 0) +
            (18 if status == 'Зарплатный клиент' else 0) +
            (10 if age >= 30 else 0) +  # Премиум карта для взрослой аудитории
            (8 if premium_spend > 100000 else 0)  # Бонус за премиальные траты
        )
        
        # 3. Кредитная карта (снижаем приоритет для клиентов с высоким балансом)
        installment_count = transfer_counts.get('installment_payment_out', 0)
        loan_count = transfer_counts.get('loan_payment_out', 0)
        cc_repayment_count = transfer_counts.get('cc_repayment_out', 0)
        
        online_categories = ['Едим дома', 'Смотрим дома', 'Играем дома', 'Онлайн услуги', 'Интернет']
        online_spend = 0
        if not spending_by_category.empty:
            online_spend = sum(spending_by_category.get(cat, 0) for cat in online_categories if cat in spending_by_category.index)
        
        balance_penalty = 0
        if avg_balance > 500000:
            balance_penalty = -5
        if avg_balance > 1000000:
            balance_penalty = -10
        
        scores['Кредитная карта'] = (
            min(total_spend / 120000, 10) +
            min(online_spend / 60000, 6) +
            (installment_count * 2) +
            (loan_count * 1.5) +
            (cc_repayment_count * 3) +
            (6 if 23 <= age <= 40 else 0) +
            balance_penalty
        )
        
        # 4. Обмен валют
        fx_activity = (transfer_counts.get('fx_buy', 0) + 
                      transfer_counts.get('fx_sell', 0))
        fx_amount = (transfer_amounts.get('fx_buy', 0) + 
                    transfer_amounts.get('fx_sell', 0))
        
        currency_diversity = 1
        if not df_transactions.empty and 'currency' in df_transactions.columns:
            currency_diversity = df_transactions['currency'].nunique()
        
        scores['Обмен валют'] = (
            fx_activity * 6 +
            min(fx_amount / 50000, 15) +
            ((currency_diversity - 1) * 8) +
            (8 if avg_balance >= 1000000 else 0)
        )
        
        # 5. Кредит наличными
        cash_need_signals = (
            transfer_counts.get('atm_withdrawal', 0) * 3 +
            transfer_counts.get('loan_payment_out', 0) * 2.5 +
            transfer_counts.get('p2p_out', 0) * 1.5 +
            (5 if avg_balance < 150000 else 0)  # Низкий баланс - сигнал
        )
        
        # Анализ кассовых разрывов
        if 'date' in df_transfers.columns and 'direction' in df_transfers.columns:
            try:
                df_transfers['date'] = pd.to_datetime(df_transfers['date'])
                monthly_outflow = df_transfers[df_transfers['direction'] == 'out'].groupby(
                    df_transfers['date'].dt.to_period('M'))['amount'].sum().mean()
                monthly_inflow = df_transfers[df_transfers['direction'] == 'in'].groupby(
                    df_transfers['date'].dt.to_period('M'))['amount'].sum().mean()
                
                if monthly_inflow > 0:
                    cash_gap_ratio = monthly_outflow / monthly_inflow
                    if cash_gap_ratio > 1.2:
                        cash_need_signals += 15
                    elif cash_gap_ratio > 1.0:
                        cash_need_signals += 10
            except:
                pass
        
        scores['Кредит наличными'] = min(cash_need_signals, 25)
        
        # 6. Депозит мультивалютный
        scores['Депозит мультивалютный'] = (
            scores['Обмен валют'] * 0.8 +
            (15 if avg_balance >= 1000000 else 0) +
            (10 if fx_activity > 2 else 0) +
            (12 if status == 'Премиальный клиент' else 0)
        )
        
        # 7. Депозит сберегательный
        stability_score = self.calculate_balance_stability(df_transfers)
            
        scores['Депозит сберегательный'] = (
            min(avg_balance / 400000, 18) +  # Увеличиваем требования к балансу
            stability_score * 2 +
            (20 if status == 'Премиальный клиент' else 0) +
            (15 if status == 'Зарплатный клиент' else 0) +
            (12 if age >= 40 else 0)
        )
        
        # 8. Депозит накопительный
        regular_savings_signals = (
            transfer_counts.get('salary_in', 0) * 4 +      # Регулярные зарплаты
            transfer_counts.get('deposit_topup_out', 0) * 5 +  # Пополнения вкладов
            transfer_counts.get('savings_out', 0) * 3 +    # Сберегательные переводы
            (12 if 50000 <= avg_balance <= 500000 else 0)  # Оптимальный баланс для накоплений
        )
        
        scores['Депозит накопительный'] = (
            min(regular_savings_signals, 18) +
            (18 if status == 'Зарплатный клиент' else 0) +
            (12 if status == 'Студент' else 0) +
            (10 if 25 <= age <= 40 else 0)
        )
        
        # 9. Инвестиции
        invest_activity = (
            transfer_counts.get('invest_out', 0) * 4 + 
            transfer_counts.get('invest_in', 0) * 3
        )
        invest_amount = (
            transfer_amounts.get('invest_out', 0) + 
            transfer_amounts.get('invest_in', 0)
        )
        
        scores['Инвестиции'] = (
            invest_activity * 6 +
            min(invest_amount / 60000, 20) +
            (18 if avg_balance >= 800000 else 0) +
            (15 if status in ['Премиальный клиент', 'Зарплатный клиент'] else 0) +
            (12 if 35 <= age <= 55 else 0)
        )
        
        # 10. Золотые слитки
        gold_activity = (
            transfer_counts.get('gold_buy_out', 0) * 5 + 
            transfer_counts.get('gold_sell_in', 0) * 4
        )
        
        scores['Золотые слитки'] = (
            gold_activity * 10 +
            (20 if avg_balance >= 2500000 else 0) +
            (18 if age >= 50 else 0) +
            (15 if status == 'Премиальный клиент' else 0) +
            (12 if stability_score > 7 else 0)
        )
        
        return scores, spending_by_category, category_counts, transfer_counts

    # ---------------------- РАСЧЁТ СТАБИЛЬНОСТИ ОСТАТКА ----------------------    
    def calculate_balance_stability(self, df_transfers):
        try:
            if 'date' not in df_transfers.columns or 'direction' not in df_transfers.columns:
                return 5
                
            df_temp = df_transfers.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            
            daily_net = []
            unique_dates = df_temp['date'].dt.date.unique()
            
            for date in unique_dates:
                day_data = df_temp[df_temp['date'].dt.date == date]
                inflow = day_data[day_data['direction'] == 'in']['amount'].sum()
                outflow = day_data[day_data['direction'] == 'out']['amount'].sum()
                daily_net.append(inflow - outflow)
            
            if len(daily_net) > 7:
                mean_net = np.mean(daily_net)
                if mean_net > 0:
                    stability = 1 - (np.std(daily_net) / mean_net)
                    return min(max(stability * 10, 0), 10)
            return 5
        except:
            return 5

    # ------------------- ГЕНЕРАЦИЯ PUSH-УВЕДОМЛЕНИЯ -------------------------
    def generate_push_notification(self, product, client_name, client_data, 
                                 spending_data, category_counts, transfer_counts):
        def format_amount(amount):
            return f"{amount:,.0f}".replace(',', ' ').replace('.', ',')
        
        status = client_data.get('status', 'Стандартный клиент')
        age = client_data.get('age', 30)
        avg_balance = client_data.get('avg_monthly_balance_KZT', 0)
        
        if product == 'Карта для путешествий':
            travel_categories = ['Путешествия', 'Отели', 'Такси', 'Авиабилеты']
            travel_spend = sum(spending_data.get(cat, 0) for cat in travel_categories if cat in spending_data.index)
            
            if travel_spend > 0:
                cashback = travel_spend * 0.04
                return f'{client_name}, в последнее время вы много путешествуете. С картой для путешествий вы могли бы вернуть до {format_amount(cashback)} ₸ кешбэка. Откройте карту в приложении!'
            else:
                return f'{client_name}, для ваших будущих поездок идеально подойдет наша карта для путешествий с повышенным кешбэком 4% на отели и такси. Оформите сейчас!'
        
        elif product == 'Премиальная карта':
            premium_categories = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны']
            premium_spend = sum(spending_data.get(cat, 0) for cat in premium_categories if cat in spending_data.index)
            
            if status == 'Премиальный клиент':
                return f'{client_name}, как премиальный клиент, получите до 4% кешбэка на все покупки и бесплатные снятия по всему миру до 3 млн ₸/мес. Активируйте карту!'
            elif avg_balance >= 1000000:
                if premium_spend > 50000:
                    return f'{client_name}, с вашим балансом и тратами на премиальные категории вы могли бы получать до 4% кешбэка. Премиальная карта даст бесплатные снятия и переводы!'
                else:
                    return f'{client_name}, у вас высокий остаток на счету. Премиальная карта даст до 4% кешбэка на все покупки и бесплатные снятия. Подключите сейчас!'
            else:
                return f'{client_name}, премиальная карта откроет доступ к повышенному кешбэку и привилегиям. Оформите карту и пользуйтесь всеми преимуществами!'
        
        elif product == 'Кредитная карта':
            if status == 'Студент':
                return f'{client_name}, специально для студентов — кредитная карта с кешбэком до 10% на развлечения и обучение. Оформите онлайн!'
            elif not spending_data.empty:
                top_categories = spending_data.nlargest(2).index.tolist()
                if len(top_categories) >= 1:
                    cats_str = ", ".join(top_categories[:2])
                    return f'{client_name}, ваши основные траты — {cats_str}. Кредитная карта даёт до 10% кешбэка в этих категориях. Оформите карту!'
            return f'{client_name}, кредитная карта с кешбэком до 10% в ваших любимых категориях. Выбирайте категории каждый месяц!'
        
        elif product == 'Инвестиции':
            invest_count = transfer_counts.get('invest_out', 0) + transfer_counts.get('invest_in', 0)
            if invest_count > 0:
                return f'{client_name}, вы уже инвестируете. Перенесите инвестиции к нам — торгуйте без комиссий в первый год. Откройте счёт!'
            elif status == 'Премиальный клиент':
                return f'{client_name}, как премиальный клиент, получите особые условия для инвестиций — нулевые комиссии на все операции. Начните сейчас!'
            else:
                return f'{client_name}, начните инвестировать с нами от 10 000 ₸. Диверсифицируйте свои сбережения!'
        
        elif product == 'Депозит сберегательный':
            if status == 'Премиальный клиент':
                return f'{client_name}, для премиальных клиентов специальная ставка 17% годовых на сберегательный вклад. Разместите средства!'
            elif avg_balance >= 1000000:
                return f'{client_name}, разместите свободные средства на сберегательном вкладе под 16,5% годовых. Надёжно и выгодно!'
            else:
                return f'{client_name}, начните копить на сберегательном вкладе от 50 000 ₸ под 15% годовых. Безопасно и прибыльно!'
        
        elif product == 'Депозит накопительный':
            salary_count = transfer_counts.get('salary_in', 0)
            if salary_count > 0 and status == 'Зарплатный клиент':
                return f'{client_name}, откладывайте часть зарплаты планомерно под 15,5% годовых. Идеально для накоплений!'
            elif status == 'Студент':
                return f'{client_name}, начните копить на будущее! Накопительный вклад для студентов под 14% годовых. Откройте сейчас!'
            else:
                return f'{client_name}, создайте привычку регулярных накоплений под 15% годовых. Пополняйте когда удобно!'
        
        elif product == 'Депозит мультивалютный':
            fx_activity = transfer_counts.get('fx_buy', 0) + transfer_counts.get('fx_sell', 0)
            if fx_activity > 0:
                return f'{client_name}, храните средства в разных валютах под 14,5% годовых. Защитите сбережения от колебаний курсов!'
            else:
                return f'{client_name}, диверсифицируйте сбережения в разных валютах под 14% годовых. Надёжно и перспективно!'
        
        elif product == 'Обмен валют':
            fx_activity = transfer_counts.get('fx_buy', 0) + transfer_counts.get('fx_sell', 0)
            if fx_activity > 0:
                return f'{client_name}, меняйте валюту выгодно без комиссий 24/7. Установите целевой курс для авто-покупки!'
            else:
                return f'{client_name}, получайте лучший курс обмена валют без комиссий. Установите уведомления о выгодном курсе!'
        
        elif product == 'Кредит наличными':
            atm_count = transfer_counts.get('atm_withdrawal', 0)
            if atm_count > 5:
                return f'{client_name}, если часто снимаете наличные — кредит до 2 000 000 ₸ с бесплатным снятием. Узнать условия!'
            else:
                return f'{client_name}, кредит наличными до 2 000 000 ₸ на любые цели с гибчным погашением. Оформите онлайн!'
        
        else:
            if status == 'Премиальный клиент':
                return f'{client_name}, как премиальный клиент, получите особые условия на покупку золотых слитков. Диверсифицируйте сбережения!'
            elif avg_balance >= 2000000:
                return f'{client_name}, диверсифицируйте сбережения золотыми слитками 999,9 пробы. Надёжно и перспективно!'
            else:
                return f'{client_name}, начните инвестировать в золото от 10 000 ₸. Защитите сбережения от инфляции!'

    # ------------------- ВЫБОР НАИЛУЧШЕГО ПРОДУКТА --------------------------
    def recommend_product(self, client_data, df_transfers, df_transactions):
        try:
            scores, spending_data, category_counts, transfer_counts = self.calculate_product_scores(
                client_data, df_transfers, df_transactions
            )
            
            print(f"Баллы для {client_data.get('name')} ({client_data.get('status')}, баланс: {client_data.get('avg_monthly_balance_KZT', 0):,.0f} ₸):")
            for product, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {product}: {score:.1f}")
            
            best_product = max(scores.items(), key=lambda x: x[1])[0]
            
            push_text = self.generate_push_notification(
                best_product, 
                client_data.get('name', 'Клиент'), 
                client_data,
                spending_data, 
                category_counts, 
                transfer_counts
            )
            
            return best_product, push_text, scores
            
        except Exception as e:
            print(f"Ошибка в recommend_product: {str(e)}")
            return 'Депозит накопительный', 'Оцените преимущества нашего накопительного вклада!', {}

# ============================================================
#                   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def load_client_data(clients_file):
    clients_df = pd.read_csv(clients_file)
    return clients_df

def process_client(client_id, clients_df, transfers_dir, transactions_dir):
    try:
        client_row = clients_df[clients_df['client_code'] == client_id]
        if client_row.empty:
            print(f"Клиент {client_id} не найден в clients.csv")
            return client_id, None, None, None
        
        client_data = client_row.iloc[0].to_dict()
        
        transfers_path = f"{transfers_dir}/client_{client_id}_transfers_3m.csv"
        transactions_path = f"{transactions_dir}/client_{client_id}_transactions_3m.csv"
        
        if not Path(transfers_path).exists():
            print(f"Файл transfers не найден для клиента {client_id}")
            return client_id, None, None, None
        
        df_transfers = pd.read_csv(transfers_path, sep=',', thousands=' ', decimal='.')
        
        if df_transfers.empty:
            print(f"Пустой файл transfers для клиента {client_id}")
            return client_id, None, None, None
        
        df_transactions = pd.DataFrame()
        if Path(transactions_path).exists():
            try:
                df_transactions = pd.read_csv(transactions_path, sep=';', thousands=' ', decimal=',')
            except:
                try:
                    df_transactions = pd.read_csv(transactions_path, sep=',', thousands=' ', decimal='.')
                except Exception as e:
                    print(f"Не удалось прочитать transactions для клиента {client_id}")
        
        recommender = ProductRecommender()
        product, push_text, scores = recommender.recommend_product(
            client_data, df_transfers, df_transactions
        )
        
        return client_id, product, push_text, scores
        
    except Exception as e:
        print(f"Ошибка обработки клиента {client_id}: {str(e)}")
        return client_id, 'Депозит накопительный', 'Оцените преимущества нашего накопительного вклада!', {}

# ============================================================
#                          MAIN-ФУНКЦИЯ
# ============================================================
def main():

    
    clients_df = load_client_data(clients_file)
    print(f"Загружено {len(clients_df)} клиентов")
    
    results = []
    
    for client_id in clients_df['client_code']:
        print(f"\n=== Обрабатываем клиента {client_id} ===")
        result = process_client(client_id, clients_df, transfers_dir, transactions_dir)
        
        if result[1] is not None:
            client_id, product, push_text, scores = result
            results.append({
                'client_code': client_id,
                'product': product,
                'push_notification': push_text,
                'scores': scores
            })
            print(f"Рекомендовано: {product}")
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("Нет данных для обработки")
        return pd.DataFrame()
    
    output_df = df_results[['client_code', 'product', 'push_notification']].copy()
    output_df.to_csv('product_recommendations.csv', index=False, quoting=1)
    
    print("\nОбработка завершена! Результаты сохранены в product_recommendations.csv")
    
    scores_df = pd.DataFrame([{**{'client_code': row['client_code']}, **row['scores']} 
                            for _, row in df_results.iterrows()])
    scores_df.to_csv('detailed_scores.csv', index=False)
    print("Детальные баллы сохранены в detailed_scores.csv")
    
    return output_df

# ============================================================
#                        ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    recommendations = main()
    
    if not recommendations.empty:
        print("\n=== Финальные рекомендации ===")
        print(recommendations.to_string(index=False))
        
        # Статистика по продуктам
        print("\n=== Распределение рекомендаций ===")
        product_stats = recommendations['product'].value_counts()
        for product, count in product_stats.items():
            print(f"{product}: {count} клиентов")
    else:
        print("Нет рекомендаций для отображения")
