import pandas as pd
import re
import numpy as np
from collections import Counter, defaultdict

#needed questions

def extract_number_from_question(question):
    match = re.search(r"\btop (\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_number_from_alarm_question(question: str) -> int:
    match = re.search(r"(\d+)\s*(alarmsignaal|alarmsignalen)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))  
    return None

def extract_number_from_signal_question(question: str) -> int:
    match = re.search(r"(\d+)\s*(signaal|signalen)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))  
    return None

def extract_percentage_from_question(question):
    match = re.search(r"(\d+)\s*(procent|%)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_months_from_question(question: str):
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match:
        number1 = int(match.group(1))
        number2 = int(match.group(2))
        return number1, number2
    match = re.search(r'(\d+)\s*en\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match:
        number1 = int(match.group(1))
        number2 = int(match.group(2))
        return number1, number2 
    match = re.search(r'(\d+)\s*tot\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match:
        number1 = int(match.group(1))
        number2 = int(match.group(2))
        return number1, number2 
    return None, None

def extract_range_from_question(question):
    match = re.search(r"(leeftijdscategorie|leeftijdsgroep)\s*(\d+)", question, re.IGNORECASE)
    if match:
        return int(match.group(2))  
    return None

def extract_month_from_question(question):
    match = re.search(r"\b(\d+)\s*maanden\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_threshold_from_question(question):
    match = re.search(r"\b(\d+)\s*keer\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_signal_in_question(question: str, signal_df: pd.DataFrame) -> str:
    for signal in signal_df['Signal']:
        if signal in question:
            return signal
    return None

def extract_relatie_in_question(question: str) -> str:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    for relatie in relaties:
        if relatie in question:
            return relatie
    return None

def extract_stoornis_in_question(question: str) -> str:
    mapping = {
        'taalontwikkelingsprobleem': 'language disorder',
        'taalontwikkelingsproblemen': 'language disorder',
        'taalontwikkelingsstoornis': 'language disorder',
        'taalontwikkelingstoornissen': 'language disorder',
        'motorische ontwikkelingproblemen': 'motoric disorder',
        'motorisch ontwikkelingsprobleem': 'motoric disorder',
        'motorische ontwikkelingsstoornis': 'motoric disorder',
        'motorische ontwikkelingsstoornissen': 'motoric disorder',
        'autisme': 'autism'}
    for key in mapping:
        if re.search(rf'\b{re.escape(key)}\b', question, re.IGNORECASE):
            return mapping[key]
    return None

def extract_domein_in_question(question: str) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    for domein in domeinen:
        if domein in question:
            return domein
    return None

#frequency questions
def analyze_freq_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relation_types = data['relation'].value_counts()
    total_entries = len(data)
    result_df = pd.DataFrame({
        "Relatie": relation_types.index,
        "Aantal keer gekozen": relation_types.values,
        "Percentage gekozen keren (%)": (relation_types.values / total_entries) * 100})
    result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
    return result_df

def analyze_freq_signaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    result_df = signal_df[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
    result_df.columns = ['Signaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren (%)']
    result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
    result_df = result_df.sort_values(by='Percentage gekozen keren (%)', ascending=False)
    return result_df

def analyze_freq_alarmsignaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    result_df = alarm_signals[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
    result_df.columns = ['Alarmsignaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren (%)']
    result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
    result_df = result_df.sort_values(by='Percentage gekozen keren (%)', ascending=False)
    return result_df

def analyze_freq_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domain_counts = {
        "Taal en/of communicatie": sum('taal en/of communicatie' in i for i in data['domains']),
        "Motoriek": sum('motoriek' in i for i in data['domains']),
        "Sociale vaardigheden": sum('sociale vaardigheden' in i for i in data['domains']),
        "Gedrag en spel": sum('gedrag en spel' in i for i in data['domains'])}
    result_df = pd.DataFrame(domain_counts.items(), columns=['Domein', 'Aantal keer gekozen'])
    return result_df

def analyze_freq_leeftijd(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    age_intervals = {
        1: "0-6 maanden",
        2: "6-12 maanden",
        3: "12-18 maanden",
        4: "18-24 maanden",
        5: "24-30 maanden",
        6: "30-36 maanden",
        7: "36-42 maanden",
        8: "42-48 maanden",
        9: "48-54 maanden",
        10: "54-60 maanden",
        11: "60-66 maanden",
        12: "66-72 maanden"}
    age_types = data['age_range'].value_counts()
    total_entries = len(data)
    result_df = pd.DataFrame({
        'Leeftijdscategorie': age_types.index,
        'Maand-interval': [age_intervals.get(age, "Onbekend") for age in age_types.index],
        'Aantal keer gekozen': age_types.values,
        'Percentage gekozen keren (%)': (age_types.values / total_entries) * 100})
    result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
    return result_df

def analyze_freq_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    disorder_counts = {
        "Taalontwikkelingsstoornis": sum('language disorder' in row['positive'] for _, row in data.iterrows()),
        "Motorische ontwikkelingsstoornis": sum('motoric disorder' in row['positive'] for _, row in data.iterrows()),
        "Autisme": sum('autism' in row['positive'] for _, row in data.iterrows())}
    result_df = pd.DataFrame(disorder_counts.items(), columns=['Ontwikkelingsprobleem', 'Aantal keer als positief aangeduid'])
    return result_df

def analyze_stoornis_specific(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    stoornis = extract_stoornis_in_question(specific_question)
    value_counts_1 = 0
    value_counts_2 = 0
    value_counts_3 = 0
    for _, row in data.iterrows():  
        if 'language disorder' in row['positive']:
            value_counts_1 += 1
        if 'motoric disorder' in row['positive']:
            value_counts_2 += 1
        if 'autism' in row['positive']:
            value_counts_3 += 1
    values = [value_counts_1, value_counts_2, value_counts_3]
    if stoornis == 'language disorder':
        stoornis_value = value_counts_1
    if stoornis == 'motoric disorder':
        stoornis_value = value_counts_2
    if stoornis == 'autism':
        stoornis_value = value_counts_3
    analysis_result = ""
    analysis_result += f"Kinderen met dit ontwikkelingsprobleem scoorden {stoornis_value} keer boven de cut-off.\n"
    return analysis_result

def analyze_relatie_specific(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    relatie = extract_relatie_in_question(specific_question)
    relation_types = data['relation'].value_counts()
    total_entries = len(data)
    result_df = pd.DataFrame({
        "Relatie": relation_types.index,
        "Aantal keer gekozen": relation_types.values,
        "Percentage gekozen keren (%)": (relation_types.values / total_entries) * 100})
    filtered_row = result_df[result_df['Relatie'].str.contains('relatie', case=False)]
    aantal_keer = filtered_row['Aantal keer gekozen'].values[0]
    percentage = filtered_row['Percentage gekozen keren (%)'].values[0]
    analysis_result = ""
    analysis_result += f"De relatie {relatie} werd {aantal_keer} keer gekozen. Dit is een percentage van {percentage}.\n"
    return analysis_result

def analyze_domein_specific(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    domein = extract_domein_in_question(specific_question)
    domain_counts = {
        "Taal en/of communicatie": sum('taal en/of communicatie' in i for i in data['domains']),
        "Motoriek": sum('motoriek' in i for i in data['domains']),
        "Sociale vaardigheden": sum('sociale vaardigheden' in i for i in data['domains']),
        "Gedrag en spel": sum('gedrag en spel' in i for i in data['domains'])}
    result_df = pd.DataFrame(domain_counts.items(), columns=['Domein', 'Aantal keer gekozen'])
    filtered_row = result_df[result_df['Domein'].str.contains('domein', case=False)]
    aantal_keer = filtered_row['Aantal keer gekozen'].values[0]
    analysis_result = ""
    analysis_result += f"Het domein {domein} werd {aantal_keer} keer gekozen.\n"
    return analysis_result

def analyze_leeftijd_specific(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    age_1, age_2 = extract_months_from_question(specific_question)
    mask = (data['age_months'] >= age_1) & (data['age_months'] <= age_2)
    aantal_keer = mask.sum()  
    totaal = len(data)  
    percentage = (aantal_keer / totaal) * 100 
    analysis_result = ""
    analysis_result = f"De data bevat {aantal_keer} kinderen tussen de {age_1} en {age_2} maanden. Dit is een percentage van {percentage:.2f}%.\n"
    return analysis_result

def most_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    relation_types = data['relation'].value_counts()
    max_value = relation_types.max()
    max_relationship = relation_types.idxmax()
    analysis_result = ""
    analysis_result += f"De meest gekozen relatie is '{max_relationship}' die {max_value} keer werd gekozen.\n"
    return analysis_result

def most_leeftijd(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    leeftijd_types = data['age_range'].value_counts()
    max_value = leeftijd_types.max()
    max_leeftijd = leeftijd_types.idxmax()
    analysis_result = ""
    analysis_result += f"De meest gekozen leeftijdscategorie is'{max_leeftijd}' die {max_value} keer werd gekozen.\n"
    return analysis_result

def most_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    language = 0
    motoric = 0
    social = 0
    behaviour = 0
    domain_types = data['domains']
    for i in domain_types:
        if 'taal en/of communicatie' in i:
            language += 1
        if 'motoriek' in i:
            motoric += 1
        if 'sociale vaardigheden' in i:
            social += 1
        if 'gedrag en spel' in i:
            behaviour += 1
    domain_value = [language, motoric, social, behaviour]
    domain_name = ['Taal en/of communicatie', 'Motoriek', 'Sociale vaardigheden', 'Gedrag en spel']
    max_value = max(domain_value)
    max_index = domain_value.index(max_value)
    max_domain = domain_name[max_index]
    analysis_result = ""
    analysis_result += f"Het meest gekozen domein is '{max_domain}' dat {max_value} keer werd gekozen.\n"
    return analysis_result

def most_signaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    max_percentage_row = signal_df.loc[signal_df['Percentage gekozen keren'].idxmax()]
    vraag = max_percentage_row['Vraag']
    aantal_keer_zichtbaar = max_percentage_row['Aantal keer zichtbaar']
    percentage_gekozen = max_percentage_row['Percentage gekozen keren']
    analysis_result = f"Het signaal '{vraag}' is het meest gekozen signaal met een percentage van {percentage_gekozen}%. Dit signaal was {aantal_keer_zichtbaar} keer zichtbaar."
    return analysis_result

def most_alarmsignaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    max_percentage_row = alarm_signals.loc[alarm_signals['Percentage gekozen keren'].idxmax()]
    vraag = max_percentage_row['Vraag']
    aantal_keer_zichtbaar = max_percentage_row['Aantal keer zichtbaar']
    percentage_gekozen = max_percentage_row['Percentage gekozen keren']
    analysis_result = (
        f"Het alarmsignaal '{vraag}' is het meest gekozen alarmsignaal met een percentage van {percentage_gekozen}%. "
        f"Dit alarmsignaal was {aantal_keer_zichtbaar} keer zichtbaar.")
    return analysis_result

def most_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    value_counts_1 = 0
    value_counts_2 = 0
    value_counts_3 = 0
    for _, row in data.iterrows():  
        if 'language disorder' in row['positive']:
            value_counts_1 += 1
        if 'motoric disorder' in row['positive']:
            value_counts_2 += 1
        if 'autism' in row['positive']:
            value_counts_3 += 1
    values = [value_counts_1, value_counts_2, value_counts_3]
    max_stoornis = max(values)
    if max_stoornis == value_counts_1:
        stoornis_value = value_counts_1
        stoornis = 'Taalontwikkelingsprobleem'
    if max_stoornis == value_counts_2:
        stoornis_value = value_counts_2
        stoornis = 'Motorisch ontwikkelingsprobleem'
    if max_stoornis == value_counts_3:
        stoornis_value = value_counts_3
        stoornis = 'Autisme'
    analysis_result = ""
    analysis_result += f"Het ontwikkelingsprobleem die het meest werd aangegeven als positief is '{stoornis}'. Deze werd {stoornis_value} keer gekozen.\n"
    return analysis_result

def least_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    relation_types = data['relation'].value_counts()
    min_value = relation_types.min()  
    min_relationship = relation_types.idxmin()  
    analysis_result = ""
    analysis_result += f"De minst gekozen relatie is '{min_relationship}' die {min_value} keer werd gekozen.\n"
    return analysis_result

def least_leeftijd(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    leeftijd_types = data['age_range'].value_counts()
    min_value = leeftijd_types.min()  
    min_leeftijd = leeftijd_types.idxmin() 
    analysis_result = ""
    analysis_result += f"De minst gekozen leeftijdscategorie is '{min_leeftijd}' die {min_value} keer werd gekozen.\n"
    return analysis_result

def least_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    language = 0
    motoric = 0
    social = 0
    behaviour = 0
    domain_types = data['domains']
    for i in domain_types:
        if 'taal en/of communicatie' in i:
            language += 1
        if 'motoriek' in i:
            motoric += 1
        if 'sociale vaardigheden' in i:
            social += 1
        if 'gedrag en spel' in i:
            behaviour += 1
    domain_value = [language, motoric, social, behaviour]
    domain_name = ['Taal en/of communicatie', 'Motoriek', 'Sociale vaardigheden', 'Gedrag en spel']
    min_value = min(domain_value)  
    min_index = domain_value.index(min_value)  
    min_domain = domain_name[min_index]
    analysis_result = ""
    analysis_result += f"Het minst gekozen domein is '{min_domain}' dat {min_value} keer werd gekozen.\n"
    return analysis_result

def least_signaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    min_percentage_row = signal_df.loc[signal_df['Percentage gekozen keren'].idxmin()]
    vraag = min_percentage_row['Vraag']
    aantal_keer_zichtbaar = min_percentage_row['Aantal keer zichtbaar']
    percentage_gekozen = min_percentage_row['Percentage gekozen keren']
    analysis_result = f"Het signaal '{vraag}' is het minst gekozen signaal met een percentage van {percentage_gekozen}%. Dit signaal was {aantal_keer_zichtbaar} keer zichtbaar."
    return analysis_result

def least_alarmsignaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    max_percentage_row = alarm_signals.loc[alarm_signals['Percentage gekozen keren'].idxmin()]
    vraag = max_percentage_row['Vraag']
    aantal_keer_zichtbaar = max_percentage_row['Aantal keer zichtbaar']
    percentage_gekozen = max_percentage_row['Percentage gekozen keren']
    analysis_result = (
        f"Het alarmsignaal '{vraag}' is het minst gekozen alarmsignaal met een percentage van {percentage_gekozen}%. "
        f"Dit alarmsignaal was {aantal_keer_zichtbaar} keer zichtbaar.")
    return analysis_result

def least_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    value_counts_1 = 0
    value_counts_2 = 0
    value_counts_3 = 0
    for _, row in data.iterrows():  
        if 'language disorder' in row['positive']:
            value_counts_1 += 1
        if 'motoric disorder' in row['positive']:
            value_counts_2 += 1
        if 'autism' in row['positive']:
            value_counts_3 += 1
    values = [value_counts_1, value_counts_2, value_counts_3]
    max_stoornis = min(values)
    if max_stoornis == value_counts_1:
        stoornis_value = value_counts_1
        stoornis = 'Taalontwikkelingsprobleem'
    if max_stoornis == value_counts_2:
        stoornis_value = value_counts_2
        stoornis = 'Motorisch ontwikkelingsprobleem'
    if max_stoornis == value_counts_3:
        stoornis_value = value_counts_3
        stoornis = 'Autisme'
    analysis_result = ""
    analysis_result += f"Het ontwikkelingsprobleem die het minst werd aangegeven als positief is '{stoornis}'. Deze werd {stoornis_value} keer gekozen.\n"
    return analysis_result

def top_most_signals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    if number is None:
        number = extract_number_from_signal_question(specific_question)
    if number is None:
        number  = 5
    top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
    return top_signals_df.rename(columns={'Vraag': 'Signaal'})[['Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def top_least_signals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    if number is None:
        number = extract_number_from_signal_question(specific_question)
    if number is None:
        number  = 5
    top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=True).head(number)
    return top_signals_df.rename(columns={'Vraag': 'Signaal'})[['Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_below_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    if number is None:
        number = 5
    filtered_signals = signal_df[signal_df['Percentage gekozen keren'] < number]
    if filtered_signals.empty:
        return f"Geen signalen komen voor in minder dan {number}% van de gevallen."
    return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_above_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    if number is None:
        number = 95
    filtered_signals = signal_df[signal_df['Percentage gekozen keren'] > number]
    if filtered_signals.empty:
        return f"Geen signalen komen voor in meer dan {number}% van de gevallen."
    return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def top_most_alarmsignals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    if number is None:
        number = extract_number_from_alarm_question(specific_question)
    if number is None:
        number = 5 
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    top_signals_df = alarm_signals.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
    return top_signals_df.rename(columns={'Vraag': 'Alarmsignaal'})[['Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def top_least_alarmsignals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    if number is None:
        number = extract_number_from_alarm_question(specific_question) 
    if number is None:
        number = 5
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    top_signals_df = alarm_signals.sort_values(by='Percentage gekozen keren', ascending=True).head(number)
    return top_signals_df.rename(columns={'Vraag': 'Alarmsignaal'})[['Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_alarm_below_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    if number is None:
        number = 5
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    filtered_signals = alarm_signals[signal_df['Percentage gekozen keren'] < number]
    if filtered_signals.empty:
        return f"Geen alarmsignalen komen voor in minder dan {number}% van de gevallen."
    return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_alarm_above_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    if number is None:
        number = 95
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    filtered_signals = alarm_signals[signal_df['Percentage gekozen keren'] > number]
    if filtered_signals.empty:
        return f"Geen alarmsignalen komen voor in minder dan {number}% van de gevallen."
    return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def how_many_alarm(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    threshold = extract_number_from_alarm_question(specific_question)
    alarm_signals_df = signal_df[(signal_df['Is een alarmsignaal'] == True)]
    relevant_signals = alarm_signals_df['Vraag'].unique()
    datapoints_with_alarm = []
    for index, row in data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            alarm_count = sum(1 for signal in signals if signal in relevant_signals)
            if alarm_count >= threshold:
                datapoints_with_alarm.append({
                    'Index': index,
                    'Aantal alarmsignalen': alarm_count})
    result_df = pd.DataFrame(datapoints_with_alarm)
    total_children = len(data)
    children_with_alarm = len(result_df)
    if not result_df.empty:
        percentage = (children_with_alarm / total_children) * 100
        analysis_results = ""
        analysis_results += f"Het aantal kinderen met tenminste {threshold} alarmsignalen/alarmsignaal is:"
        analysis_results += f"{len(data)}"
        analysis_results += f"Dit bedraagt een percentage van {percentage:.2f}% van de kinderen."
    return analysis_results

def how_many_signal(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    analysis_results = ""
    domein = extract_domein_in_question(specific_question)
    if domein is None:
        stoornis = extract_stoornis_in_question(specific_question)
    if domein is None and stoornis is None:
        analysis_results += f"Er werd geen ontwikkelingsprobleem of domein herkend in de vraag. Is alles juist geschreven?"
        return analysis_results
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_signal_question(specific_question)
    if number is None:
        analysis_results += f"Er werd geen hoeveelheid aan signalen herkend in deze vraag."
        return analysis_results
    datapoints_with_count = []
    for index, row in data.iterrows():
        count = 0
        if domein is not None:
            if isinstance(row.get("domains"), list) and domein in row["domains"]:
                if isinstance(row.get("signals"), list):
                    count = sum(1 for signal in row["signals"]
                        if signal in signal_df["Vraag"].values and 
                        signal_df.loc[signal_df["Vraag"] == signal, "Domein"].values[0] == domein)
        elif stoornis is not None:
            if isinstance(row.get("positive"), list) and stoornis in row["positive"]:
                if isinstance(row.get("signals"), list):
                    count = sum(1 for signal in row["signals"]
                        if signal in signal_df["Vraag"].values and 
                        signal_df.loc[signal_df["Vraag"] == signal, "Disorder"].values[0] == stoornis)
        if count > 0:
            datapoints_with_count.append({"Index": index, "Aantal signalen": count})
    result_df = pd.DataFrame(datapoints_with_count)
    total_children = len(data)
    if result_df.empty and domein is not None:
        analysis_results += f"Er zijn geen kinderen gevonden met tenminste {number} signalen in het domein {domein}."
        return analysis_results
    if result_df.empty and stoornis is not None:
        analysis_results += f"Er zijn geen kinderen gevonden met tenminste {number} signalen voor het ontwikkelingsprobleem {stoornis}."
        return analysis_results
    if not result_df.empty:
        children_with_threshold = len(result_df[result_df["Aantal signalen"] > number])
        percentage = (children_with_threshold / total_children) * 100
        if stoornis is not None:
            total_children_with_stoornis = len(data[data["positive"].apply(lambda x: isinstance(x, list) and stoornis in x)])
            if total_children_with_stoornis > 0:
                percentage_2 = (children_with_threshold / total_children_with_stoornis) * 100
        if domein is not None:
            total_children_with_domein = len(data[data["domains"].apply(lambda x: isinstance(x, list) and domein in x)])
            if total_children_with_domein > 0:
                percentage_2 = (children_with_threshold / total_children_with_domein) * 100
        analysis_results += f"Het aantal kinderen met meer dan {number} signalen is: {children_with_threshold}.\n"
        analysis_results += f"Dit bedraagt een percentage van {percentage:.2f}% van het totaal aantal kinderen." 
        if stoornis is not None:
            analysis_results += f"Dit bedraagt een percentage van {percentage_2:.2f}% van het totaal aantal kinderen die boven de cut-off scoorden voor {stoornis}."
        if domein is not None:
            analysis_results += f"Dit bedraagt een percentage van {percentage_2:.2f}% van het totaal aantal kinderen die waarvoor het domein {domein} gekozen werd."
    return analysis_results
    
#verbanden questions
def combo_most_relatie_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    if domein is None or relatie is None:
        return f"\nEr werd geen domein of relatie herkent in de vraag."
    filtered_data = data[(data['relation'] == relatie) & (data['domains'].apply(lambda x: domein in x))]
    if filtered_data.empty:
        return f"\nEr zijn geen signalen om weer te geven."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"\nEr zijn geen signalen om weer te geven."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x, na=False)]['Percentage gekozen keren'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x, na=False)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x, na=False)]['Aantal keer zichtbaar'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x, na=False)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=False)
    return signalen_df.head(number)

def combo_least_relatie_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    if domein is None or relatie is None:
        return f"\nEr werd geen domein of relatie herkent in de vraag."
    filtered_data = data[(data['relation'] == relatie) & (data['domains'].apply(lambda x: domein in x))]
    if filtered_data.empty:
        return f"\nEr zijn geen signalen om weer te geven."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"\nEr zijn geen signalen om weer te geven."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x, na=False)]['Percentage gekozen keren'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x, na=False)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x, na=False)]['Aantal keer zichtbaar'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x, na=False)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=True)
    return signalen_df.head(number)

def combo_most_domein_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    if not domein or not stoornis:
        return f"\nEr werd geen domein of ontwikkelingsprobleem herkent in de vraag."
    filtered_data = data[(data['domains'].apply(lambda x: domein in x)) & (data['positive'].apply(lambda x: stoornis in x))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwtikkelingsprobleem '{stoornis}' in het domein '{domein}'."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor het ontwikkelingsprobleem '{stoornis}' in het domein '{domein}'."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    return signalen_df.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
    
def combo_least_domein_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame)-> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    if not domein or not stoornis:
        return f"\nEr werd geen domein of ontwikkelingsprobleem herkent in de vraag."
    filtered_data = data[(data['domains'].apply(lambda x: domein in x)) & (data['positive'].apply(lambda x: stoornis in x))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwtikkelingsprobleem '{stoornis}' in het domein '{domein}'."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor het ontwikkelingsprobleem '{stoornis}' in het domein '{domein}'."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    return signalen_df.sort_values(by='Percentage gekozen keren', ascending=True).head(number)

def combo_most_domain_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if domein is None:
        return f"\nEr werd geen domein herkent in de vraag."
    filtered_data = data[(data['domains'].apply(lambda x: domein in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het domein '{domein}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                dom == domein and min_leeftijd <= age_months <= max_leeftijd and 
                signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor het domein '{domein}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=False)
    return signalen_df.head(number) 

def combo_least_domain_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if domein is None:
        return f"\nEr werd geen domein herkent in de vraag."
    filtered_data = data[(data['domains'].apply(lambda x: domein in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het domein '{domein}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                dom == domein and min_leeftijd <= age_months <= max_leeftijd and 
                signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor het domein '{domein}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=True)
    return signalen_df.head(number) 

def combo_most_relatie_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    if relatie is None:
        return  f"\nEr werd geen relatie herkent in de vraag."
    filtered_data = data[(data['relation']==relatie) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor de relatie '{relatie}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                min_leeftijd <= age_months <= max_leeftijd and 
                signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor de relatie '{relatie}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=False)
    return signalen_df.head(number)

def combo_least_relatie_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    if relatie is None:
        return  f"\nEr werd geen relatie herkent in de vraag."
    filtered_data = data[(data['relation']==relatie) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor de relatie '{relatie}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                min_leeftijd <= age_months <= max_leeftijd and 
                signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return f"Er zijn geen signalen gevonden voor de relatie '{relatie}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=True)
    return signalen_df.head(number)

def combo_most_domain_age_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if relatie is None:
        return f"Er werd geen relatie herkend in de vraag."
    if domein is None:
        return f"Er werd geen domein herkend in de vraag."
    filtered_data = data[
        (data['domains'].apply(lambda x: domein in x)) & 
        (data['age_months'].between(age_1, age_2)) & 
        (data['relation'] == relatie)]
    if filtered_data.empty:
        return (f"Er zijn geen signalen gevonden voor het domein '{domein}' met relatie '{relatie}' "
                f"voor een leeftijd tussen '{age_1}' en '{age_2}' maanden.")
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                    dom == domein and min_leeftijd <= age_months <= max_leeftijd and 
                    signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return (f"Er zijn geen signalen gevonden voor het domein '{domein}' met relatie '{relatie}' "
                f"voor een leeftijd tussen '{age_1}' en '{age_2}' maanden. ")
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=False)
    return signalen_df.head(number)
    
def combo_least_domain_age_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if relatie is None:
        return f"Er werd geen relatie herkend in de vraag."
    if domein is None:
        return f"Er werd geen domein herkend in de vraag."
    filtered_data = data[
        (data['domains'].apply(lambda x: domein in x)) & 
        (data['age_months'].between(age_1, age_2)) & 
        (data['relation'] == relatie)]
    if filtered_data.empty:
        return (f"Er zijn geen signalen gevonden voor het domein '{domein}' met relatie '{relatie}' "
                f"voor een leeftijd tussen '{age_1}' en '{age_2}' maanden.")
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        min_leeftijd = row['Minimum leeftijd']
        max_leeftijd = row['Maximum leeftijd']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            age_months = signal_row['age_months']
            for signal in signalen_lijst:
                if (isinstance(signal, str) and signal in vraag and 
                    dom == domein and min_leeftijd <= age_months <= max_leeftijd and 
                    signal not in gevonden_signalen):
                    gevonden_signalen.append(signal)
    if not gevonden_signalen:
        return (f"Er zijn geen signalen gevonden voor het domein '{domein}' met relatie '{relatie}' "
                f"voor een leeftijd tussen '{age_1}' en '{age_2}' maanden. ")
    signalen_df = pd.DataFrame(gevonden_signalen, columns=['Signaal'])
    signalen_df['Percentage gekozen keren'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Percentage gekozen keren'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df['Aantal keer zichtbaar'] = signalen_df['Signaal'].apply(
        lambda x: signal_df[signal_df['Vraag'].str.contains(x)]['Aantal keer zichtbaar'].values[0] 
        if len(signal_df[signal_df['Vraag'].str.contains(x)]) > 0 else None)
    signalen_df = signalen_df.sort_values(by='Percentage gekozen keren', ascending=True)
    return signalen_df.head(number)

def combo_most_signal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    all_signals = set(signal for signals_list in data["signals"] for signal in signals_list)
    one_hot = pd.DataFrame({signal: data["signals"].apply(lambda x: signal in x) for signal in all_signals}).astype(int)
    co_occurrence = one_hot.T.dot(one_hot)
    np.fill_diagonal(co_occurrence.values, 0)
    signal_df['leeftijd_range'] = list(zip(signal_df['Minimum leeftijd'], signal_df['Maximum leeftijd']))
    signal_df_indexed = defaultdict(list)
    for _, row in signal_df.iterrows():
        signal_df_indexed[row['Vraag']].append(row['leeftijd_range'])
    co_occurrence_pairs = []
    for signal1 in co_occurrence.index:
        for signal2 in co_occurrence.columns:
            if signal1 < signal2: 
                co_occurrence_pairs.append((signal1, signal2, co_occurrence.loc[signal1, signal2]))
    pair_percentages = []
    for signal1, signal2, freq in co_occurrence_pairs:
        count = 0
        for age in data['age_months']:
            signal1_age_ranges = signal_df_indexed.get(signal1, [])
            signal2_age_ranges = signal_df_indexed.get(signal2, [])
            signal1_valid = any(min_age <= age <= max_age for min_age, max_age in signal1_age_ranges)
            signal2_valid = any(min_age <= age <= max_age for min_age, max_age in signal2_age_ranges)
            if signal1_valid and signal2_valid:
                count += 1
        if count > 10:
            percentage = (freq / count) * 100
            pair_percentages.append((signal1, signal2, freq, count, percentage))
    sorted_pairs = sorted(pair_percentages, key=lambda x: x[4], reverse=True)[:5]
    result_df = pd.DataFrame(sorted_pairs, columns=["Signaal 1", "Signaal 2", "Aantal keer samen gekozen", "Aantal keer samen zichtbaar", "Percentage"])
    return result_df

def combo_least_signal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    all_signals = set(signal for signals_list in data["signals"] for signal in signals_list)
    one_hot = pd.DataFrame({signal: data["signals"].apply(lambda x: signal in x) for signal in all_signals}).astype(int)
    co_occurrence = one_hot.T.dot(one_hot)
    np.fill_diagonal(co_occurrence.values, 0)
    signal_df['leeftijd_range'] = list(zip(signal_df['Minimum leeftijd'], signal_df['Maximum leeftijd']))
    signal_df_indexed = defaultdict(list)
    for _, row in signal_df.iterrows():
        signal_df_indexed[row['Vraag']].append(row['leeftijd_range'])
    co_occurrence_pairs = []
    for signal1 in co_occurrence.index:
        for signal2 in co_occurrence.columns:
            if signal1 < signal2: 
                co_occurrence_pairs.append((signal1, signal2, co_occurrence.loc[signal1, signal2]))
    pair_percentages = []
    for signal1, signal2, freq in co_occurrence_pairs:
        count = 0
        for age in data['age_months']:
            signal1_age_ranges = signal_df_indexed.get(signal1, [])
            signal2_age_ranges = signal_df_indexed.get(signal2, [])
            signal1_valid = any(min_age <= age <= max_age for min_age, max_age in signal1_age_ranges)
            signal2_valid = any(min_age <= age <= max_age for min_age, max_age in signal2_age_ranges)
            if signal1_valid and signal2_valid:
                count += 1
        if count > 10:
            percentage = (freq / count) * 100
            pair_percentages.append((signal1, signal2, freq, count, percentage))
    sorted_pairs = sorted(pair_percentages, key=lambda x: x[4], reverse=False)[:5]
    result_df = pd.DataFrame(sorted_pairs, columns=["Signaal 1", "Signaal 2", "Aantal keer samen gekozen", "Aantal keer samen zichtbaar", "Percentage"])
    return result_df

def combo_most_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x if isinstance(x, list) else False))]
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_most_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    age_1, age_2 = extract_months_from_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    if age_1 is None or age_2 is None:
        age_1, age_2 = extract_month_from_question(specific_question), 0
    filtered_data = data[(data['age_months'].between(age_1, age_2))]
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    age_1, age_2 = extract_months_from_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    if age_1 is None or age_2 is None:
        age_1, age_2 = extract_month_from_question(specific_question), 0
    filtered_data = data[(data['age_months'].between(age_1, age_2))]
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)      

def combo_most_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domein = extract_domein_in_question(specific_question)
    filtered_data = data[(data['domains'].apply(lambda x: domein in x))]
    gevonden_signalen = []
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number) 

def combo_least_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domein = extract_domein_in_question(specific_question)
    filtered_data = data[(data['domains'].apply(lambda x: domein in x))]
    gevonden_signalen = []
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number) 

def combo_most_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relatie = extract_relatie_in_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['relation'] == relatie)]
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number) 

def combo_least_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relatie = extract_relatie_in_question(specific_question)
    number = extract_number_from_signal_question
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['relation'] == relatie)]
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)  

def zelden_voorkomen(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    threshold = 5
    number = extract_range_from_question(specific_question)
    age_data = data[data['age_range'] == number]
    all_signals = [signal for signals_list in age_data['signals'] for signal in signals_list]
    signals_df = signal_df[signal_df['Vraag'].isin(all_signals)]
    low_percentage_signals = signals_df[signals_df['Percentage gekozen keren'] < threshold]
    if not low_percentage_signals.empty:
        low_percentage_signals = low_percentage_signals.copy()
        low_percentage_signals["Leeftijdscategorie"] = number
        return low_percentage_signals[['Leeftijdscategorie', 'Signaal', 'Percentage gekozen keren']]
    else:
        return f"Er werden geen signalen gevonden."

def vaak_voorkomen(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    threshold = 5
    number = extract_range_from_question(specific_question)
    age_data = data[data['age_range'] == number]
    all_signals = [signal for signals_list in age_data['signals'] for signal in signals_list]
    signals_df = signal_df[signal_df['Vraag'].isin(all_signals)]
    low_percentage_signals = signals_df[signals_df['Percentage gekozen keren'] > threshold]
    if not low_percentage_signals.empty:
        low_percentage_signals = low_percentage_signals.copy()
        low_percentage_signals["Leeftijdscategorie"] = number
        return low_percentage_signals[['Leeftijdscategorie', 'Signaal', 'Percentage gekozen keren']]
    else:
        return f"Er werden geen signalen gevonden."

def combo_most_relatie_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    stoornis = extract_stoornis_in_question(specific_question)
    if not relatie or not stoornis:
        return f"\nEr werd geen relatie of ontwikkelingsprobleem herkent in de vraag."
    filtered_data = data[(data['relation'] == relatie) & (data['positive'].apply(lambda x: stoornis in x))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en de relatie '{relatie}'."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Geen relevante signalen gevonden voor ontwikkelingsprobleem '{stoornis}' en relatie '{relatie}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_relatie_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    stoornis = extract_stoornis_in_question(specific_question)
    if not relatie or not stoornis:
        return f"\nEr werd geen relatie of ontwikkelingsprobleem herkent in de vraag."
    filtered_data = data[(data['relation'] == relatie) & (data['positive'].apply(lambda x: stoornis in x))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en de relatie '{relatie}'."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Geen relevante signalen gevonden voor ontwikkelingsprobleem '{stoornis}' en relatie '{relatie}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_most_stoornis_age_domain(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    stoornis = extract_stoornis_in_question(specific_question)
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if not stoornis or not domein:
        return f"Er is geen ontwikkelingsprobleem of domein gevonden in de vraag."
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['domains'].apply(lambda x: dom in x)) & 
        (data['age_months'] >= age_1) & 
        (data['age_months'] <= age_2)]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en het domein '{domein}'."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en het domein '{domein}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_stoornis_age_domain(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    stoornis = extract_stoornis_in_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if not stoornis or not domein:
        return f"Er is geen ontwikkelingsprobleem of domein gevonden in de vraag."
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['domains'].apply(lambda x: dom in x)) & 
        (data['age_months'] >= age_1) & 
        (data['age_months'] <= age_2)]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en het domein '{domein}'."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor het onwikkelingsprobleem '{stoornis}' en het domein '{domein}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_most_stoornis_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden. Tip: Check of alles juist geschreven is."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor een leeftijd tussen '{age_1}' en '{age_2}' en het ontwikkelingsprobleem '{stoornis}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_stoornis_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden. Tip: Check of alles juist geschreven is."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor een leeftijd tussen '{age_1}' en '{age_2}' en het ontwikkelingsprobleem '{stoornis}'."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_most_age_relatie_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    filtered_data = data[
        (data['relation'] == relatie) &
        (data['positive'].apply(lambda x: stoornis in x)) &
        (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return (
            f"Er zijn geen signalen gevonden voor relatie '{relatie}', "
            f"een positieve stoornis '{stoornis}', en een leeftijd tussen '{age_1}' en '{age_2}' maanden. "
            "Tip: Check of alles juist geschreven is.")
    gevonden_signalen = []

def combo_most_stoornis_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_age_relatie_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    filtered_data = data[
        (data['relation'] == relatie) &
        (data['positive'].apply(lambda x: stoornis in x)) &
        (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return (
            f"Er zijn geen signalen gevonden voor relatie '{relatie}', "
            f"een positieve stoornis '{stoornis}', en een leeftijd tussen '{age_1}' en '{age_2}' maanden. "
            "Tip: Check of alles juist geschreven is.")
    gevonden_signalen = []

def combo_most_stoornis_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x)) & (data['age_months'].between(age_1, age_2))]
    if filtered_data.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    if signal_df_sorted.empty:
        return f"Er zijn geen signalen gevonden voor een positieve stoornis '{stoornis}' voor een leeftijd tussen '{age_1}' en '{age_2}' maanden."
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_most_domain_stoornis_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for rel in relaties:
        if rel in specific_question:
            relatie = rel        
    filtered_data = data[
        (data['domains'].apply(lambda x: domein in x)) & 
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['relation'] == relatie)]
    if filtered_data.empty:
        return (f"Er zijn geen signalen gevonden voor de stoornis '{stoornis}' in het domein '{domein}' "
                f"met relatie '{relatie}'. Tip: Check of het domein, de stoornis en de relatie juist geschreven zijn.")
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)   
    if not gevonden_signalen:
        return (f"Er zijn geen signalen gevonden voor de stoornis '{stoornis}' in het domein '{domein}' "
                f"met relatie '{relatie}' die overeenkomen met de vraag. "
                "Tip: Check of het domein, de stoornis en de relatie juist geschreven zijn.")
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_domain_stoornis_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for rel in relaties:
        if rel in specific_question:
            relatie = rel        
    filtered_data = data[
        (data['domains'].apply(lambda x: domein in x)) & 
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['relation'] == relatie)]
    if filtered_data.empty:
        return (f"Er zijn geen signalen gevonden voor de stoornis '{stoornis}' in het domein '{domein}' "
                f"met relatie '{relatie}'. Tip: Check of het domein, de stoornis en de relatie juist geschreven zijn.")
    gevonden_signalen = []
    for index, row in signal_df.iterrows():
        vraag = row['Vraag']
        dom = row['Domein']
        for _, signal_row in filtered_data.iterrows():
            signalen_lijst = signal_row['signals']
            for signal in signalen_lijst:
                if isinstance(signal, str) and signal in vraag:
                    if dom == domein and signal not in gevonden_signalen:
                        gevonden_signalen.append(signal)   
    if not gevonden_signalen:
        return (f"Er zijn geen signalen gevonden voor de stoornis '{stoornis}' in het domein '{domein}' "
                f"met relatie '{relatie}' die overeenkomen met de vraag. "
                "Tip: Check of het domein, de stoornis en de relatie juist geschreven zijn.")
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_most_all(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    analysis_results = ''
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x if isinstance(x, list) else False)) & 
        (data['domains'].apply(lambda x: domein in x if isinstance(x, list) else False)) & 
        (data['relation'] == relatie) & 
        (data['age_months'] >= age_1) & 
        (data['age_months'] <= age_2)]
    if filtered_data.empty:
        return f"Er zijn geen relevante signalen."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=False)
    return signal_df_sorted.head(number)

def combo_least_all(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    analysis_results = ''
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    stoornis = extract_stoornis_in_question(specific_question)
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for rel in relaties:
        if rel in specific_question:
            relatie = rel
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x if isinstance(x, list) else False)) & 
        (data['domains'].apply(lambda x: domein in x if isinstance(x, list) else False)) & 
        (data['relation'] == relatie) & 
        (data['age_months'] >= age_1) & 
        (data['age_months'] <= age_2)]
    if filtered_data.empty:
        return f"Er zijn geen relevante signalen."
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)

def combo_least_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    stoornis = extract_stoornis_in_question(specific_question)
    filtered_data = data[(data['positive'].apply(lambda x: stoornis in x if isinstance(x, list) else False))]
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 5
    gevonden_signalen = []
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                signal_row = signal_df[signal_df['Vraag'] == signal]
                if not signal_row.empty:
                    percentage = signal_row['Percentage gekozen keren'].values[0]
                    gevonden_signalen.append({'Signal': signal, 'Percentage gekozen keren': percentage})
    signal_df_sorted = pd.DataFrame(gevonden_signalen)
    signal_df_sorted = signal_df_sorted.drop_duplicates(subset='Signal')
    signal_df_sorted = signal_df_sorted.sort_values(by='Percentage gekozen keren', ascending=True)
    return signal_df_sorted.head(number)
    
#trends questions

def time_more(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold == None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_signals = [signal for signals in old_period['signals'].dropna() for signal in signals]
    new_signals = [signal for signals in new_period['signals'].dropna() for signal in signals]
    old_counts = pd.Series(old_signals).value_counts()
    new_counts = pd.Series(new_signals).value_counts()
    significant_signals = []
    for signal, new_count in new_counts.items():
        old_count = old_counts.get(signal, 0)
        if old_count > 0 and new_count >= threshold * old_count:  
            significant_signals.append(f"{signal} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_signals:
        analysis_results = "De signalen die meer gekozen zijn, zijn:\n" + "\n".join(significant_signals)
    else:
        analysis_results = "Er zijn geen signalen die meer voorkomen."
    return analysis_results

def time_less(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold == None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_signals = [signal for signals in old_period['signals'].dropna() for signal in signals]
    new_signals = [signal for signals in new_period['signals'].dropna() for signal in signals]
    old_counts = pd.Series(old_signals).value_counts()
    new_counts = pd.Series(new_signals).value_counts()
    significant_signals = []
    for signal, new_count in new_counts.items():
        old_count = old_counts.get(signal, 0)
        if old_count > 0 and new_count <= old_count / threshold: 
            significant_signals.append(f"{signal} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_signals:
        analysis_results = "De signalen die minder gekozen zijn, zijn::\n" + "\n".join(significant_signals)
    else:
        analysis_results = "Er zijn geen signalen die minder voorkomen."
    return analysis_results

def time_less_domein(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_domains = [domain for sublist in old_period['domains'].dropna() for domain in (sublist if isinstance(sublist, list) else [sublist])]
    new_domains = [domain for sublist in new_period['domains'].dropna() for domain in (sublist if isinstance(sublist, list) else [sublist])]
    old_counts = pd.Series(old_domains).value_counts()
    new_counts = pd.Series(new_domains).value_counts()
    significant_domains = []
    for domain, new_count in new_counts.items():
        old_count = old_counts.get(domain, 0)
        if old_count > 0 and new_count <= old_count / threshold: 
            significant_domains.append(f"{domain} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_domains:
        analysis_results = "De domeinen die minder gekozen zijn, zijn:\n" + "\n".join(significant_domains)
    else:
        analysis_results = "Er zijn geen domeinen die minder voorkomen."
    return analysis_results

def time_more_domein(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_domains = [domain for sublist in old_period['domains'].dropna() for domain in (sublist if isinstance(sublist, list) else [sublist])]
    new_domains = [domain for sublist in new_period['domains'].dropna() for domain in (sublist if isinstance(sublist, list) else [sublist])]
    old_counts = pd.Series(old_domains).value_counts()
    new_counts = pd.Series(new_domains).value_counts()
    significant_domains = []
    for domain, new_count in new_counts.items():
        old_count = old_counts.get(domain, 0)
        if old_count > 0 and new_count >= old_count * threshold: 
            significant_domains.append(f"{domain} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_domains:
        analysis_results = "De domeinen die meer gekozen zijn, zijn:\n" + "\n".join(significant_domains)
    else:
        analysis_results = "Er zijn geen domeinen die meer voorkomen."
    return analysis_results

def time_domain(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    exploded_data = filtered_data.explode('domains').dropna(subset=['domains'])
    monthly_counts = exploded_data.groupby(['month', 'domains']).size().reset_index(name='Totaal aantal')
    monthly_counts.rename(columns={'month': 'Maand', 'domains': 'Domein'}, inplace=True)
    return monthly_counts.sort_values(by=['Maand', 'Totaal aantal'], ascending=[True, False])

def time_add_data(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    new_data_count = len(filtered_data)
    analysis_results = f"In de laatste {months} maanden zijn er {new_data_count} nieuwe datapunten geregistreerd.\n"
    return analysis_results

def time_data(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    monthly_counts = filtered_data.groupby('month').size()
    monthly_counts.rename(columns={'month': 'Maand'}, inplace=True)
    return monthly_counts.sort_values(by='Maand')

def time_more_relatie(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_relaties = old_period['relation'].dropna()
    new_relaties = new_period['relation'].dropna()
    old_counts = old_relaties.value_counts()
    new_counts = new_relaties.value_counts()
    significant_relaties = []
    for relatie, new_count in new_counts.items():
        old_count = old_counts.get(relatie, 0)
        if old_count > 0 and new_count >= old_count * threshold: 
            significant_relaties.append(f"{relatie} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_relaties:
        analysis_results = "De relaties die meer gekozen zijn, zijn:\n" + "\n".join(significant_relaties)
    else:
        analysis_results = "Er zijn geen relaties die meer voorkomen."
    return analysis_results

def time_less_relatie(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_relaties = old_period['relation'].dropna()
    new_relaties = new_period['relation'].dropna()
    old_counts = old_relaties.value_counts()
    new_counts = new_relaties.value_counts()
    significant_relaties = []
    for relatie, new_count in new_counts.items():
        old_count = old_counts.get(relatie, 0)
        if old_count > 0 and new_count <= old_count / threshold: 
            significant_relaties.append(f"{relatie} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_relaties:
        analysis_results = "De relaties die minder gekozen zijn, zijn:\n" + "\n".join(significant_relaties)
    else:
        analysis_results = "Er zijn geen relaties die minder voorkomen."
    return analysis_results

def time_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    monthly_counts = (
        filtered_data.groupby(['month', 'relation'])
        .size()
        .reset_index(name='Total Count'))
    monthly_counts.rename(columns={'month': 'Maand', 'relation': 'Relatie'}, inplace=True)
    return monthly_counts.sort_values(by=['Maand', 'Relatie'])

def time_more_age(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_ages = old_period['age_range'].dropna()
    new_ages = new_period['age_range'].dropna()
    old_counts = old_ages.value_counts()
    new_counts = new_ages.value_counts()
    significant_ages = []
    for age, new_count in new_counts.items():
        old_count = old_counts.get(age, 0)
        if old_count > 0 and new_count >= old_count * threshold: 
            significant_ages.append(f"{age} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_ages:
        analysis_results = "De leeftijdscategorieën die meer gekozen zijn, zijn leeftijdscategorie:\n" + "\n".join(significant_ages)
    else:
        analysis_results = "Er zijn geen leeftijdscategorieën die meer voorkomen."
    return analysis_results

def time_less_age(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_ages = old_period['age_range'].dropna()
    new_ages = new_period['age_range'].dropna()
    old_counts = old_ages.value_counts()
    new_counts = new_ages.value_counts()
    significant_ages = []
    for age, new_count in new_counts.items():
        old_count = old_counts.get(age, 0)
        if old_count > 0 and new_count <= old_count / threshold: 
            significant_ages.append(f"{age} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_ages:
        analysis_results = "De leeftijdscategorieën die minder gekozen zijn, zijn leeftijdscategorie:\n" + "\n".join(significant_ages)
    else:
        analysis_results = "Er zijn geen leeftijdscategorieën die minder voorkomen."
    return analysis_results

def time_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    monthly_counts = (
        filtered_data.groupby(['month', 'age_range'])
        .size()
        .reset_index(name='Totaal aantal'))
    monthly_counts.rename(columns={'month': 'Maand', 'age_range': 'Leeftijdscategorie'}, inplace=True)
    return monthly_counts.sort_values(by=['Maand', 'Leeftijdscategorie'])

def time_more_stoornis(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_stoornis = [stoornis.strip() for sublist in old_period['positive'].dropna() 
                    for stoornis in (sublist.split(',') if isinstance(sublist, str) else [sublist])]
    new_stoornis = [stoornis.strip() for sublist in new_period['positive'].dropna() 
                    for stoornis in (sublist.split(',') if isinstance(sublist, str) else [sublist])]
    old_counts = pd.Series(old_stoornis).value_counts()
    new_counts = pd.Series(new_stoornis).value_counts()
    significant_stoornis = []
    for stoornis, new_count in new_counts.items():
        old_count = old_counts.get(stoornis, 0)
        if old_count > 0 and new_count >= old_count * threshold: 
            significant_stoornis.append(f"{stoornis} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_stoornis:
        analysis_results = "De limitaties die meer als positief gerekend zijn, zijn:\n" + "\n".join(significant_stoornis)
    else:
        analysis_results = "Er zijn geen limitaties die meer voorkomen."
    return analysis_results

def time_less_stoornis(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    month = extract_month_from_question(specific_question)
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=month)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    old_stoornis = [stoornis.strip() for sublist in old_period['positive'].dropna() 
                    for stoornis in (sublist.split(',') if isinstance(sublist, str) else [sublist])]
    new_stoornis = [stoornis.strip() for sublist in new_period['positive'].dropna() 
                    for stoornis in (sublist.split(',') if isinstance(sublist, str) else [sublist])]
    old_counts = pd.Series(old_stoornis).value_counts()
    new_counts = pd.Series(new_stoornis).value_counts()
    significant_stoornis = []
    for stoornis, new_count in new_counts.items():
        old_count = old_counts.get(stoornis, 0)
        if old_count > 0 and new_count <= old_count / threshold: 
            significant_stoornis.append(f"{stoornis} (Oud: {old_count}, Nieuw: {new_count})")
    if significant_stoornis:
        analysis_results = "De limitaties die minder als positief gerekend zijn, zijn:\n" + "\n".join(significant_stoornis)
    else:
        analysis_results = "Er zijn geen limitaties die minder voorkomen."
    return analysis_results

def time_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    months = extract_month_from_question(specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    exploded_data = filtered_data.dropna(subset=['positive'])
    monthly_counts = (
        exploded_data.groupby(['month', 'positive'])
        .size()
        .reset_index(name='Total Count'))
    monthly_counts.rename(columns={'month': 'Maand', 'positive': 'Ontwikkelingsprobleem'}, inplace=True)
    return monthly_counts.sort_values(by=['Maand', 'Ontwikkelingsprobleem'])

def stijgen_freq(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    range_age = extract_range_from_question(specific_question)
    filtered_data = data[(data['age_range'] == range_age)]
    if filtered_data.empty:
        return pd.DataFrame(columns=['Signal', 'Age', 'Frequency'])
    signal_age_frequencies = {}
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                age = row['age_months']
                if signal not in signal_age_frequencies:
                    signal_age_frequencies[signal] = {}
                signal_age_frequencies[signal][age] = signal_age_frequencies[signal].get(age, 0) + 1
    increasing_signals = []
    for signal, age_frequencies in signal_age_frequencies.items():
        sorted_ages = sorted(age_frequencies.keys())
        frequency_trend = [age_frequencies[age] for age in sorted_ages]
        for i in range(len(frequency_trend) - 3):  
            if all(frequency_trend[i+j] < frequency_trend[i+j+1] for j in range(3)):  
                increasing_signals.append({
                    'Signal': signal, 
                    'Age': sorted_ages[i:i+4],  
                    'Frequency': frequency_trend[i:i+4]})
    result_df = pd.DataFrame(increasing_signals)
    if result_df.empty:
        return pd.DataFrame(columns=['Signal', 'Age', 'Frequency'])
    return result_df

def dalen_freq(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    age_range= extract_range_from_question(specific_question)
    filtered_data = data[(data['age_range'] >= age_range)]
    if filtered_data.empty:
        return pd.DataFrame(columns=['Signal', 'Age', 'Frequency'])
    signal_age_frequencies = {}
    for index, row in filtered_data.iterrows():
        signals = row['signals']
        if isinstance(signals, list):
            for signal in signals:
                age = row['age_months']
                if signal not in signal_age_frequencies:
                    signal_age_frequencies[signal] = {}
                signal_age_frequencies[signal][age] = signal_age_frequencies[signal].get(age, 0) + 1
    decreasing_signals = []
    for signal, age_frequencies in signal_age_frequencies.items():
        sorted_ages = sorted(age_frequencies.keys())
        frequency_trend = [age_frequencies[age] for age in sorted_ages]
        for i in range(len(frequency_trend) - 3): 
            if all(frequency_trend[i+j] > frequency_trend[i+j+1] for j in range(3)):  
                decreasing_signals.append({
                    'Signal': signal, 
                    'Age': sorted_ages[i:i+4], 
                    'Frequency': frequency_trend[i:i+4]})
    result_df = pd.DataFrame(decreasing_signals)
    if result_df.empty:
        return pd.DataFrame(columns=['Signal', 'Age', 'Frequency'])
    return result_df

def time_sign_frame(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    signal = extract_signal_in_question(specific_question, signal_df)
    if signal is None:
        raise ValueError("Geen signaal gevonden in de zin. Check of het zeker juist geschreven is.")
    signal_info = signal_df[signal_df['Vraag'] == signal]
    min_age = signal_info['Minimum leeftijd'].values[0]
    max_age = signal_info['Maximum leeftijd'].values[0]
    months = list(range(min_age, max_age + 1))
    result_df = pd.DataFrame({'Month': months, 'Count': [0] * len(months)})
    for _, row in data.iterrows():
        if signal in row['signals']: 
            age_months = row['age_months']
            if min_age <= age_months <= max_age:
                result_df.loc[result_df['Month'] == age_months, 'Count'] += 1
    return result_df

