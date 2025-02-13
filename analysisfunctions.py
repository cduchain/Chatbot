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

def extract_signal_in_question(text: str):
    words_in_quotes = re.findall(r"'([^']*)'", text)
    return words_in_quotes

#frequency questions
def analyze_freq_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relation_types = data['relation'].value_counts()
    total_entries = len(data)
    result_df = pd.DataFrame({
        "Relatie": relation_types.index,
        "Aantal keer gekozen": relation_types.values,
        "Percentage (%)": (relation_types.values / total_entries) * 100})
    result_df["Percentage (%)"] = result_df["Percentage (%)"].round(2)
    return result_df

def analyze_freq_signaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    result_df = signal_df[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
    result_df.columns = ['Signaal', 'Aantal keer zichtbaar', 'Percentage (%)']
    result_df["Percentage (%)"] = result_df["Percentage (%)"].round(2)
    result_df = result_df.sort_values(by='Percentage (%)', ascending=False)
    return result_df

def analyze_freq_alarmsignaal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
    result_df = alarm_signals[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
    result_df.columns = ['Alarmsignaal', 'Aantal keer zichtbaar', 'Percentage (%)']
    result_df["Percentage (%)"] = result_df["Percentage (%)"].round(2)
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
    age_types = data['age_range'].value_counts()
    total_entries = len(data)
    result_df = pd.DataFrame({
        'Leeftijdscategorie': age_types.index,
        'Aantal keer gekozen': age_types.values,
        'Percentage (%)': (age_types.values / total_entries) * 100
    })
    result_df["Percentage (%)"] = result_df["Percentage (%)"].round(2)
    return result_df

def analyze_freq_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    disorder_counts = {
        "Language impairment": sum('language disorder' in row['positive'] for _, row in data.iterrows()),
        "Motoric impairment": sum('motoric disorder' in row['positive'] for _, row in data.iterrows()),
        "Autism": sum('autism' in row['positive'] for _, row in data.iterrows())}
    result_df = pd.DataFrame(disorder_counts.items(), columns=['Impairment', 'Aantal keer als positief aangeduid'])
    return result_df

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
    max_index = values.index(max_stoornis)
    stoornissen = ['Language disorder', 'Motoric disorder', 'Autism']
    stoornis = stoornissen[max_index]
    analysis_result = ""
    analysis_result += f"Het ontwikkelingsprobleem die het meest werd aangegeven als positief is '{stoornis}'. Deze werd {max_stoornis} keer gekozen.\n"
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
    min_stoornis = min(values)
    min_index = values.index(min_stoornis)
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    stoornis = stoornissen[min_index]
    analysis_result = ""
    analysis_result += f"Het ontwikkelingsprobleem die het minst aangegeven werd als positief is '{stoornis}'. Deze werd {min_stoornis} keer gekozen.\n"
    return analysis_result

def top_most_signals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
    return top_signals_df[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def top_least_signals(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_number_from_question(specific_question)
    top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=True).head(number)
    return top_signals_df[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_below_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    filtered_signals = signal_df[signal_df['Percentage gekozen keren'] < number]
    if filtered_signals.empty:
        return f"Geen signalen komen voor in minder dan {number}% van de gevallen."
    return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]

def signals_below_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    filtered_signals = signal_df[signal_df['Percentage gekozen keren'] > number]
    if filtered_signals.empty:
        return f"Geen signalen komen voor in meer dan {number}% van de gevallen."
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
    if not result_df.empty:
        print(f"Het aantal kinderen met tenminste {threshold} alarmsignalen/alarmsignaal is:")
        print(f"{len(data)}")
    return result_df

#verbanden questions
def combo_most_relatie_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    stoornis = next((stoor for stoor in stoornissen if stoor in specific_question), None)
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    domein = next((dom for dom in domeinen if dom in specific_question), None)
    stoornis = next((stoor for stoor in stoornissen if stoor in specific_question), None)
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

def combo_most_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for age_range in sorted(data['age_range'].unique()):
        age_data = data[data['age_range'] == age_range]
        valid_signals = []
        for signals_list in age_data['signals']:
            for signal in signals_list:
                valid_signals.append(signal)
        signal_percentages = []
        for signal in set(valid_signals):
            signal_info = signal_df[signal_df['Vraag'] == signal]
            if not signal_info.empty:
                percentage = signal_info['Percentage gekozen keren'].values[0]
                signal_percentages.append((signal, percentage))
        if signal_percentages:
            top_signal = max(signal_percentages, key=lambda x: x[1])
            top_signal_name = top_signal[0]
            top_signal_percentage = top_signal[1]
            results.append((age_range, top_signal_name, top_signal_percentage))
            result_df = pd.DataFrame(results, columns=["Leeftijdscategorie", "Meest Gekozen Signaal", "Percentage"])
    return result_df

def combo_least_age(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for age_range in sorted(data['age_range'].unique()):
        age_data = data[data['age_range'] == age_range]
        valid_signals = []
        for signals_list in age_data['signals']:
            for signal in signals_list:
                valid_signals.append(signal)
        signal_percentages = []
        for signal in set(valid_signals):
            signal_info = signal_df[signal_df['Vraag'] == signal]
            if not signal_info.empty:
                percentage = signal_info['Percentage gekozen keren'].values[0]
                signal_percentages.append((signal, percentage))
        if signal_percentages:
            top_signal = min(signal_percentages, key=lambda x: x[1])
            top_signal_name = top_signal[0]
            top_signal_percentage = top_signal[1]
            results.append((age_range, top_signal_name, top_signal_percentage))
            result_df = pd.DataFrame(results, columns=["Leeftijdscategorie", "Minst Gekozen Signaal", "Percentage"])
    return result_df       

def combo_most_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for domein in signal_df['Domein'].unique():
        domain_signals = signal_df[signal_df['Domein'] == domein]
        sorted_signals = domain_signals.sort_values(by='Percentage gekozen keren', ascending=False)
        top_signal = sorted_signals.iloc[0]
        results.append((domein, top_signal['Vraag'], top_signal['Percentage gekozen keren']))
        result_df = pd.DataFrame(results, columns=["Domein", "Meest Gekozen Signaal", "Percentage"])
    return result_df

def combo_least_domein(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for domein in signal_df['Domein'].unique():
        domain_signals = signal_df[signal_df['Domein'] == domein]
        sorted_signals = domain_signals.sort_values(by='Percentage gekozen keren', ascending=True)
        top_signal = sorted_signals.iloc[0]
        results.append((domein, top_signal['Vraag'], top_signal['Percentage gekozen keren']))
        result_df = pd.DataFrame(results, columns=["Domein", "Minst Gekozen Signaal", "Percentage"])
    return result_df

def combo_most_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for relation in sorted(data['relation'].unique()):
        relation_data = data[data['relation'] == relation]
        valid_signals = []
        for signals_list in relation_data['signals']:
            for signal in signals_list:
                valid_signals.append(signal)
        signal_percentages = []
        for signal in valid_signals:
            signal_info = signal_df[signal_df['Vraag'] == signal]
            if not signal_info.empty:
                percentage = signal_info['Percentage gekozen keren'].values[0]
                signal_percentages.append((signal, percentage))
        if signal_percentages:
            top_signal = max(signal_percentages, key=lambda x: x[1])
            results.append((relation, top_signal[0], top_signal[1]))
            result_df = pd.DataFrame(results, columns=["Relatie", "Meest Gekozen Signaal", "Percentage"])
    return result_df 

def combo_least_relatie(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for relation in sorted(data['relation'].unique()):
        relation_data = data[data['relation'] == relation]
        valid_signals = []
        for signals_list in relation_data['signals']:
            for signal in signals_list:
                valid_signals.append(signal)
        signal_percentages = []
        for signal in valid_signals:
            signal_info = signal_df[signal_df['Vraag'] == signal]
            if not signal_info.empty:
                percentage = signal_info['Percentage gekozen keren'].values[0]
                signal_percentages.append((signal, percentage))
        if signal_percentages:
            top_signal = min(signal_percentages, key=lambda x: x[1])
            results.append((relation, top_signal[0], top_signal[1]))
            result_df = pd.DataFrame(results, columns=["Relatie", "Minst Gekozen Signaal", "Percentage"])
    return result_df 

def combo_most_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    processed_stoornissen = set()
    for index, row in data.iterrows():
        for stoornis in row['positive']:
            if stoornis in processed_stoornissen:
                continue  
            processed_stoornissen.add(stoornis)
            signals_in_row = row['signals']
            relevant_signals_df = signal_df[signal_df['Vraag'].isin(signals_in_row)]
            if not relevant_signals_df.empty:
                sorted_signals = relevant_signals_df.sort_values(by='Percentage gekozen keren', ascending=False)
                top_signal = sorted_signals.iloc[0]
                results.append((stoornis, top_signal['Vraag'], top_signal['Percentage gekozen keren']))
    result_df = pd.DataFrame(results, columns=["Ontwikkelingsprobleem", "Meest Gekozen Signaal", "Percentage"])
    return result_df

def combo_least_stoornis(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    processed_stoornissen = set()
    for index, row in data.iterrows():
        for stoornis in row['positive']:
            if stoornis in processed_stoornissen:
                continue  
            processed_stoornissen.add(stoornis)
            signals_in_row = row['signals']
            relevant_signals_df = signal_df[signal_df['Vraag'].isin(signals_in_row)]
            if not relevant_signals_df.empty:
                sorted_signals = relevant_signals_df.sort_values(by='Percentage gekozen keren', ascending=True)
                top_signal = sorted_signals.iloc[0]
                results.append((stoornis, top_signal['Vraag'], top_signal['Percentage gekozen keren']))
    result_df = pd.DataFrame(results, columns=["Ontwikkelingsprobleem", "Minst Gekozen Signaal", "Percentage"])
    return result_df

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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    stoornis = next((stoor for stoor in stoornissen if stoor in specific_question), None)
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    relatie = next((rel for rel in relaties if rel in specific_question), None)
    stoornis = next((stoor for stoor in stoornissen if stoor in specific_question), None)
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if not stoornis or not domein:
        return f"Er is geen ontwikkelingsprobleem of domein gevonden in de vraag."
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['domain'] == domein) & 
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    if not stoornis or not domein:
        return f"Er is geen ontwikkelingsprobleem of domein gevonden in de vraag."
    filtered_data = data[
        (data['positive'].apply(lambda x: stoornis in x)) & 
        (data['domain'] == domein) & 
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism' ]
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    number = extract_number_from_question(specific_question)
    if number is None:
        number = 5
    for dom in domeinen:
        if dom in specific_question:
            domein = dom
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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
    stoornissen = ['language disorder', 'motoric disorder', 'autism']
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    age_1, age_2 = extract_months_from_question(specific_question)
    number = extract_number_from_question(specific_question)
    for stoor in stoornissen:
        if stoor in specific_question:
            stoornis = stoor
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



