import pandas as pd
import re
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import scipy.stats as stats


#needed questions
def extract_subdomain_from_question(question):
    subdomeinen = ['grove motoriek', 'houdingsveranderingen', 
                   'fijne motoriek', 'zelfredzaamheid', 
                   'kwaliteit van bewegen', 'taalbegrip', 
                   'taalproductie', 'articulatie', 'communicatie', 
                   'interactie met anderen', 'imitatie', 'samen spelen', 
                   'stereotiep spel en stereotiepe bewegingen', 'spel',
                    'angst', 'andere opvallende gedragingen']
    gevonden_subdomeinen = []
    for subdom in subdomeinen:
        if subdom in question:
            gevonden_subdomeinen.append(subdom)
    if len(gevonden_subdomeinen)==1:
        return gevonden_subdomeinen[0]
    if len(gevonden_subdomeinen) > 1:
        return gevonden_subdomeinen
    else:
        return None
        
def extract_eerste_from_question(sentence: str) -> int:
    match = re.search(r'\b(eerste|laatste)\s+(\d+)', sentence, re.IGNORECASE)
    if match:
        return int(match.group(2))  # Geeft het getal terug na 'eerste' of 'laatste'
    return None  # Geeft None terug als er geen getal wordt gevonden

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
    match = re.search(r"(\d+)\s*(meest|meeste)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))   
    return None

def extract_percentage_from_question(question):
    match = re.search(r"(\d+)\s*(procent|%)", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_number_after_minimum(question):
    match = re.search(r"(ten minste|minstens|minimum)\s*(\d+)", question, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return None

def extract_threshold_from_question(question):
    match = re.search(r"\b(\d+)\s*keer\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_signal_in_question(question: str, signal_df: pd.DataFrame) -> str:
    for signal in signal_df['Vraag']:
        if signal in question:
            return signal
    return None

def extract_relatie_in_question(question: str) -> str:
    relaties = ['ouder', 'kinderbegeleider of verantwoordelijke kinderopvang', 'clb', 'leerkracht', 
                'psycholoog, orthopedagoog, logopedist, kinesitherapeut, ergotherapeut, thuisbegeleider',
                'huisarts', 'familie', 'zorgcoördinator of zorgleerkracht', 'ander', 'andere arts', 
                'kind&gezin', 'andere functie op school', 'pediater']
    gevonden_relaties = []
    for relatie in relaties:
        if relatie in question:
            gevonden_relaties.append(relatie)
    if len(gevonden_relaties) == 1:
       return gevonden_relaties[0]
    if len(gevonden_relaties) > 1:
        return list(gevonden_relaties)
    else: 
        return None

def extract_stoornis_in_question(question: str) -> str:
    mapping = {
        'taalontwikkelingsprobleem': 'language disorder',
        'taalontwikkelingsproblemen': 'language disorder',
        'taalontwikkelingsstoornis': 'language disorder',
        'taalontwikkelingstoornissen': 'language disorder',
        'motorische ontwikkelingproblemen': 'motoric disorder',
        'motorisch ontwikkelingsprobleem': 'motoric disorder',
        'motorische problemen': 'motoric disorder',
        'motorisch probleem': 'motoric disorder',
        'motorische ontwikkelingsstoornis': 'motoric disorder',
        'motorische ontwikkelingsstoornissen': 'motoric disorder',
        'autisme': 'autism'}
    found_disorders = []
    for key in mapping:
        if re.search(rf'\b{re.escape(key)}\b', question, re.IGNORECASE):
            found_disorders.append(mapping[key])
    if len(found_disorders)==1:
        return found_disorders[0]
    if found_disorders:
        return list(found_disorders)
    return None

def extract_domein_in_question(question: str) -> str:
    domeinen = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
    gevonden_domeinen = []
    for domein in domeinen:
        if domein in question:
            gevonden_domeinen.append(domein)
    if len(gevonden_domeinen) == 1:
        return gevonden_domeinen[0]
    if len(gevonden_domeinen) > 1:
        return gevonden_domeinen
    else:
        return None

def extract_number_after_multiple(specific_question: str):
    keywords = ['leeftijdscategorieën', 'leeftijdsgroepen', 'relaties', 'domeinen', 'stoornissen', 
                'ontwikkelingsproblemen', 'ontwikkelingsstoornissen', 'problemen']
    pattern = r"(\d+)\s*(" + '|'.join(keywords) + r")"
    match = re.search(pattern, specific_question, re.IGNORECASE)
    if match:
        return int(match.group(1)) 
    return None 

def extract_range_from_question(question):
    match = re.search(r"(leeftijdscategorie|leeftijdsgroep)\s*(\d+)", question, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return None

def extract_time_from_question(question: str):
    match_months = re.search(r'(\d+)\s*-\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match_months:
        return int(match_months.group(1)), int(match_months.group(2))
    match_months = re.search(r'(\d+)\s*en\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match_months:
        return int(match_months.group(1)), int(match_months.group(2))
    match_months = re.search(r'(\d+)\s*tot\s*(\d+)\s*maanden', question, re.IGNORECASE)
    if match_months:
        return int(match_months.group(1)), int(match_months.group(2))
    match_years = re.search(r'(\d+)\s*-\s*(\d+)\s*jaar', question, re.IGNORECASE)
    if match_years:
        return int(match_years.group(1)) * 12, int(match_years.group(2)) * 12
    match_years = re.search(r'(\d+)\s*en\s*(\d+)\s*jaar', question, re.IGNORECASE)
    if match_years:
        return int(match_years.group(1)) * 12, int(match_years.group(2)) * 12
    match_years = re.search(r'(\d+)\s*tot\s*(\d+)\s*jaar', question, re.IGNORECASE)
    if match_years:
        return int(match_years.group(1)) * 12, int(match_years.group(2)) * 12
    match_single_year = re.search(r"\b(\d+)\s*jaar\b", question, re.IGNORECASE)
    if match_single_year:
        number = int(match_single_year.group(1)) * 12
        return number, number
    match_single_month = re.search(r"\b(\d+)\s*maand\b", question, re.IGNORECASE)
    if match_single_month:
        number = int(match_single_month.group(1))
        return number, number
    match_single_months = re.search(r"\b(\d+)\s*maanden\b", question, re.IGNORECASE)
    if match_single_months:
        number = int(match_single_months.group(1))
        return number, number
    return None, None

def filter_data(data: pd.DataFrame, specific_question: str) -> pd.DataFrame:
    getal = extract_month_from_question(specific_question)
    if getal is not None:
        specific_question = specific_question.replace(str(getal), 'random')
    relatie = extract_relatie_in_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    stoornis = extract_stoornis_in_question(specific_question)
    leeftijd = extract_range_from_question(specific_question)
    leeftijd_1, leeftijd_2 = extract_time_from_question(specific_question)
    top = True
    if any(word in specific_question for word in ['meeste', 'meest', 'vaakste', 'vaakst', 'frequenst']):
        top = False
    filtered_data = data.copy()
    filtering_applied = False
    woorden = ['probleem','problemen','ontwikkelingsproblemen', 'ontwikkelingsprobleem', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis', 'stoornis', 'stoornissen']
    pattern = re.search(r"(onder|lager dan)\s(alle|elke)\s(" + "|".join(map(re.escape, woorden)) + r")", specific_question)
    woorden_2 = ['de cut-off', 'de grens','de drempel', 'het niveau', 'de cutoff']
    pattern_2 = re.search(r"(onder|lager dan)\s(" + "|".join(map(re.escape, woorden_2)) + r")", specific_question)
    if (pattern is not None) or (pattern_2 is not None):
        filtered_data = filtered_data[filtered_data['positive'].apply(lambda x: len(x) == 0)]
        filtering_applied = True
    if relatie is not None:
        filtered_data = filtered_data[filtered_data['relation'].isin([relatie])]
        filtering_applied = True
    if isinstance(domein, list):
        filtered_data = filtered_data[filtered_data['domains'].apply(lambda x: any(d in x for d in domein))]
        filtering_applied = True
    elif domein is not None:
        filtered_data = filtered_data[filtered_data['domains'].apply(lambda x: domein in x)]
        filtering_applied = True
    if isinstance(stoornis, list):
        filtered_data = filtered_data[filtered_data['positive'].apply(lambda x: any(s in x for s in stoornis))]
        filtering_applied = True
    elif stoornis is not None:
        filtered_data = filtered_data[filtered_data['positive'].apply(lambda x: stoornis in x)]
        filtering_applied = True
    if leeftijd is not None:
        filtered_data = filtered_data[(filtered_data['age_range'] == leeftijd)]
        filtering_applied = True
    if leeftijd_1 != leeftijd_2:
        filtered_data = filtered_data[(filtered_data['age_months'].between(leeftijd_1, leeftijd_2))]
        filtering_applied = True
    if leeftijd_1 is not None and leeftijd_1 == leeftijd_2:
        filtered_data = filtered_data[(filtered_data['age_months'] == leeftijd_1)]
        filtering_applied = True
    if not filtering_applied:
        filtered_data = data.copy()  
    return filtered_data

def filter_data_comp(data: pd.DataFrame, specific_question: str, element) -> pd.DataFrame:
    getal = extract_month_from_question(specific_question)
    if getal is not None:
        specific_question = specific_question.replace(str(getal), 'random')
    relatie = extract_relatie_in_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    stoornis = extract_stoornis_in_question(specific_question)
    leeftijd = extract_range_from_question(specific_question)
    leeftijd_1, leeftijd_2 = extract_time_from_question(specific_question)
    if element == 'relatie':
        relatie = None
    if element == 'leeftijd':
        leeftijd, leeftijd_1, leeftijd_2 = None, None, None
    top = True
    if any(word in specific_question for word in ['meeste', 'meest', 'vaakste', 'vaakst', 'frequenst']):
        top = False
    filtered_data = data.copy()
    filtering_applied = False
    if relatie is not None:
        filtered_data = filtered_data[filtered_data['relation'].isin([relatie])]
        filtering_applied = True
    if isinstance(domein, list):
        filtered_data = filtered_data[filtered_data['domains'].apply(lambda x: any(d in x for d in domein))]
        filtering_applied = True
    elif domein is not None:
        filtered_data = filtered_data[filtered_data['domains'].apply(lambda x: domein in x)]
        filtering_applied = True
    if isinstance(stoornis, list):
        filtered_data = filtered_data[filtered_data['positive'].apply(lambda x: any(s in x for s in stoornis))]
        filtering_applied = True
    elif stoornis is not None:
        filtered_data = filtered_data[filtered_data['positive'].apply(lambda x: stoornis in x)]
        filtering_applied = True
    if leeftijd is not None:
        filtered_data = filtered_data[(filtered_data['age_range'] == leeftijd)]
        filtering_applied = True
    if leeftijd_1 != leeftijd_2:
        filtered_data = filtered_data[(filtered_data['age_months'].between(leeftijd_1, leeftijd_2))]
        filtering_applied = True
    if leeftijd_1 is not None and leeftijd_1 == leeftijd_2:
        filtered_data = filtered_data[(filtered_data['age_months'] == leeftijd_1)]
        filtering_applied = True
    if not filtering_applied:
        filtered_data = data.copy()  
    return filtered_data

def extract_month_from_question(question):
    match = re.search(r'(sinds|laatste|voorbije|vorige|afgelopen)\s*(\d+)\s*maand', question, re.IGNORECASE)
    if match:
        number = int(match.group(2))
        return number
    return None

#echte functies
def analyze_freq(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    filtered_data = filter_data(data, specific_question)
    if not filtered_data.equals(data):
        aantal = len(filtered_data)
        analysis_results = f"De Wegwijzer werd {aantal} keer ingevuld voor die specificaties."
        return analysis_results
    if 'screener' in specific_question:
        total_responses = len(data)
        ja_count = data['screener_answers'].apply(lambda x: len(x) > 0).sum()
        ja_percentage = round((ja_count / total_responses) * 100, 2) if total_responses > 0 else 0
        analysis_results = ""
        analysis_results += f"De screener werd {ja_count} keer ingevuld. Dit telt voor {ja_percentage}%."
        return analysis_results
    if 'advies' in specific_question:
        total_responses = len(data)
        data['advies_opvolgen'] = data['advies_opvolgen'].str.strip().str.lower()
        ja_count = (data['advies_opvolgen'] == "ja").sum()
        ja_percentage = round((ja_count / total_responses) * 100, 2) if total_responses > 0 else 0
        analysis_results = ""
        analysis_results += f"Het advies werd {ja_count} keer gevolgd. Dit telt voor {ja_percentage}%."
        return analysis_results
    if any(word in specific_question for word in ['probleem','problemen','ontwikkelingsproblemen', 'ontwikkelingsprobleem', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis', 'stoornis', 'stoornissen']):
        disorder_counts = {
        "Taalontwikkelingsstoornis": sum('language disorder' in row['positive'] for _, row in data.iterrows()),
        "Motorische ontwikkelingsstoornis": sum('motoric disorder' in row['positive'] for _, row in data.iterrows()),
        "Autisme": sum('autism' in row['positive'] for _, row in data.iterrows())}
        result_df = pd.DataFrame(disorder_counts.items(), columns=['Ontwikkelingsprobleem', 'Aantal keer als positief aangeduid'])
        return result_df
    if any(word in specific_question for word in ['relatie','relaties']):
        relation_types = data['relation'].value_counts()
        total_entries = len(data)
        result_df = pd.DataFrame({
            "Relatie": relation_types.index,
            "Aantal keer gekozen": relation_types.values,
            "Percentage gekozen keren (%)": (relation_types.values / total_entries) * 100})
        result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
        return result_df 
    if re.search(r'\b(signaal|signalen)\b', specific_question):
        result_df = signal_df[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
        result_df.columns = ['Signaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren (%)']
        result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
        result_df = result_df.sort_values(by='Percentage gekozen keren (%)', ascending=False)
        return result_df
    if re.search(r'\b(domein|domeinen)\b', specific_question):
        domain_counts = {
        "Taal en/of communicatie": sum('taal en/of communicatie' in i for i in data['domains']),
        "Motoriek": sum('motoriek' in i for i in data['domains']),
        "Sociale vaardigheden": sum('sociale vaardigheden' in i for i in data['domains']),
        "Gedrag en spel": sum('gedrag en spel' in i for i in data['domains'])}
        result_df = pd.DataFrame(domain_counts.items(), columns=['Domein', 'Aantal keer gekozen'])
        return result_df
    if any(word in specific_question for word in ['alarmsignalen','alarmsignaal']):
        alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
        result_df = alarm_signals[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
        result_df.columns = ['Alarmsignaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren (%)']
        result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
        result_df = result_df.sort_values(by='Percentage gekozen keren (%)', ascending=False)
        return result_df
    if any(word in specific_question for word in ["leeftijdsgroep","leeftijdscategorie", 'leeftijdsgroepen', "leeftijdscategorieën"]):
        age_intervals = {
        0: "0-6 maanden",
        1: "6-12 maanden",
        2: "12-18 maanden",
        3: "18-24 maanden",
        4: "24-30 maanden",
        5: "30-36 maanden",
        6: "36-42 maanden",
        7: "42-48 maanden",
        8: "48-54 maanden",
        9: "54-60 maanden",
        10: "60-66 maanden",
        11: "66-72 maanden",
        12: "72-78 maanden",
        13: "78-84 maanden"}
        age_types = data['age_range'].value_counts()
        total_entries = len(data)
        result_df = pd.DataFrame({
            'Leeftijdscategorie': age_types.index,
            'Maand-interval': [age_intervals.get(age, "Onbekend") for age in age_types.index],
            'Aantal keer gekozen': age_types.values,
            'Percentage gekozen keren (%)': (age_types.values / total_entries) * 100})
        result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
        return result_df
    if any(word in specific_question for word in ['subdomeinen', 'subdomein']):
        subdomain_count = []
        for index, row in data.iterrows():
            signals = row['signals'] 
            if isinstance(signals, list):
                for signal in signals:
                    signal_row = signal_df[signal_df['Vraag'] == signal]
                    if not signal_row.empty:
                        subdomain = signal_row['Subdomein'].values[0]
                        subdomain_count.append({'Subdomein': subdomain})
        subdomain_count_df = pd.DataFrame(subdomain_count)
        subdomain_count_summary = subdomain_count_df.groupby('Subdomein').size().reset_index(name='Aantal keer gekozen')
        subdomain_count_summary = subdomain_count_summary.sort_values(by='Aantal keer gekozen', ascending=False)
        return subdomain_count_summary
    
def range_signal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    signal = extract_signal_in_question(specific_question, signal_df)
    if signal is None:
        signal_df_sorted = signal_df.sort_values(by=['Minimum leeftijd', 'Maximum leeftijd'])
        signal_df_sorted = signal_df_sorted.drop(columns=['Subdomein', 'Is een alarmsignaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren', 'Domein', 'Disorder'])
        return signal_df_sorted
    elif any(word in specific_question for word in ['maandelijks', 'per maand', 'iedere maand', 'per', 'per leeftijd']):
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
    else:
        signal_row = signal_df[signal_df['Vraag'] == signal]
        min_age = signal_row['Minimum leeftijd'].values[0]
        max_age = signal_row['Maximum leeftijd'].values[0]
        age_range = f"Leeftijdsverdeling voor het signaal '{signal}': {min_age} tot {max_age} maanden."
        return age_range

def most_element(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:
    relatie = extract_relatie_in_question(specific_question)
    age_1, age_2 = extract_time_from_question(specific_question)
    range = extract_range_from_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    signaal = extract_signal_in_question(specific_question, signal_df)
    stoornis = extract_stoornis_in_question(specific_question)
    number = extract_number_from_question(specific_question)
    total_rows = len(data)
    if number is None:
        number = extract_number_from_signal_question(specific_question)
    if any(word in specific_question for word in ['subdomein', 'subdomeinen']):
        domein = extract_domein_in_question(specific_question)
        filtered_data = filter_data(data, specific_question)
        filtered_signal_df = signal_df[signal_df['Domein'] == domein]
        subdomeinen = filtered_signal_df['Subdomein'].unique().tolist()
        result_df = pd.DataFrame({'Subdomein': subdomeinen, 'Count': [0] * len(subdomeinen)})
        for subdomein in subdomeinen:
            signalen = filtered_signal_df[filtered_signal_df['Subdomein'] == subdomein]['Vraag'].tolist()
            total_count = sum(filtered_data['signals'].apply(lambda signals: sum(signal in signals for signal in signalen)))
            result_df.loc[result_df['Subdomein'] == subdomein, 'Count'] += total_count
        return result_df
    elif any(word in specific_question for word in ['relatie', 'relaties']) and relatie is None:
        relation_types = data['relation'].value_counts()
        max_value = relation_types.max()
        max_relationship = relation_types.idxmax()
        amount_entries = len(data)
        percentage = max_value/amount_entries * 100
        analysis_result = ""
        analysis_result += f"De meest gekozen relatie is '{max_relationship}' die {max_value} keer werd gekozen. Dit telt voor een percentage van {percentage}%.\n"
        return analysis_result
    elif any(word in specific_question for word in ['leeftijdsgroepen', 'leeftijdsgroep', 'leeftijdscategorie', 'leeftijdscategorieën']) and age_1 is None:
        leeftijd_types = data['age_range'].value_counts()
        max_value = leeftijd_types.max()
        max_leeftijd = leeftijd_types.idxmax()
        amount_entries = len(data)
        percentage = max_value/amount_entries * 100
        analysis_result = ""
        analysis_result += f"De meest gekozen leeftijdscategorie is'{max_leeftijd}' die {max_value} keer werd gekozen. Dit telt voor een percentage van {percentage}%.\n"
        return analysis_result
    elif any(word in specific_question for word in ['domeinen', 'domein']) and domein is None:
        domain_counts = {
            "Taal en/of communicatie": sum('taal en/of communicatie' in i for i in data['domains']),
            "Motoriek": sum('motoriek' in i for i in data['domains']),
            "Sociale vaardigheden": sum('sociale vaardigheden' in i for i in data['domains']),
            "Gedrag en spel": sum('gedrag en spel' in i for i in data['domains'])}
        result_df = pd.DataFrame(domain_counts.items(), columns=['Domein', 'Aantal keer gekozen'])
        max_domein = max(domain_counts, key=domain_counts.get)
        max_count = domain_counts[max_domein]
        amount_entries = len(data)
        percentage = max_count/amount_entries * 100
        analysis_result = ""
        analysis_result += f"Het meest gekozen domein is '{max_domein}' met {max_count} keer. Dit telt voor een percentage van {percentage}.\n"
        return analysis_result
    elif re.search(r'\b(signaal)\b', specific_question) and signaal is None:
        max_percentage_row = signal_df.loc[signal_df['Percentage gekozen keren'].idxmax()]
        vraag = max_percentage_row['Vraag']
        aantal_keer_zichtbaar = max_percentage_row['Aantal keer zichtbaar']
        percentage_gekozen = max_percentage_row['Percentage gekozen keren']
        analysis_result = f"Het signaal '{vraag}' is het meest gekozen signaal met een percentage van {percentage_gekozen}%. Dit signaal was {aantal_keer_zichtbaar} keer zichtbaar."
        return analysis_result
    elif re.search(r'\b(signalen)\b', specific_question) and signaal is None:
        if number is None:
            number = 5
        top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
        return top_signals_df.rename(columns={'Vraag': 'Signaal'})[['Signaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]
    elif re.search(r'\b(alarmsignaal)\b', specific_question) and signaal is None:
        alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
        result_df = alarm_signals[['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].copy()
        result_df.columns = ['Alarmsignaal', 'Aantal keer zichtbaar', 'Percentage gekozen keren (%)']
        result_df["Percentage gekozen keren (%)"] = result_df["Percentage gekozen keren (%)"].round(2)
        result_df = result_df.sort_values(by='Percentage gekozen keren (%)', ascending=False)
        top_signal = result_df.iloc[0]
        vraag = top_signal['Alarmsignaal']
        aantal_keer_zichtbaar = top_signal['Aantal keer zichtbaar']
        percentage_gekozen = top_signal['Percentage gekozen keren (%)']
        analysis_result = f"Het alarmsignaal '{vraag}' is het meest gekozen signaal met een percentage van {percentage_gekozen}%. Dit signaal was {aantal_keer_zichtbaar} keer zichtbaar."
        return analysis_result 
    elif re.search(r'\b(alarmsignalen)\b', specific_question) and signaal is None:
        if number is None:
            number = 5 
        alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
        alarm_signals = alarm_signals.reset_index(drop=True)
        top_signals_df = alarm_signals.sort_values(by='Percentage gekozen keren', ascending=False).head(number)
        return top_signals_df.rename(columns={'Vraag': 'Alarmsignaal'})[['Alarmsignaal','Percentage gekozen keren', 'Aantal keer zichtbaar']]
    elif re.search(r'(?i)(?<!motorische\s)(?<!motoriek\s)\b(stoornis|stoornissen|ontwikkelingsprobleem|ontwikkelingsproblemen|ontwikkelingsstoornis|ontwikkelingsstoornissen)\b', specific_question, re.IGNORECASE) and stoornis is None:
        if any(word in specific_question for word in ['onder', 'lager', 'beneden', 'negatief', 'negatieve']):
            total_rows = len(data)
            value_counts = (data['positive'].apply(lambda x: len(x) == 0)).sum()
            percentages = value_counts / total_rows * 100 if total_rows > 0 else 0
            analysis_result = f"Het aantal kinderen dat voor alles onder de cut-off scoort is '{value_counts}'. Dit telt voor een percentage van {percentages}% van de kinderen.\n"          
            return analysis_result
        else:
            total_rows = len(data)
            disorders = {
                'Taalontwikkelingsprobleem': 'language disorder',
                'Motorisch ontwikkelingsprobleem': 'motoric disorder',
                'Autisme': 'autism'}
            value_counts = {key: data['positive'].apply(lambda x: disorder in x).sum() for key, disorder in disorders.items()}
            percentages = {key: (count / total_rows) * 100 if total_rows > 0 else 0 for key, count in value_counts.items()}
            stoornis, stoornis_value = max(value_counts.items(), key=lambda x: x[1])
            stoornis_percentage = percentages[stoornis]
            analysis_result = f"Het ontwikkelingsprobleem dat het meest werd aangegeven als positief is '{stoornis}'. " \
                        f"Deze werd {stoornis_value} keer gekozen, wat neerkomt op {stoornis_percentage:.2f}% van de gevallen.\n"
            return analysis_result
    elif relatie is not None:
        relation_correct = len(data[data['relation']==relatie])
        total_entries = len(data)
        percentage = relation_correct/total_entries * 100
        analysis_result = ""
        analysis_result += f"De relatie {relatie} werd {relation_correct} keer gekozen. Dit is een percentage van {percentage}.\n"
        return analysis_result
    elif range is not None:
        age_correct = len(data[data['age_range'] == range])
        total_entries = len(data)
        percentage = age_correct/total_entries*100
        analysis_result = ""
        analysis_result += f"De leeftijdsgroep {range} werd {age_correct} keer gekozen. Dit is een percentage van {percentage}.\n"
        return analysis_result
    elif (age_1 is not None) and (age_1 == age_2):
        mask = (data['age_months'] == age_1)
        aantal_keer = mask.sum()  
        totaal = len(data)  
        percentage = (aantal_keer / totaal) * 100 
        analysis_result = ""
        analysis_result = f"De data bevat {aantal_keer} kinderen van {age_1} maand. Dit is een percentage van {percentage:.2f}%.\n"
        return analysis_result
    elif age_1 != age_2:
        mask = (data['age_months'] >= age_1) & (data['age_months'] <= age_2)
        aantal_keer = mask.sum()  
        totaal = len(data)  
        percentage = (aantal_keer / totaal) * 100 
        analysis_result = ""
        analysis_result = f"De data bevat {aantal_keer} kinderen tussen de {age_1} en {age_2} maanden. Dit is een percentage van {percentage:.2f}%.\n"
        return analysis_result
    elif domein is not None:
        if isinstance(domein, list):
            total_rows = len(data)
            both_domein_value = data['domains'].apply(lambda x: all(s in x for s in domein)).sum()
            both_domein_percentage = (both_domein_value / total_rows) * 100 if total_rows > 0 else 0
            analysis_result = f"{', '.join(domein)} komen samen {both_domein_value} keer voor, wat overeenkomt met {both_domein_percentage:.2f}%."
            return analysis_result
        else:    
            domain_counts = {
                "Taal en/of communicatie": sum('taal en/of communicatie' in i for i in data['domains']),
                "Motoriek": sum('motoriek' in i for i in data['domains']),
                "Sociale vaardigheden": sum('sociale vaardigheden' in i for i in data['domains']),
                "Gedrag en spel": sum('gedrag en spel' in i for i in data['domains'])}
            result_df = pd.DataFrame(domain_counts.items(), columns=['Domein', 'Aantal keer gekozen'])
            filtered_row = result_df[result_df['Domein'].str.contains(domein, case=False)]
            aantal_keer = filtered_row['Aantal keer gekozen'].values[0]
            amount_entries = len(data)
            percentage = aantal_keer/amount_entries * 100
            analysis_result = ""
            analysis_result += f"Het domein {domein} werd {aantal_keer} keer gekozen. Dit telt voor een percentage van {percentage}.\n"    
            return analysis_result
    elif signaal is not None:
        aantal_keer = signal_df.loc[signal_df['Vraag'] == signaal, 'Aantal keer zichtbaar'].values
        percentage = signal_df.loc[signal_df['Vraag'] == signaal, 'Percentage gekozen keren'].values
        analysis_result = f"Het signaal {signaal} was {aantal_keer} keer zichtbaar. Het werd in {percentage}% van de keren gekozen."
        return analysis_result
    elif stoornis is not None:
        if isinstance(stoornis, list):
            total_rows = len(data)
            both_stoornissen_value = data['positive'].apply(lambda x: all(s in x for s in stoornis)).sum()
            both_stoornissen_percentage = (both_stoornissen_value / total_rows) * 100 if total_rows > 0 else 0
            analysis_result = f"{', '.join(stoornis)} komen samen {both_stoornissen_value} keer voor, wat overeenkomt met {both_stoornissen_percentage:.2f}%."
            return analysis_result
        else:
            stoornis_value = data['positive'].apply(lambda x: stoornis in x).sum() 
            stoornis_percentage = (stoornis_value / total_rows) * 100 if total_rows > 0 else 0
            analysis_result = f"{stoornis} komt {stoornis_value} keer voor, wat overeenkomt met {stoornis_percentage:.2f}%."
            return analysis_result
        
def least_element(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) ->str:    
    total_row = len(data)
    number = extract_number_from_question(specific_question)
    if number is None:
        number = extract_number_from_signal_question(specific_question)
    if number is None:
        number  = 5
    if any(word in specific_question for word in ['relatie', 'relaties']):
        relation_types = data['relation'].value_counts()
        min_value = relation_types.min()  
        min_relationship = relation_types.idxmin() 
        percentage = min_value/total_row*100 
        analysis_result = ""
        analysis_result += f"De minst gekozen relatie is '{min_relationship}' die {min_value} keer werd gekozen. Dit telt voor {percentage}%.\n"
        return analysis_result
    if any(word in specific_question for word in ['leeftijdsgroepen', 'leeftijdsgroep', 'leeftijdscategorie', 'leeftijdscategorieën']):
        leeftijd_types = data['age_range'].value_counts()
        min_value = leeftijd_types.min()  
        min_leeftijd = leeftijd_types.idxmin()
        percentage = min_value/total_row*100 
        analysis_result = ""
        analysis_result += f"De minst gekozen leeftijdscategorie is '{min_leeftijd}' die {min_value} keer werd gekozen. Dit telt voor {percentage}%.\n"
        return analysis_result
    if re.search(r'\b(leeftijd)\b', specific_question):
        age_counts = data['age_months'].value_counts()
        min_age = age_counts.idxmin()
        min_count = age_counts.min()
        percentage = min_count/total_row*100
        return f"De leeftijd van {min_age} maanden komt het minst voor, namelijk {min_count} keer. Dit telt voor {percentage}%.\n"
    if any(word in specific_question for word in ['domeinen', 'domein']):
        domain_names = ['Taal en/of communicatie', 'Motoriek', 'Sociale vaardigheden', 'Gedrag en spel']
        domain_keywords = ['taal en/of communicatie', 'motoriek', 'sociale vaardigheden', 'gedrag en spel']
        domain_counts = Counter({name: sum(keyword in domains for domains in data['domains']) 
                                for name, keyword in zip(domain_names, domain_keywords)})
        min_domain, min_value = min(domain_counts.items(), key=lambda x: x[1])
        percentage = min_value/total_row*100
        analysis_result = f"Het minst gekozen domein is '{min_domain}' dat {min_value} keer werd gekozen. Dit telt voor {percentage}%.\n"
        return analysis_result
    if re.search(r'\b(signaal)\b', specific_question):
        min_percentage_row = signal_df.loc[signal_df['Percentage gekozen keren'].idxmin()]
        vraag = min_percentage_row['Vraag']
        aantal_keer_zichtbaar = min_percentage_row['Aantal keer zichtbaar']
        percentage_gekozen = min_percentage_row['Percentage gekozen keren']
        analysis_result = f"Het signaal '{vraag}' is het minst gekozen signaal met een percentage van {percentage_gekozen}%. Dit signaal was {aantal_keer_zichtbaar} keer zichtbaar."
        return analysis_result
    if re.search(r'\b(alarmsignaal)\b', specific_question):
        alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
        alarm_signals = alarm_signals.reset_index(drop=True)
        max_percentage_row = alarm_signals.loc[alarm_signals['Percentage gekozen keren'].idxmin()]
        vraag = max_percentage_row['Vraag']
        aantal_keer_zichtbaar = max_percentage_row['Aantal keer zichtbaar']
        percentage_gekozen = max_percentage_row['Percentage gekozen keren']
        analysis_result = (
            f"Het alarmsignaal '{vraag}' is het minst gekozen alarmsignaal met een percentage van {percentage_gekozen}%. "
            f"Dit alarmsignaal was {aantal_keer_zichtbaar} keer zichtbaar.")
        return analysis_result
    if re.search(r'\b(signalen)\b', specific_question):
        top_signals_df = signal_df.sort_values(by='Percentage gekozen keren', ascending=True).head(number)
        return top_signals_df.rename(columns={'Vraag': 'Signaal'})[['Signaal','Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']].sort_values(
            by='Percentage gekozen keren', ascending=False)
    if re.search(r'\b(alarmsignalen)\b', specific_question):
        alarm_signals = signal_df[signal_df['Is een alarmsignaal'] == True]
        alarm_signals = alarm_signals.reset_index(drop=True)
        top_signals_df = alarm_signals.sort_values(by='Percentage gekozen keren', ascending=True).head(number)
        return top_signals_df.rename(columns={'Vraag': 'Alarmsignaal'})[['Alarmsignaal', 'Is een alarmsignaal', 'Percentage gekozen keren', 'Aantal keer zichtbaar']]
    if any(word in specific_question for word in ['stoornis', 'stoornissen', 'ontwikkelingsprobleem', 'ontwikkelingsproblemen', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis']):
        disorders = {
        'Taalontwikkelingsprobleem': 'language disorder',
        'Motorisch ontwikkelingsprobleem': 'motoric disorder',
        'Autisme': 'autism' }
        disorder_counts = {name: data['positive'].apply(lambda x: disorder in x).sum() for name, disorder in disorders.items()}
        stoornis, stoornis_value = min(disorder_counts.items(), key=lambda x: x[1])
        percentage = stoornis_value/total_row*100
        return f"Het ontwikkelingsprobleem die het minst werd gescoord als positief is '{stoornis}'. Deze werd {stoornis_value} keer als positief gescoord. Dit telt voor {percentage}%.\n"        

def signals_percentage(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_percentage_from_question(specific_question)
    if number is None:
        number = 5
    top = 1
    relatie = extract_relatie_in_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    stoornis = extract_stoornis_in_question(specific_question)
    filtered_data = filter_data(data, specific_question)
    all_signals_in_data = set(signal for signals_list in filtered_data['signals'] for signal in signals_list)
    filtered_signals = signal_df[signal_df['Vraag'].isin(all_signals_in_data)]
    if any(word in specific_question for word in ['alarmsignalen', 'alarmsignaal']):
        filtered_signals = filtered_signals[filtered_signals['Is een alarmsignaal'] == True]
    if re.search(r'\b(onder|minder|lager|niet meer|niet boven|niet hoger)\b', specific_question):
        top = 1
    if re.search(r'\b(boven|meer|hoger|niet minder|niet onder|niet lager)\b',specific_question):
        top= 2
    if any(word in specific_question for word in ['relatie', 'relaties']) and relatie is None:
        signal_percentages = []
        for rel in filtered_data['relation'].unique():                    
            filtered_data_relation = filtered_data[filtered_data['relation'] == rel]
            for _, signal_row in filtered_signals.iterrows():
                signal = signal_row['Vraag']
                signal_domain = signal_row['Domein']
                min_age = signal_row['Minimum leeftijd']                    
                max_age = signal_row['Maximum leeftijd']
                filtered_data_domain = filtered_data_relation[filtered_data_relation['domains'].apply(lambda x: signal_domain in x)]
                filtered_data_age = filtered_data_domain[(filtered_data_domain['age_months'] >= min_age) & 
                                                          (filtered_data_domain['age_months'] <= max_age)]
                signal_count = sum(signal in signals for signals in filtered_data_age['signals'])
                signal_percentage = (signal_count / len(filtered_data_age)) * 100 if len(filtered_data_age) > 0 else 0
                if top == 1:
                    if signal_percentage < number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
                if top == 2:
                    if signal_percentage > number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
        percentage_df = pd.DataFrame(signal_percentages, columns=['Vraag', 'Relatie', 'Aantal keer zichtbaar', 'Percentage gekozen keren'])
        return percentage_df[['Vraag', 'Relatie', 'Aantal keer zichtbaar', 'Percentage gekozen keren']].sort_values(
                            by='Percentage gekozen keren', ascending=False)    
    if any(word in specific_question for word in ['domein', 'domeinen']) and domein is None:
        signal_percentages = []
        for dom in filtered_signals['Domein'].unique():
            filtered_data_domain = filtered_data[filtered_data['domains'].apply(lambda x: dom in x)]
            for _, signal_row in filtered_signals.iterrows():
                signal = signal_row['Vraag']
                signal_domain = signal_row['Domein']
                min_age = signal_row['Minimum leeftijd']
                max_age = signal_row['Maximum leeftijd']
                filtered_data_age = filtered_data_domain[(filtered_data_domain['age_months'] >= min_age) & 
                                                          (filtered_data_domain['age_months'] <= max_age)]
                signal_count = sum(signal in signals for signals in filtered_data_age['signals'])
                signal_percentage = (signal_count / len(filtered_data_age)) * 100 if len(filtered_data_age) > 0 else 0
                if top == 1:
                    if signal_percentage < number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
                    percentage_df = pd.DataFrame(signal_percentages, columns=['Vraag', 'Relatie', 'Aantal keer zichtbaar', 'Percentage gekozen keren'])
                    return percentage_df[['Vraag', 'Relatie', 'Percentage gekozen keren']].sort_values(
                    by='Percentage gekozen keren', ascending=False)
                if top == 2:
                    if signal_percentage > number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
        percentage_df = pd.DataFrame(signal_percentages, columns=['Vraag', 'Relatie','Aantal keer zichtbaar', 'Percentage gekozen keren'])
        return percentage_df[['Vraag', 'Relatie', 'Percentage gekozen keren']].sort_values(
                            by='Percentage gekozen keren', ascending=False)
    if any(word in specific_question for word in ['stoornis', 'stoornissen', 'ontwikkelingsprobleem', 'ontwikkelingsproblemen', 'problemen', 'probleem']) and stoornis is None:
        signal_percentages = []
        for stoor in filtered_signals['Disorder'].unique():
            filtered_data_stoor = filtered_data[filtered_data['positive'].apply(lambda x: stoor in x)]
            for _, signal_row in filtered_signals.iterrows():
                signal = signal_row['Vraag']
                signal_domain = signal_row['Domein']
                min_age = signal_row['Minimum leeftijd']
                max_age = signal_row['Maximum leeftijd']
                filtered_data_age = filtered_data_stoor[(filtered_data_stoor['age_months'] >= min_age) & 
                                                          (filtered_data_stoor['age_months'] <= max_age)]
                signal_count = sum(signal in signals for signals in filtered_data_age['signals'])
                signal_percentage = (signal_count / len(filtered_data_age)) * 100 if len(filtered_data_age) > 0 else 0
                if top == 1:
                    if signal_percentage < number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
                if top == 2:
                    if signal_percentage > number:
                        signal_percentages.append((signal, rel, signal_count, signal_percentage))
        percentage_df = pd.DataFrame(signal_percentages, columns=['Vraag', 'Relatie','Aantal keer zichtbaar', 'Percentage gekozen keren'])
        return percentage_df[['Vraag', 'Relatie', 'Percentage gekozen keren']].sort_values(
                            by='Percentage gekozen keren', ascending=False)
    else:
        signal_percentages = []
        for _, signal_row in filtered_signals.iterrows():
            signal = signal_row['Vraag']
            signal_count = sum(signal in signals for signals in filtered_data['signals'])
            signal_percentage = (signal_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
            signal_percentages.append((signal, signal_count, signal_percentage))
            percentage_df = pd.DataFrame(signal_percentages, columns=['Vraag', 'Aantal keer zichtbaar', 'Percentage gekozen keren'])
            if top == 1:
                filtered_signals = percentage_df[percentage_df['Percentage gekozen keren'] < number]
            if top == 2:
                filtered_signals = percentage_df[percentage_df['Percentage gekozen keren'] > number]
        if filtered_signals.empty:
            if top == 1:
                return f"Geen signalen komen voor in minder dan {number}% van de gevallen."
            else:
                return f"Geen signalen komen voor in meer dan {number}% van de gevallen."
        return filtered_signals[['Vraag', 'Percentage gekozen keren', 'Aantal keer zichtbaar']].sort_values(
                    by='Percentage gekozen keren', ascending=False)

def signal_in_range(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    number = extract_eerste_from_question(specific_question)
    keer = extract_threshold_from_question(specific_question)
    if keer is None:
        keer = 3
    results = []
    if 'eerste' in specific_question:
        if re.search(r'\b(signaal|signalen)\b', specific_question):
            for _, row in signal_df.iterrows():
                signal = row['Vraag']
                min_age = row['Minimum leeftijd']
                max_age = row['Maximum leeftijd']
                months = list(range(min_age, max_age + 1))
                result_df = pd.DataFrame({'Month': months, 'Count': [0] * len(months)})
                for _, data_row in data.iterrows():
                    if signal in data_row['signals']: 
                        age_months = data_row['age_months']
                        if min_age <= age_months <= max_age:
                            result_df.loc[result_df['Month'] == age_months, 'Count'] += 1
                first_part = result_df['Count'].iloc[:number]
                remaining_part = result_df['Count'].iloc[number:]
                if all(first_part > remaining_part.max() * keer):
                    if signal not in results: 
                        results.append(signal)
            results_df = pd.DataFrame(results, columns=['Signal']) if results else pd.DataFrame()
            return results_df 
        if any(word in specific_question for word in ['alarmsignalen', 'alarmsignaal']):
            for _, row in signal_df.iterrows():
                if row['Is een alarmsignaal'] == True:
                    signal = row['Vraag']
                    min_age = row['Minimum leeftijd']
                    max_age = row['Maximum leeftijd']
                    months = list(range(min_age, max_age + 1))
                    result_df = pd.DataFrame({'Month': months, 'Count': [0] * len(months)})
                    for _, data_row in data.iterrows():
                        if signal in data_row['signals']: 
                            age_months = data_row['age_months']
                            if min_age <= age_months <= max_age:
                                result_df.loc[result_df['Month'] == age_months, 'Count'] += 1
                    first_part = result_df['Count'].iloc[:number]
                    remaining_part = result_df['Count'].iloc[number:]
                    if all(first_part > remaining_part.max() * keer):
                        if signal not in results: 
                            results.append(signal)
            results_df = pd.DataFrame(results, columns=['Signal']) if results else pd.DataFrame()
            return results_df  
    if 'laatste' in specific_question:
        if re.search(r'\b(signaal|signalen)\b', specific_question):
            for _, row in signal_df.iterrows():
                signal = row['Vraag']
                min_age = row['Minimum leeftijd']
                max_age = row['Maximum leeftijd']
                months = list(range(min_age, max_age + 1))
                result_df = pd.DataFrame({'Month': months, 'Count': [0] * len(months)})
                for _, data_row in data.iterrows():
                    if signal in data_row['signals']: 
                        age_months = data_row['age_months']
                        if min_age <= age_months <= max_age:
                            result_df.loc[result_df['Month'] == age_months, 'Count'] += 1
                first_part = result_df['Count'].iloc[number:]
                remaining_part = result_df['Count'].iloc[:number]
                if all(first_part > remaining_part.max() * keer):
                    if signal not in results: 
                        results.append(signal)
            results_df = pd.DataFrame(results, columns=['Signal']) if results else pd.DataFrame()
            return results_df 
        if any(word in specific_question for word in ['alarmsignalen', 'alarmsignaal']):
            for _, row in signal_df.iterrows():
                if row['Is een alarmsignaal'] == True:
                    signal = row['Vraag']
                    min_age = row['Minimum leeftijd']
                    max_age = row['Maximum leeftijd']
                    months = list(range(min_age, max_age + 1))
                    result_df = pd.DataFrame({'Month': months, 'Count': [0] * len(months)})
                    for _, data_row in data.iterrows():
                        if signal in data_row['signals']: 
                            age_months = data_row['age_months']
                            if min_age <= age_months <= max_age:
                                result_df.loc[result_df['Month'] == age_months, 'Count'] += 1
                    first_part = result_df['Count'].iloc[number:]
                    remaining_part = result_df['Count'].iloc[:number]
                    if all(first_part > remaining_part.max() * keer):
                        if signal not in results: 
                            results.append(signal)
            results_df = pd.DataFrame(results, columns=['Signal']) if results else pd.DataFrame()
            return results_df
    return pd.DataFrame()
              
def combo_signal(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    threshold = extract_percentage_from_question(specific_question)
    data = filter_data(data, specific_question)
    if any(word in specific_question for word in ['alleen', 'individueel', 'afgezonderd']):
        signal_counts = {}
        total_rows = len(data)
        for signals in data['signals']:
            if isinstance(signals, list) and len(signals) == 1:
                signal = signals[0]
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        signal_single_df = pd.DataFrame(
            [(signal, count / total_rows * 100) for signal, count in signal_counts.items()],
            columns=['Signaal', 'Percentage (in hoeveel procent van de totale data komt dit signaal alleen voor)'])
        signal_single_df_sorted = signal_single_df.sort_values(by='Percentage (in hoeveel procent van de totale data komt dit signaal alleen voor)', ascending=False)
        return signal_single_df_sorted
    if threshold is None:
        signal_counts = {}
        total_rows = len(data)
        for signals in data['signals']:
            if isinstance(signals, list):
                for combo in combinations(sorted(signals), 2):  
                    signal_counts[combo] = signal_counts.get(combo, 0) + 1
        signal_pairs_df = pd.DataFrame(
            [(s1, s2, count / total_rows*100) for (s1, s2), count in signal_counts.items()],
            columns=['Signaal 1', 'Signaal 2', 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)'])
        if any(word in specific_question for word in ['vaakst', 'meest', 'meeste', 'vaakste']):
            signal_pairs_df_sorted = signal_pairs_df.sort_values(by= 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)', ascending=False)
            return signal_pairs_df_sorted.head(5)
        if any(word in specific_question for word in ['minste', 'minst']):
            signal_pairs_df_sorted = signal_pairs_df.sort_values(by= 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)', ascending=True)
            return signal_pairs_df_sorted.head(5)
    else:        
        signal_counts = {}
        total_rows = len(data)
        for signals in data['signals']:
            if isinstance(signals, list):
                for combo in combinations(sorted(signals), 2):  
                    signal_counts[combo] = signal_counts.get(combo, 0) + 1
        signal_pairs_df = pd.DataFrame(
            [(s1, s2, count / total_rows*100) for (s1, s2), count in signal_counts.items()],
            columns=['Signaal 1', 'Signaal 2', 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)'])
        if any(word in specific_question for word in ['vaker','meer']):
            return signal_pairs_df[signal_pairs_df['Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)'] > threshold].sort_values(by= 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)', ascending=False)
        else: 
            return signal_pairs_df[signal_pairs_df['Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)'] < threshold].sort_values(by= 'Percentage (in hoeveel procent van de totale data komen deze signalen samen voor)', ascending=False) 

def combo_function(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(data)
    filtered_data = filter_data(data, specific_question)
    top = 0
    if any(word in specific_question for word in ['meeste', 'meest', 'vaakste', 'vaakst', 'frequenst']):
        top = 2
    else: 
        top = 1
    if any(word in specific_question for word in ['signalen', 'signaal']):
        domein = extract_domein_in_question(specific_question)
        stoornis = extract_stoornis_in_question(specific_question)
        if filtered_data.equals(data):
            if re.search(r'\b(domein)\b', specific_question):
                signal_df["Impact"] = signal_df["Aantal keer zichtbaar"] * (signal_df["Percentage gekozen keren"] / 100)
                resultaat = signal_df.groupby("Domein")["Impact"].sum().reset_index()
                totaal_impact = resultaat["Impact"].sum()
                resultaat["Percentage"] = (resultaat["Impact"] / totaal_impact) * 100
                resultaat = resultaat.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
                resultaat = resultaat.drop(columns=['Impact'])
                return resultaat
            if 'subdomein' in specific_question:
                signal_df["Impact"] = signal_df["Aantal keer zichtbaar"] * (signal_df["Percentage gekozen keren"]) / 100
                resultaat = signal_df.groupby("Subdomein")["Impact"].sum().reset_index()
                totaal_impact = resultaat["Impact"].sum()
                resultaat["Percentage"] = (resultaat["Impact"] / totaal_impact) * 100
                resultaat = resultaat.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
                resultaat = resultaat.drop(columns=['Impact'])
                return resultaat
        else:
            alarm = 0
            number = extract_number_from_signal_question(specific_question)
            if 'signaal' in specific_question:
                number = 1
            elif number is None:
                number = 5
            if any(word in specific_question for word in ['alarmsignalen', 'alarmsignaal']):
                number = extract_number_from_alarm_question(specific_question)
                alarm = 1
            signal_counts = {}
            for _,row in filtered_data.iterrows():
                for signal in row['signals']:
                    if alarm == 1:
                        if signal_df.loc[signal_df['Vraag'] == signal, 'Is een alarmsignaal'].values[0]:
                            if signal in signal_counts:
                                signal_counts[signal] += 1
                            else:
                                signal_counts[signal] = 1
                    else:
                        if signal in signal_counts:
                            signal_counts[signal] += 1 
                        else:
                            signal_counts[signal] = 1 
            signal_count_df = pd.DataFrame(list(signal_counts.items()), columns=['Signaal', 'Aantal keer zichtbaar'])
            if domein is not None and isinstance(domein, str):
                domein = [domein] 
                for signal in signal_count_df['Signaal']:
                    signaal_domein = signal_df.loc[signal_df['Vraag'] == signal, 'Domein'].values
                    if len(signaal_domein) == 0 or signaal_domein[0] not in domein:
                        signal_count_df = signal_count_df[signal_count_df['Signaal'] != signal]
            if stoornis is not None:
                stoornis = [stoornis] 
                for signal in signal_count_df['Signaal']:
                    signaal_stoornis = signal_df.loc[signal_df['Vraag'] == signal, 'Disorder'].values
                    if len(signaal_stoornis) == 0 or signaal_stoornis[0] not in domein:
                        signal_count_df = signal_count_df[signal_count_df['Signaal'] != signal]
            signal_count_df['Percentage'] = (signal_count_df['Aantal keer zichtbaar'] / total_rows) * 100
            if top == 1:
                signal_count_df = signal_count_df.sort_values(by='Aantal keer zichtbaar', ascending=True)
            elif top == 2:
                signal_count_df = signal_count_df.sort_values(by='Aantal keer zichtbaar', ascending=False)
            return signal_count_df.head(number)
    else: 
        number_2 = extract_number_after_multiple(specific_question)
        if number_2 is None:
            number_2 = 5
        if 'relaties'in specific_question:
            results = filtered_data["relatie"].value_counts().reset_index()
            results.columns = ["Relatie", "Aantal"]
        elif 'domeinen' in specific_question:
            results = pd.Series([dom for sublist in filtered_data["domains"] for dom in sublist]).value_counts().reset_index()
            results.columns = ["Domein", "Aantal"]
        elif any(word in specific_question for word in ['stoornissen', 'ontwikkelingsproblemen',
                                                    'ontwikkelingsstoornissen', 'problemen']):
            results = pd.Series([dom for sublist in filtered_data["positive"] for dom in sublist]).value_counts().reset_index()
            results.columns = ["Ontwikkelingsprobleem", "Aantal"]
        elif any(word in specific_question for word in ['leeftijdscategorieën', 'leeftijdsgroepen']):
            results = filtered_data["age_range"].value_counts().reset_index()
            results.columns = ["Leeftijdsgroep", "Aantal"]
        if top == 1:
            results = results.sort_values(by="Aantal", ascending=True)
        if top == 2:
            results = results.sort_values(by="Aantal", ascending=False)
        return results.head(number_2)

def combo_correlation(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    stoornis = extract_stoornis_in_question(specific_question)
    relatie = extract_relatie_in_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    signal = extract_signal_in_question(specific_question, signal_df)
    variables = [relatie, stoornis, domein, signal]
    non_none_variables = [var for var in variables if var is not None]
    if len(non_none_variables) != 1:
        return "Er mag maar 1 extra variabele naast leeftijd zijn om de correlatie te berekenen."
    var_1 = non_none_variables[0]
    if var_1 == relatie:
        freq_per_leeftijd = data.groupby('age_months')['relation'].apply(lambda x: (x == relatie).sum()).reset_index()
        freq_per_leeftijd.columns = ['age_months', f'{relatie}_frequency']
    elif var_1 == domein:
        freq_per_leeftijd = data.groupby('age_months')['domains'].apply(lambda x: x.apply(lambda domains: domein in domains).sum()).reset_index()
        freq_per_leeftijd.columns = ['age_months', f'{domein}_frequency']
    elif var_1 == stoornis:
        freq_per_leeftijd = data.groupby('age_months')['positive'].apply(lambda x: x.apply(lambda positive: stoornis in positive).sum()).reset_index()
        freq_per_leeftijd.columns = ['age_months', f'{stoornis}_frequency']
    elif var_1 == signal:
        freq_per_leeftijd = data.groupby('age_months')['signals'].apply(lambda x: x.apply(lambda signals: signal in signals).sum()).reset_index()
        freq_per_leeftijd.columns = ['age_months', f'{signal}_frequency']
    data_var_1 = freq_per_leeftijd[f'{var_1}_frequency']
    data_var_2 = freq_per_leeftijd['age_months']
    correlation, p_value = stats.pearsonr(data_var_1, data_var_2)
    analysis_results = (
        f"De correlatie tussen {var_1} en leeftijd is {correlation:.2f} en de p-waarde is {p_value:.4f}. "
        "De correlatiecoëfficiënt geeft aan hoe sterk de lineaire relatie is tussen de twee variabelen: "
        "- Een waarde van 1 duidt op een perfecte positieve lineaire relatie, wat betekent dat als de ene variabele toeneemt, de andere ook toeneemt. "
        "- Een waarde van -1 duidt op een perfecte negatieve lineaire relatie, wat betekent dat als de ene variabele toeneemt, de andere afneemt. "
        "- Een waarde van 0 betekent geen lineaire relatie, wat inhoudt dat er geen consistente richting is in de relatie tussen de twee variabelen. "
        f"De p-waarde geeft aan of deze correlatie statistisch significant is. "
        f"Een p-waarde kleiner dan 0.05 wijst op een significante correlatie, wat betekent dat de kans groot is dat de correlatie niet door toeval komt. "
        f"Een p-waarde groter dan 0.05 suggereert dat de correlatie mogelijk het gevolg is van toeval, en dus niet statistisch significant is.")
    return analysis_results

def combo_howmany(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    filtered_data = filter_data(data, specific_question)
    subdomein = extract_subdomain_from_question(specific_question)
    if any(word in specific_question for word in ['signalen', 'signaal']):
        if subdomein is None:
            total_signals_per_row = filtered_data['signals'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            avg_signals = total_signals_per_row.median()
            alarm_signal_set = set(signal_df[signal_df['Is een alarmsignaal'] == True]['Vraag'])
            alarm_signals_per_row = filtered_data['signals'].apply(lambda x: sum(1 for s in x if s in alarm_signal_set) if isinstance(x, list) else 0)
            avg_alarm_signals = alarm_signals_per_row.median()
            analysis_result = f"Mediaan aantal signalen: {avg_signals:.2f}\n Mediaan aantal alarmsignalen: {avg_alarm_signals:.2f}"
            return analysis_result
        else:
            relevant_signals = set(signal_df[signal_df['Subdomein'] == subdomein]['Vraag'])
            total_signals_per_row = filtered_data['signals'].apply(
                lambda x: sum(1 for s in x if s in relevant_signals) if isinstance(x, list) else 0)
            avg_signals = total_signals_per_row.median()
            relevant_alarm_signals = set(signal_df[(signal_df['Subdomein'] == subdomein) & (signal_df['Is een alarmsignaal'] == True)]['Vraag'])
            alarm_signals_per_row = filtered_data['signals'].apply(
                lambda x: sum(1 for s in x if s in relevant_alarm_signals) if isinstance(x, list) else 0)
            avg_alarm_signals = alarm_signals_per_row.median()
            analysis_result = f"Mediaan aantal signalen in {subdomein}: {avg_signals:.2f}\n Mediaan aantal alarmsignalen: {avg_alarm_signals:.2f}"
            return analysis_result
    else:
        aantal = len(filtered_data)
        analysis_result = f"Dit geldt voor {aantal} kinderen."
        return analysis_result

def combo_atleast(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(data)
    filtered_data = filter_data(data, specific_question)
    subdomein = extract_subdomain_from_question(specific_question)
    threshold = extract_number_after_minimum(specific_question)
    if subdomein is None:
        total_signals_per_row = filtered_data['signals'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        avg_signals = total_signals_per_row[total_signals_per_row >= threshold].count()
        percentage = avg_signals/total_rows*100
        if any(word in specific_question for word in ['alarmsignaal', 'alarmsignalen']):
            alarm_signal_set = set(signal_df[signal_df['Is een alarmsignaal'] == True]['Vraag'])
            alarm_signals_per_row = filtered_data['signals'].apply(lambda x: sum(1 for s in x if s in alarm_signal_set) if isinstance(x, list) else 0)
            avg_alarm_signals = alarm_signals_per_row[alarm_signals_per_row >= threshold].count()
            percentage = avg_alarm_signals/total_rows*100
            analysis_result = f"Er zijn {avg_alarm_signals} kinderen met ten minste {threshold} alarmsignalen. Dat is {percentage}% van de totale dataset.\n"
            return analysis_result
        else:
            analysis_result = f"Er zijn {avg_signals} kinderen met ten minste {threshold} signalen. Dat is {percentage}% van de totale dataset.\n"
            return analysis_result
    if 'signalen' not in specific_question:
        aantal = (data['positive'].apply(len) > threshold).sum()
        analysis_result = f"Er zijn {aantal} kinderen die ten minste {threshold} keer boven de cutoff scoren voor een ontwikkelingsprobleem."
    
    else:
        relevant_signals = set(signal_df[signal_df['Subdomein'] == subdomein]['Vraag'])
        total_signals_per_row = filtered_data['signals'].apply(
            lambda x: sum(1 for s in x if s in relevant_signals) if isinstance(x, list) else 0)
        avg_signals = total_signals_per_row[total_signals_per_row >= threshold].count()
        percentage = avg_signals/total_rows*100
        if any(word in specific_question for word in ['alarmsignaal', 'alarmsingalen']):
            relevant_alarm_signals = set(signal_df[(signal_df['Subdomein'] == subdomein) & (signal_df['Is een alarmsignaal'] == True)]['Vraag'])
            alarm_signals_per_row = filtered_data['signals'].apply(
                lambda x: sum(1 for s in x if s in relevant_alarm_signals) if isinstance(x, list) else 0)
            avg_alarm_signals = alarm_signals_per_row[alarm_signals_per_row >= threshold].count()
            percentage = avg_alarm_signals/total_rows*100
            analysis_result = f"Er zijn {avg_alarm_signals} kinderen met ten minste {threshold} alarmsignalen. Dat is {percentage}% van de totale dataset.\n"
            return analysis_result
        else: 
            analysis_result = f"Er zijn {avg_signals} kinderen met ten minste {threshold} signalen. Dat is {percentage}% van de totale dataset.\n"
            return analysis_result

def combo_comparison(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame, clf, vectorizer, encoder) -> pd.DataFrame:
    question = vectorizer.transform([specific_question])
    y_pred = clf.predict(question)
    element = encoder.inverse_transform(y_pred)[0]
    number = extract_number_from_signal_question(specific_question)
    if number is None:
        number = extract_number_from_question(specific_question)
        if number is None:
            number = 3
    top = 2
    if any(word in specific_question for word in ['meer', 'vaker']):
        top = 1
    if any(word in specific_question for word in ['vergelijk', 'vergelijken']):
       top = 3 
    filtered_data = filter_data_comp(data, specific_question, element)
    if element == 'relatie':
        relatie = extract_relatie_in_question(specific_question)
        if len(relatie) != 2:
            analysis_result = f'Er is iets mis in de vraag. Voeg 2 elementen toe uit relatie/leeftijd om te vergelijken.' 
            return analysis_result
        else:
            df1 = filtered_data[filtered_data['relation'] == relatie[0]]
            df2 = filtered_data[filtered_data['relation'] == relatie[1]]
    if element == 'leeftijd':
        range = extract_month_from_question(specific_question)
        if range is None:
            number_1, number_2 = extract_time_from_question(specific_question)
            specific_question = specific_question.replace(str(number_1), None).replace(str(number_2), None)
            number_3, number_4 = extract_time_from_question(specific_question)
            if any(i is None for i in [number_2, number_1, number_3, number_4]):
                analysis_result = f'Er is iets mis in de vraag. Voeg 2 elementen toe uit relatie/leeftijd om te vergelijken.' 
                return analysis_result
            else:
                if number_1 == number_2:
                    df1 = filtered_data[filtered_data['age_months']==number_1]
                if number_3 == number_4:
                    df2 = filtered_data[filtered_data['age_months']==number_3]
                if number_3 != number_4:
                    df2 = filtered_data[(filtered_data['age_months'].between(number_4, number_3))]
                if number_1 != number_2:
                    df2 = filtered_data[(filtered_data['age_months'].between(number_1, number_2))]
        else:
            specific_question = specific_question.replace(str(range), None)
            range_2 = extract_month_from_question(specific_question)
            if any(i is None for i in [range, range_2]):
                analysis_result = f'Er is iets mis in de vraag. Voeg 2 elementen toe uit relatie/leeftijd om te vergelijken.' 
                return analysis_result
            else: 
                df2 = filtered_data[filtered_data['age_range']==range]
                df1 = filtered_data[filtered_data['age_range']==range_2]
    
    all_signals_1 = [signal for sublist in df1['signals'] for signal in sublist]
    all_signals_2 = [signal for sublist in df2['signals'] for signal in sublist]
    signal_counts_1 = pd.DataFrame(Counter(all_signals_1).items(), columns=['signals', 'count_1'])
    signal_counts_2 = pd.DataFrame(Counter(all_signals_2).items(), columns=['signals', 'count_2'])
    signal_counts_1['signals'] = signal_counts_1['signals'].apply(
    lambda x: 'geen signalen aangeduid' if isinstance(x, list) and len(x) == 0 else tuple(x) if isinstance(x, list) else x)
    signal_counts_2['signals'] = signal_counts_2['signals'].apply(
    lambda x: 'geen signalen aangeduid' if isinstance(x, list) and len(x) == 0 else tuple(x) if isinstance(x, list) else x)
    merged_counts = pd.merge(signal_counts_1, signal_counts_2, on='signals', how='outer').fillna(0)
    if top == 1:
        merged_counts['match'] = ((merged_counts['count_1'] > 0) & 
                                (merged_counts['count_2'] > 0) & 
                                (merged_counts['count_1'] >= number * merged_counts['count_2'])) | \
                                ((merged_counts['count_2'] == 0) & 
                                (merged_counts['count_1'] == number))

    if top == 2:
        merged_counts['match'] = ((merged_counts['count_1'] > 0) & 
                                (merged_counts['count_2'] > 0) & 
                                (merged_counts['count_2'] >= number * merged_counts['count_1'])) | \
                                ((merged_counts['count_1'] == 0) & 
                                (merged_counts['count_2'] == number))
    if top == 3:
        merged_counts['match'] = ((merged_counts['count_1'] > 0) & (merged_counts['count_2'] > 0) & 
                          ((merged_counts['count_1'] >= number * merged_counts['count_2']) | 
                           (merged_counts['count_2'] >= number * merged_counts['count_1']))) | \
                         ((merged_counts['count_1'] == 0) & (merged_counts['count_2'] == number)) | \
                         ((merged_counts['count_2'] == 0) & (merged_counts['count_1'] == number))
    results = merged_counts[merged_counts['match']].copy()
    results = results.rename(columns={
    'count_1': str(relatie[1]),
    'count_2': str(relatie[0]),
    'signals': 'Signaal'})
    results = results.drop(columns=['match'])
    return results   

def time_more(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    data = filter_data(data, specific_question)
    month,_ = extract_time_from_question(specific_question)
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
    if any(word in specific_question for word in ['meer', 'vaker', 'frequenter']):
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
    if any(word in specific_question for word in ['minder', 'minder vaak']):
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

def time_evolution_element(data: pd.DataFrame, specific_question: str, signal_df: pd.DataFrame) -> str:
    leeftijd_1, leeftijd_2 = extract_time_from_question(specific_question)    
    threshold = extract_threshold_from_question(specific_question)
    if threshold is None:
        threshold = 5
    latest_timestamp = data['timestamp'].max()
    past_timestamp = latest_timestamp - pd.DateOffset(months=leeftijd_1)
    old_period = data[data['timestamp'] < past_timestamp]
    new_period = data[data['timestamp'] >= past_timestamp]
    if any(word in specific_question for word in ['meer', 'vaker', 'frequenter']) and domein is None:
        if any(word in specific_question for word in ['domeinen', 'domein']):
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
        if any(word in specific_question for word in ['relaties', 'relatie']):
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
        if any(word in specific_question for word in ['stoornis', 'stoornissen', 'ontwikkelingsprobleem', 'ontwikkelingsproblemen', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis']):
            old_stoornis = [stoornis for sublist in old_period['positive'].dropna() for stoornis in (sublist if isinstance(sublist, list) else [sublist])]
            new_stoornis = [stoornis for sublist in new_period['positive'].dropna() for stoornis in (sublist if isinstance(sublist, list) else [sublist])]
            old_counts = pd.Series(old_stoornis).value_counts()
            new_counts = pd.Series(new_stoornis).value_counts()
            significant_stoornis = []
            for stoornis, new_count in new_counts.items():
                old_count = old_counts.get(stoornis, 0)
                if old_count > 0 and new_count >= old_count * threshold: 
                    significant_stoornis.append(f"{stoornis} (Oud: {old_count}, Nieuw: {new_count})")
            if significant_stoornis:
                analysis_results = "Het ontwikkelingsprobleem die meer als positief gerekend zijn, zijn:\n" + "\n".join(significant_stoornis)
            else:
                analysis_results = "Er zijn geen ontwikkelingsproblemen die meer voorkomen."
            return analysis_results
        if any(word in specific_question for word in ['leeftijdsgroep', 'leeftijdsgroepen', 'leeftijdscategorieën', 'leeftijdscategorie']):
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
    if any(word in specific_question for word in ['minder', 'minder vaak']):
        if any(word in specific_question for word in ['domeinen', 'domein']):
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
        if any(word in specific_question for word in ['relaties', 'relatie']):
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
        if any(word in specific_question for word in ['stoornis', 'stoornissen', 'ontwikkelingsprobleem', 'ontwikkelingsproblemen', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis']):
            old_stoornis = [stoornis for sublist in old_period['positive'].dropna() for stoornis in (sublist if isinstance(sublist, list) else [sublist])]
            new_stoornis = [stoornis for sublist in new_period['positive'].dropna() for stoornis in (sublist if isinstance(sublist, list) else [sublist])]
            old_counts = pd.Series(old_stoornis).value_counts()
            new_counts = pd.Series(new_stoornis).value_counts()
            significant_stoornis = []
            for stoornis, new_count in new_counts.items():
                old_count = old_counts.get(stoornis, 0)
                if old_count > 0 and new_count <= old_count / threshold: 
                    significant_stoornis.append(f"{stoornis} (Oud: {old_count}, Nieuw: {new_count})")
            if significant_stoornis:
                analysis_results = "Het onwikkelingsprobleem die minder als positief gerekend zijn, zijn:\n" + "\n".join(significant_stoornis)
            else:
                analysis_results = "Er zijn geen ontwikkelingsproblemen die minder voorkomen."
            return analysis_results
        if any(word in specific_question for word in ['leeftijdsgroep', 'leeftijdsgroepen', 'leeftijdscategorieën', 'leeftijdscategorie']):
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
            
def time_element(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> pd.DataFrame:
    relatie = extract_relatie_in_question(specific_question)
    domein = extract_domein_in_question(specific_question)
    stoornis = extract_stoornis_in_question(specific_question)
    leeftijd = extract_range_from_question(specific_question)
    leeftijd_1, leeftijd_2 = extract_time_from_question(specific_question)
    signal = extract_signal_in_question(specific_question, signal_df)
    month = extract_month_from_question(specific_question)
    data = filter_data(data, specific_question)
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=month)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    if any(word in specific_question for word in ['domeinen', 'domein']) and domein is None:
        exploded_data = filtered_data.explode('domains').dropna(subset=['domains'])
        monthly_counts = exploded_data.groupby(['month', 'domains']).size().reset_index(name='Totaal aantal')
        monthly_counts.rename(columns={'month': 'Maand', 'domains': 'Domein'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Totaal aantal'], ascending=[True, False])
    if any(word in specific_question for word in ['relatie', 'relaties']) and relatie is None:
        monthly_counts = (
            filtered_data.groupby(['month', 'relation'])
            .size()
            .reset_index(name='Total Count'))
        monthly_counts.rename(columns={'month': 'Maand', 'relation': 'Relatie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Relatie'])
    if any(word in specific_question for word in ['leeftijdsgroepen', 'leeftijdsgroep', 'leeftijdscategorie', 'leeftijdscategorieën']) and leeftijd_1 is None:
        monthly_counts = (
            filtered_data.groupby(['month', 'age_range'])
            .size()
            .reset_index(name='Totaal aantal'))
        monthly_counts.rename(columns={'month': 'Maand', 'age_range': 'Leeftijdscategorie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Leeftijdscategorie'])
    if any(word in specific_question for word in ['stoornis', 'stoornissen', 'ontwikkelingsprobleem', 'ontwikkelingsproblemen', 'ontwikkelingsstoornissen', 'ontwikkelingsstoornis']) and stoornis is None:
        exploded_data = filtered_data.dropna(subset=['positive'])
        monthly_counts = (
            exploded_data.groupby(['month', 'positive'])
            .size()
            .reset_index(name='Total Count'))
        monthly_counts.rename(columns={'month': 'Maand', 'positive': 'Ontwikkelingsprobleem'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Ontwikkelingsprobleem'])
    if relatie is not None:
        filtered_data = filtered_data[(filtered_data['relation'] == relatie)]
        monthly_counts = (
            filtered_data.groupby(['month', 'relation'])
            .size()
            .reset_index(name='Total Count'))
        monthly_counts.rename(columns={'month': 'Maand', 'relation': 'Relatie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Relatie'])
    if domein is not None:
        filtered_data = filtered_data[(filtered_data['domains'].apply(lambda x: domein in x))]
        exploded_data = filtered_data.explode('domains').dropna(subset=['domains'])
        monthly_counts = exploded_data.groupby(['month', 'domains']).size().reset_index(name='Totaal aantal')
        monthly_counts.rename(columns={'month': 'Maand', 'domains': 'Domein'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Totaal aantal'], ascending=[True, False])
    if stoornis is not None:
        filtered_data = filtered_data[(filtered_data['positive'].apply(lambda x: stoornis in x))]
        exploded_data = filtered_data.dropna(subset=['positive'])
        monthly_counts = (
            exploded_data.groupby(['month', 'positive'])
            .size()
            .reset_index(name='Total Count'))
        monthly_counts.rename(columns={'month': 'Maand', 'positive': 'Ontwikkelingsprobleem'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Ontwikkelingsprobleem'])
    if leeftijd is not None:
        filtered_data = filtered_data[(filtered_data['age_range'] == leeftijd)]
        monthly_counts = (
            filtered_data.groupby(['month', 'age_range'])
            .size()
            .reset_index(name='Totaal aantal'))
        monthly_counts.rename(columns={'month': 'Maand', 'age_range': 'Leeftijdscategorie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Leeftijdscategorie'])
    if leeftijd_1 != leeftijd_2:
        filtered_data = filtered_data[(filtered_data['age_months'].between(leeftijd_1, leeftijd_2))]
        monthly_counts = (
            filtered_data.groupby(['month', 'age_months'])
            .size()
            .reset_index(name='Totaal aantal'))
        monthly_counts.rename(columns={'month': 'Maand', 'age_range': 'Leeftijdscategorie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Leeftijdscategorie'])
    if leeftijd_1 is not None:
        filtered_data = filtered_data[(filtered_data['age_months'] == leeftijd_1)]
        monthly_counts = (
            filtered_data.groupby(['month', 'age_months'])
            .size()
            .reset_index(name='Totaal aantal'))
        monthly_counts.rename(columns={'month': 'Maand', 'age_range': 'Leeftijdscategorie'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Leeftijdscategorie'])
    if signal is not None:
        filtered_data = filtered_data[(filtered_data['signals'].apply(lambda x: signal in x))]
        monthly_counts = (
            exploded_data.groupby(['month', 'signals'])
            .size()
            .reset_index(name='Total Count'))
        monthly_counts.rename(columns={'month': 'Maand', 'signals': 'Signaal'}, inplace=True)
        return monthly_counts.sort_values(by=['Maand', 'Signaal'])

def time_data(data: pd.DataFrame, specific_question, signal_df: pd.DataFrame) -> str:
    data = filter_data(data, specific_question)
    months = extract_month_from_question(specific_question)
    if months is None:
        months = 6
    data['month'] = data['timestamp'].dt.to_period('M')
    latest_timestamp = data['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.DateOffset(months=months)
    filtered_data = data[data['timestamp'] >= cutoff_timestamp]
    if any(word in specific_question for word in ['per maand', 'maandelijks']):
        monthly_counts = filtered_data.groupby('month').size()
        monthly_counts.index.name = 'Maand'
        return monthly_counts.sort_values()
    else:  
        new_data_count = len(filtered_data)
        analysis_results = f"In de laatste {months} maanden zijn er {new_data_count} nieuwe datapunten geregistreerd.\n"
        return analysis_results


  