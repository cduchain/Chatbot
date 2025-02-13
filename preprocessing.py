import pandas as pd
import json
import re

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Apply to all string columns in the DataFrame
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(lambda x: re.sub(r'[\x00-\x1F\x7F\xA0]', '', str(x)) if isinstance(x, str) else x)
    return df

def preprocess_data(data_file: str, signal_file: str, domein_file: str, slider_1, slider_2, slider_3) -> pd.DataFrame:
    df = pd.read_csv(data_file, sep=';', on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['Created At'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')
    gegevens = pd.DataFrame(df.iloc[:, 11])
    data_rows = []
    
    for index, row in gegevens.iterrows():
        if isinstance(row['Gegevens'], str):
            try:
                json_data = json.loads(row['Gegevens'])
            except json.JSONDecodeError:
                print(f"JSON decode error at index {index}: {row['Gegevens']}")
                continue
        else:
            print(f"Skipping index {index} due to non-string type: {row['Gegevens']}")
            continue

        if json_data['referrals']:
            first_referral = json_data['referrals'][0]  
            language_disorder = first_referral.get('language_disorder', False)
            motoric_disorder = first_referral.get('motoric_disorder', False)
            ass_or_multiple_disorder = first_referral.get('ass_or_multiple_disorders', False)
        else:
            language_disorder = motoric_disorder = ass_or_multiple_disorder = False

        data_rows.append({
            'age_months': json_data['age_months'],
            'relation': json_data['relation'],
            'domains': json_data['domains'],
            'signals': json_data['signals'],
            'screener_answers': json_data['screener_answers'],
            'referrals': json_data['referrals'],
            'language_disorder': language_disorder,
            'motoric_disorder': motoric_disorder,
            'ass_or_multiple_disorder': ass_or_multiple_disorder,
            'timestamp': df.loc[index, 'timestamp']
        })
    print(f"Total rows appended: {len(data_rows)}")
    processed_df = pd.DataFrame(data_rows)
    processed_df.drop(columns=['referrals', 'screener_answers', 'language_disorder', 'motoric_disorder', 'ass_or_multiple_disorder'], inplace=True)
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], utc=True, errors='coerce')
    bins = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
    labels = list(range(len(bins) - 1))
    processed_df['age_range'] = pd.cut(processed_df['age_months'], bins=bins, labels=labels, right=True)
    signal_df = pd.read_csv(signal_file, sep=';')
    signal_df = signal_df.drop(signal_df.columns[[i for i in range(8)] + [13]], axis=1)
    signal_df['Aantal keer zichtbaar'] = 0
    signal_df['Percentage gekozen keren'] = 0.0 
    print(signal_df.columns)
    for index, signal_row in signal_df.iterrows():
        signal_text = signal_row['Vraag']
        min_age = signal_row['Minimum leeftijd']
        max_age = signal_row['Maximum leeftijd']
        relevant_children = processed_df[ 
            (processed_df['age_months'] >= min_age) & 
            (processed_df['age_months'] <= max_age)]
        signal_df.at[index, 'Aantal keer zichtbaar'] = len(relevant_children)
        if len(relevant_children) > 0:
            selected_count = relevant_children['signals'].apply(
                lambda x: signal_text in x if isinstance(x, list) else False).sum()
            signal_df.at[index, 'Percentage gekozen keren'] = round((selected_count / len(relevant_children)) * 100, 2)
        else:
            signal_df.at[index, 'Percentage gekozen keren'] = 0.0
    domain_df = pd.read_csv(domein_file, sep=';')
    domain_df = domain_df.drop(domain_df.columns[[i for i in range(8)] + [10]], axis=1)
    domain_mapping = dict(zip(domain_df['Naam'], domain_df['Domein']))
    domain_mapping['Communicatie'] = 'taal en/of communicatie'  
    signal_df['Domein'] = signal_df['Subdomein'].map(domain_mapping)  
    signal_df['Is een alarmsignaal'] = signal_df['Is een alarmsignaal'].astype(str).str.lower() == 'true'
    stoornis = ['language disorder', 'motoric disorder', 'autism']
    positive_list = []
    subdomain_to_disorder_mapping = {
        'Grove motoriek': 'motoric disorder',
        'Houdingsveranderingen': 'motoric disorder',
        'Fijne motoriek': 'motoric disorder',
        'Zelfredzaamheid': 'motoric disorder',
        'Kwaliteit van bewegen': 'motoric disorder',
        'Taalbegrip': 'language disorder',
        'Taalproductie': 'language disorder',
        'Articulatie': 'language disorder',
        'Communicatie': 'autism',
        'Interactie met anderen': 'autism',
        'Imitatie': 'autism',
        'Samen spelen': 'autism',
        'Stereotiep spel en stereotiepe bewegingen': 'autism',
        'Spel': 'autism',
        'Angst': 'autism',
        'Andere opvallende gedragingen': 'autism'}
    signal_df['Disorder'] = signal_df['Subdomein'].map(subdomain_to_disorder_mapping)
    processed_df['positive'] = processed_df.apply(lambda x: [], axis=1)
    for index, row in processed_df.iterrows():
        disorder_counts = {'language disorder': 0, 'motoric disorder': 0, 'autism': 0}
        alarm_signal_counts = {'language disorder': 0, 'motoric disorder': 0, 'autism': 0}
        for signal in row['signals']:
            signal_row = signal_df[signal_df['Vraag'] == signal].iloc[0]  # Get corresponding signal row
            disorder = signal_row['Disorder']
            disorder_counts[disorder] += 1
            if signal_row['Is een alarmsignaal']:
                alarm_signal_counts[disorder] += 1
        positive = set()
        for disorder, count in disorder_counts.items():
            if count > slider_1:
                positive.add(disorder)
            if count > slider_2:
                positive.add(disorder)
        for disorder, alarm_count in alarm_signal_counts.items():
            if alarm_count > slider_3:
                positive.add(disorder)
        processed_df.at[index, 'positive'] = list(positive)
    processed_df = processed_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    signal_df = signal_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    processed_df['domains'] = processed_df['domains'].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else x)
    processed_df['signals'] = processed_df['signals'].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else x)
    clean_dataframe(signal_df)
    clean_dataframe(processed_df)
    return processed_df, signal_df
