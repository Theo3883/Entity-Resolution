import pandas as pd
import numpy as np
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from collections import defaultdict
import networkx as nx
import multiprocessing
from functools import partial

def clean_text(text):
    """Clean and standardize text"""
    if not isinstance(text, str):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def create_blocks(dataframe):
    """Create blocking strategy with multiple keys
    
    BLOCKING STRATEGY:
    This implements a multi-key blocking approach to efficiently reduce the comparison space:
    1. Domain Blocking: Groups records with the same website domain root (ignoring TLDs)
    2. Phone Blocking: Groups records with identical normalized phone numbers
    3. Name+Location Blocking: Groups records with similar name prefixes in the same country
    
    This strategy reduces comparisons from O(n²) to a much smaller subset while maintaining high recall.
    """
    blocks = defaultdict(list)
    
    # Domain blocking - records with the same website domain root
    domain_mask = dataframe['norm_domain'] != ''
    if domain_mask.any():
        dataframe.loc[domain_mask].groupby('norm_domain').apply(
            lambda x: blocks.update({f'domain_{x.name}': x.index.tolist()}) 
            if len(x) > 1 else None
        )
    
    # Phone blocking - records with the same normalized phone number
    phone_mask = (dataframe['norm_phone'] != '') & (dataframe['norm_phone'].str.len() >= 6)
    if phone_mask.any():
        dataframe.loc[phone_mask].groupby('norm_phone').apply(
            lambda x: blocks.update({f'phone_{x.name}': x.index.tolist()})
            if len(x) > 1 else None
        )
    
    # Name+Location blocking - records with similar name prefixes in the same country    
    name_loc_mask = (dataframe['name_prefix'] != '') & (dataframe['main_country'] != '')
    if name_loc_mask.any():
        name_loc_key = dataframe['name_prefix'] + '|' + dataframe['main_country']
        name_loc_groups = dataframe.loc[name_loc_mask].groupby(name_loc_key)
        for key, indices in name_loc_groups.groups.items():
            if len(indices) > 1 and len(indices) < 500:
                blocks[f'nameprefix_country_{key}'] = indices.tolist()
    
    return blocks

def calculate_name_similarity(names1, names2):
    """Calculate similarity between arrays of names using vectorized operations"""
    similarities = []
    for i in range(len(names1)):
        name1, name2 = names1[i], names2[i]
        if (name1 and name2 and 
            (name1.startswith(name2) or name2.startswith(name1)) and 
            min(len(name1), len(name2)) > 3):
            similarities.append(0.9)
        else:
            similarity = difflib.SequenceMatcher(None, name1, name2).quick_ratio()
            similarities.append(similarity)
    return similarities

def process_block(block_data, dataframe):
    """Process a single block of potentially matching records
    
    MULTI-ATTRIBUTE COMPARISON:
    This function implements a multi-faceted comparison strategy that considers:
    1. Name Similarity: Using sequence matching with special handling for prefixes
    2. Location Matching: Hierarchical matching of country → city → street
    3. Contact Information: Exact matching of normalized phone numbers
    4. Digital Identity: Domain matching with TLD normalization
    
    """
    block_name, indices = block_data
    edges = []
    
    if len(indices) > 5000:
        print(f"Skipping oversized block {block_name} with {len(indices)} records")
        return edges
    
    # For domain blocks, consider all pairs as matches (high confidence in domain uniqueness)
    if 'domain_' in block_name:
        for i in range(len(indices)-1):
            for j in range(i+1, len(indices)):
                edges.append((indices[i], indices[j]))
        return edges
    
    # For phone blocks, verify phone numbers are valid before matching
    if 'phone_' in block_name and len(indices) < 100:
        for i in range(len(indices)-1):
            for j in range(i+1, len(indices)):
                if (dataframe.iloc[indices[i]]['norm_phone'] and 
                    dataframe.iloc[indices[j]]['norm_phone'] and
                    len(dataframe.iloc[indices[i]]['norm_phone']) >= 6):
                    edges.append((indices[i], indices[j]))
        return edges
    
    # For larger blocks, process in chunks to manage memory usage
    chunk_size = 100
    if len(indices) > chunk_size:
        chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]
        for chunk in chunks:
            for i, idx1 in enumerate(chunk):
                record1 = dataframe.iloc[idx1]
                
                for idx2 in chunk[i+1:]:
                    record2 = dataframe.iloc[idx2]
                    
                    # Company name similarity comparison
                    if (record1['clean_company_name'] and record2['clean_company_name']):
                        name_similarity = difflib.SequenceMatcher(None, 
                                                                 record1['clean_company_name'], 
                                                                 record2['clean_company_name']).ratio()
                        
                        if name_similarity > 0.85:
                            edges.append((idx1, idx2))
                            continue
                    
                    # Location-based matching (country, city, street hierarchy)
                    if (record1['main_country'] == record2['main_country'] and
                        record1['main_city'] == record2['main_city'] and
                        record1['main_country'] and record2['main_country'] and
                        record1['main_city'] and record2['main_city']):
                        
                        if (record1['main_street'] and record2['main_street'] and
                            record1['main_street'] == record2['main_street']):
                            edges.append((idx1, idx2))
    # Process small blocks more directly
    else:
        for i, idx1 in enumerate(indices[:-1]):
            record1 = dataframe.iloc[idx1]
            
            for idx2 in indices[i+1:]:
                record2 = dataframe.iloc[idx2]
                
                # Compare attributes using same logic as above
                if (record1['clean_company_name'] and record2['clean_company_name']):
                    name_similarity = difflib.SequenceMatcher(None, 
                                                             record1['clean_company_name'], 
                                                             record2['clean_company_name']).ratio()
                    
                    if name_similarity > 0.85:
                        edges.append((idx1, idx2))
                        continue
                
                if (record1['main_country'] == record2['main_country'] and
                    record1['main_city'] == record2['main_city'] and
                    record1['main_country'] and record2['main_country'] and
                    record1['main_city'] and record2['main_city']):
                    
                    if (record1['main_street'] and record2['main_street'] and
                        record1['main_street'] == record2['main_street']):
                        edges.append((idx1, idx2))
    
    return edges

def completeness_score(record):
    """Calculate completeness score for a record"""
    non_empty = sum(1 for val in record if pd.notna(val) and val != '')
    
    important_fields = ['company_name', 'website_domain', 'main_country', 'primary_phone']
    extra_points = sum(2 for field in important_fields if field in record.index and pd.notna(record[field]) and record[field] != '')
    
    return non_empty + extra_points

def main():
    
    start_time = time.time()

    print("Loading data...")
    file_path = 'veridion_entity_resolution_challenge.snappy.parquet'
    df = pd.read_parquet(file_path)
    print(f"Data loaded. Number of records: {len(df)}")

    print("Preprocessing data...")
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].fillna('')

    df['clean_company_name'] = df['company_name'].apply(clean_text)
    df['clean_commercial_names'] = df['company_commercial_names'].apply(clean_text)

    df['norm_domain'] = df['website_domain'].str.split('.').str[0]
    df['norm_phone'] = df['primary_phone'].apply(lambda x: re.sub(r'[^\d]', '', str(x)))

    print("Creating efficient blocking keys...")
    df['name_prefix'] = df['clean_company_name'].str[:3].fillna('')
    df['location_key'] = (df['main_country'] + '|' + df['main_city']).fillna('')

    blocks = create_blocks(df)
    print(f"Created {len(blocks)} efficient blocks for comparison")

    print("Building similarity graph...")
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))

    print("Comparing records within blocks...")
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    block_items = list(blocks.items())

    if num_cpus > 1 and len(block_items) > 10:
        print(f"Using {num_cpus} parallel processes")
        process_block_with_df = partial(process_block, dataframe=df)
        with multiprocessing.Pool(processes=num_cpus) as pool:
            all_edges = pool.map(process_block_with_df, block_items)
    else:
        print("Processing blocks sequentially")
        all_edges = []
        total_blocks = len(block_items)
        for i, block_data in enumerate(block_items):
            if i % 100 == 0:
                print(f"Processed {i}/{total_blocks} blocks...")
            all_edges.append(process_block(block_data, df))

    print("Building entity graph...")
    for edges in all_edges:
        G.add_edges_from(edges)

    print("Identifying entities...")
    # GRAPH-BASED RESOLUTION:
    # This approach constructs a similarity graph where:
    # - Nodes represent individual company records
    # - Edges connect records that are likely the same entity 
    # - Connected components in the graph represent the same real-world company
    # 
    # This handles transitive relationships effectively (if A=B and B=C, then A=C)
    # and provides a flexible framework that can incorporate different types of similarity evidence.
    connected_components = list(nx.connected_components(G))
    print(f"Found {len(connected_components)} connected components")

    print("Selecting representative records...")

    # Map each record to its entity ID based on connected components
    entity_mapping = {}
    for entity_id, component in enumerate(connected_components):
        for idx in component:
            entity_mapping[idx] = entity_id

    # Handle singleton records (no duplicates found)
    next_entity_id = len(connected_components)
    for idx in range(len(df)):
        if idx not in entity_mapping:
            entity_mapping[idx] = next_entity_id
            next_entity_id += 1

    # Select the most complete record as the representative for each entity
    representatives = []
    for entity_id in range(next_entity_id):
        indices = [idx for idx, eid in entity_mapping.items() if eid == entity_id]
        
        if len(indices) == 1:
            representatives.append(indices[0])
        else:
            entity_records = df.iloc[indices]
            scores = entity_records.apply(completeness_score, axis=1)
            best_idx = scores.idxmax()
            representatives.append(best_idx)

    df_deduped = df.loc[representatives].copy()

    print("Generating output...")

    columns_to_drop = ['clean_company_name', 'clean_commercial_names', 
                    'norm_domain', 'norm_phone', 'name_prefix', 'location_key']
    df_result = df_deduped.drop(columns=columns_to_drop)

    df_result['entity_id'] = df_result.index.map(entity_mapping)

    total_records = len(df)
    unique_entities = len(df_result)
    print(f"Total original records: {total_records}")
    print(f"Unique entities after deduplication: {unique_entities}")
    print(f"Duplicate sets found: {total_records - unique_entities}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")

    output_path = 'entity_resolution_deduped_solved.parquet'
    df_result.to_parquet(output_path)
    print(f"Results saved to: {output_path}")

    print("\nSample of deduplicated entities (original had multiple records):")
    sample_entities = [eid for eid, count in 
                    pd.Series([entity_mapping[i] for i in range(len(df))]).value_counts().items() 
                    if count > 1][:5]

    for entity_id in sample_entities:
        original_indices = [i for i, eid in entity_mapping.items() if eid == entity_id]
        rep_index = [i for i in representatives if entity_mapping[i] == entity_id][0]
        
        print(f"\nEntity {entity_id} - Originally had {len(original_indices)} records")
        print("Kept representative record:")
        print(f"  - {df.iloc[rep_index]['company_name']} ({df.iloc[rep_index]['main_country']})")
        print("Original duplicates:")
        for idx in original_indices:
            if idx != rep_index:
                print(f"  - {df.iloc[idx]['company_name']} ({df.iloc[idx]['main_country']})")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()