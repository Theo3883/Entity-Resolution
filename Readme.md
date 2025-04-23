# Entity Resolution for Company Data

## Approach and Methodology

### 1. Data Analysis and Understanding

Before coding, I analyzed what defines a company's identity:

- **Legal Identity**: Official registered name, registration numbers, tax IDs
- **Physical Presence**: Headquarters location, branch addresses
- **Digital Footprint**: Website domains, email addresses
- **Contact Information**: Phone numbers, contact persons
- **Operational Identity**: Industry classifications, products/services

Many of these identifiers may vary across systems while still referring to the same entity. For example, "IBM" and "International Business Machines Corporation" are the same company.

### 2. Blocking Strategy

With 33,446 records, comparing every record pair would require over 500 million comparisons. To make this efficient, I implemented multi-key blocking:

- **Domain Blocking**: Records with the same website domain root (ignoring TLDs)
- **Phone Blocking**: Records with the same normalized phone number
- **Name+Location Blocking**: Records with similar name prefixes in the same country

This reduces the comparison space by orders of magnitude while maintaining high recall.

### 3. Multi-attribute Comparison

I designed a multi-faceted comparison strategy that considers:

- **Name Similarity**: Using sequence matching with special handling for prefixes and common name variations
- **Location Matching**: Hierarchical matching of country → city → street
- **Contact Information**: Exact matching of normalized phone numbers
- **Digital Identity**: Domain matching with TLD normalization

### 4. Graph-based Resolution

Rather than using pairwise decisions, I constructed a similarity graph where:
- Nodes represent individual company records
- Edges connect records that are likely the same entity
- Connected components in this graph represent the same real-world company

This approach handles transitive relationships effectively (if A=B and B=C, then A=C).

## Key Design Decisions

### Selection of Matching Attributes

I prioritized attributes based on their:

1. **Uniqueness**: How distinctive the attribute is for identifying a company
2. **Availability**: How often the attribute is present in the dataset
3. **Consistency**: How reliably the attribute stays the same across different systems

The most valuable attributes were:
- Website domain (high uniqueness, moderate availability)
- Phone numbers (high uniqueness when available)
- Company name + location (moderate uniqueness, high availability)

### Handling Incomplete Data

Many records have missing fields, requiring careful handling:

- **Fallback Strategy**: If primary identifiers (domain/phone) are unavailable, fall back to name + location matching
- **Field Weighting**: Give higher confidence to matches on more unique fields
- **Confidence Thresholds**: Adjust match thresholds based on available information

### Selecting Representative Records

For each cluster of duplicate records, I implemented a scoring system to select the most complete record as the representative, using:
- Number of non-empty fields
- Presence of key identifiers (company name, website, phone, address)

## Implementation Details

### Tech Stack

- **Language**: Python 
- **Core Libraries**:
  - pandas: Data manipulation and preprocessing
  - NetworkX: Graph construction and analysis
  - scikit-learn: TF-IDF vectorization for text similarity
  - difflib: Sequence matching for name comparisons
  - multiprocessing: Parallel processing for efficiency

### Performance Optimization

Several techniques were used to optimize performance:

1. **Efficient Blocking**: Creating targeted comparison blocks
2. **Chunked Processing**: Processing large blocks in smaller chunks
3. **Early Termination**: For obvious matches (exact domain matches,etc)
4. **Parallelization**: Using multiple CPU cores for block processing
5. **Sparse Representations**: For memory-efficient similarity calculations

## Scalability Considerations

While this solution works well for ~33K records, scaling to billions would require:

1. **Distributed Processing**: Using frameworks 
2. **Database-backed Graph**: Using graph databases for connectivity analysis
3. **Optimized Blocking**: More sophisticated blocking strategies
4. **Incremental Processing**: Processing new records incrementally

## Results and Evaluation

The entity resolution process successfully identified duplicate records with high accuracy:

- **Total Records**: 33,446
- **Unique Entities After Resolution**: 5,820
- **Duplicate Sets Found**: 27,626 sets of duplicate records

The most common patterns of duplicates were:
- Same company with different locations
- Variations in company name formatting
- Same company imported from different data sources

## Future Improvements

1. **Machine Learning**: Using supervised models to learn matching patterns
2. **Industry-Specific Rules**: Adding rules for specific industries or company types
3. **Cross-lingual Matching**: Better handling of companies with names in multiple languages
4. **Hierarchical Entity Resolution**: Handling parent-child company relationships
