# Categorical Mappings

This document describes the categorical value mappings and canonicalization rules used by the AI Data Cleaner.

## Status Mappings

### Order Status
```json
{
  "delivered": ["delivered", "completed", "shipped", "fulfilled", "done", "delivered successfully"],
  "pending": ["pending", "processing", "in_progress", "waiting", "awaiting", "in queue"],
  "cancelled": ["cancelled", "canceled", "aborted", "terminated", "voided", "stopped"],
  "returned": ["returned", "refunded", "exchanged", "reversed", "sent back"]
}
```

### Payment Status
```json
{
  "paid": ["paid", "completed", "successful", "processed", "confirmed"],
  "pending": ["pending", "processing", "awaiting", "in_progress", "pending payment"],
  "failed": ["failed", "declined", "rejected", "error", "unsuccessful"],
  "refunded": ["refunded", "reversed", "returned", "cancelled"]
}
```

### Shipping Status
```json
{
  "shipped": ["shipped", "dispatched", "sent", "in_transit", "on_the_way"],
  "delivered": ["delivered", "received", "completed", "arrived"],
  "pending": ["pending", "processing", "preparing", "packaging"],
  "cancelled": ["cancelled", "voided", "aborted", "stopped"]
}
```

## Gender Mappings

```json
{
  "male": ["male", "m", "man", "masculine", "gentleman", "boy"],
  "female": ["female", "f", "woman", "feminine", "lady", "girl"],
  "other": ["other", "non-binary", "nb", "nonbinary", "non_binary", "prefer_not_to_say", "unknown"]
}
```

## Rating Mappings

### Quality Ratings
```json
{
  "excellent": ["excellent", "outstanding", "amazing", "fantastic", "perfect", "superb", "exceptional"],
  "good": ["good", "great", "nice", "satisfactory", "decent", "acceptable", "fine"],
  "average": ["average", "okay", "ok", "mediocre", "fair", "moderate", "so-so"],
  "poor": ["poor", "bad", "terrible", "awful", "horrible", "disappointing", "unsatisfactory"]
}
```

### Numeric Ratings (1-5 scale)
```json
{
  "5": ["5", "five", "excellent", "outstanding", "amazing", "perfect"],
  "4": ["4", "four", "good", "great", "very good", "above average"],
  "3": ["3", "three", "average", "okay", "fair", "moderate"],
  "2": ["2", "two", "below average", "poor", "not good", "disappointing"],
  "1": ["1", "one", "terrible", "awful", "horrible", "very poor"]
}
```

### Likert Scale
```json
{
  "strongly_agree": ["strongly agree", "strongly_agree", "completely agree", "totally agree"],
  "agree": ["agree", "somewhat agree", "mostly agree", "generally agree"],
  "neutral": ["neutral", "neither agree nor disagree", "no opinion", "undecided"],
  "disagree": ["disagree", "somewhat disagree", "mostly disagree", "generally disagree"],
  "strongly_disagree": ["strongly disagree", "strongly_disagree", "completely disagree", "totally disagree"]
}
```

## Priority Mappings

```json
{
  "high": ["high", "urgent", "critical", "priority", "important", "asap"],
  "medium": ["medium", "normal", "standard", "regular", "moderate"],
  "low": ["low", "minor", "non-urgent", "when possible", "low priority"]
}
```

## Industry-Specific Mappings

### Healthcare

#### Symptoms
```json
{
  "fever": ["fever", "high temperature", "elevated temperature", "pyrexia"],
  "cough": ["cough", "coughing", "persistent cough", "dry cough"],
  "headache": ["headache", "head pain", "migraine", "cephalgia"],
  "nausea": ["nausea", "feeling sick", "queasy", "nauseous"],
  "fatigue": ["fatigue", "tiredness", "exhaustion", "weakness"]
}
```

#### Diagnoses
```json
{
  "hypertension": ["hypertension", "high blood pressure", "htn", "elevated bp"],
  "diabetes": ["diabetes", "diabetes mellitus", "dm", "high blood sugar"],
  "asthma": ["asthma", "bronchial asthma", "reactive airway disease"],
  "pneumonia": ["pneumonia", "lung infection", "respiratory infection"]
}
```

#### Medications
```json
{
  "aspirin": ["aspirin", "asa", "acetylsalicylic acid"],
  "ibuprofen": ["ibuprofen", "advil", "motrin", "brufen"],
  "acetaminophen": ["acetaminophen", "tylenol", "paracetamol"],
  "insulin": ["insulin", "humulin", "lantus", "novolog"]
}
```

### Finance

#### Account Types
```json
{
  "checking": ["checking", "current", "demand deposit", "transaction account"],
  "savings": ["savings", "deposit account", "time deposit"],
  "credit": ["credit", "credit card", "revolving credit", "line of credit"],
  "investment": ["investment", "brokerage", "securities", "portfolio"]
}
```

#### Transaction Types
```json
{
  "deposit": ["deposit", "credit", "incoming", "receipt", "payment received"],
  "withdrawal": ["withdrawal", "debit", "outgoing", "payment made", "expense"],
  "transfer": ["transfer", "move", "shift", "relocate", "reallocate"],
  "fee": ["fee", "charge", "cost", "expense", "service charge"]
}
```

#### Currency Codes
```json
{
  "USD": ["usd", "us dollar", "dollar", "$", "us$"],
  "EUR": ["eur", "euro", "€", "european currency"],
  "GBP": ["gbp", "pound", "sterling", "£", "british pound"],
  "JPY": ["jpy", "yen", "¥", "japanese yen"],
  "INR": ["inr", "rupee", "₹", "indian rupee"]
}
```

### Retail

#### Product Categories
```json
{
  "electronics": ["electronics", "electronic", "tech", "technology", "digital"],
  "clothing": ["clothing", "apparel", "fashion", "garments", "wear"],
  "home": ["home", "household", "domestic", "residential", "furniture"],
  "books": ["books", "literature", "publications", "reading", "texts"],
  "sports": ["sports", "athletic", "fitness", "exercise", "recreation"]
}
```

#### Order Status
```json
{
  "active": ["active", "live", "current", "ongoing", "in progress"],
  "inactive": ["inactive", "dormant", "suspended", "paused", "stopped"],
  "discontinued": ["discontinued", "ended", "terminated", "phased out", "retired"]
}
```

#### Customer Segments
```json
{
  "premium": ["premium", "vip", "gold", "platinum", "high_value"],
  "standard": ["standard", "regular", "normal", "basic", "typical"],
  "budget": ["budget", "economy", "low_cost", "affordable", "value"]
}
```

## Geographic Mappings

### Countries
```json
{
  "united_states": ["united states", "usa", "us", "america", "united states of america"],
  "united_kingdom": ["united kingdom", "uk", "britain", "great britain", "england"],
  "canada": ["canada", "ca", "canadian"],
  "australia": ["australia", "au", "australian"],
  "germany": ["germany", "de", "german", "deutschland"],
  "france": ["france", "fr", "french", "france"],
  "japan": ["japan", "jp", "japanese", "nippon"],
  "india": ["india", "in", "indian", "bharat"]
}
```

### States/Provinces
```json
{
  "california": ["california", "ca", "calif", "cal"],
  "new_york": ["new york", "ny", "ny state"],
  "texas": ["texas", "tx", "tex"],
  "florida": ["florida", "fl", "fla"],
  "ontario": ["ontario", "on", "ont"],
  "quebec": ["quebec", "qc", "que"],
  "england": ["england", "eng", "english"],
  "scotland": ["scotland", "scot", "scottish"]
}
```

## Time and Date Mappings

### Days of Week
```json
{
  "monday": ["monday", "mon", "m", "1", "first day"],
  "tuesday": ["tuesday", "tue", "tues", "t", "2", "second day"],
  "wednesday": ["wednesday", "wed", "w", "3", "third day"],
  "thursday": ["thursday", "thu", "thurs", "th", "4", "fourth day"],
  "friday": ["friday", "fri", "f", "5", "fifth day"],
  "saturday": ["saturday", "sat", "s", "6", "sixth day"],
  "sunday": ["sunday", "sun", "su", "7", "seventh day"]
}
```

### Months
```json
{
  "january": ["january", "jan", "1", "01"],
  "february": ["february", "feb", "2", "02"],
  "march": ["march", "mar", "3", "03"],
  "april": ["april", "apr", "4", "04"],
  "may": ["may", "5", "05"],
  "june": ["june", "jun", "6", "06"],
  "july": ["july", "jul", "7", "07"],
  "august": ["august", "aug", "8", "08"],
  "september": ["september", "sep", "sept", "9", "09"],
  "october": ["october", "oct", "10"],
  "november": ["november", "nov", "11"],
  "december": ["december", "dec", "12"]
}
```

## Boolean Mappings

```json
{
  "true": ["true", "t", "yes", "y", "1", "on", "enabled", "active", "positive"],
  "false": ["false", "f", "no", "n", "0", "off", "disabled", "inactive", "negative"]
}
```

## Custom Mappings

### Adding Custom Mappings

To add custom mappings for your specific domain:

1. Create a new mapping file in the `mappings/` directory
2. Follow the JSON structure above
3. Update the `CleanerPolicy` class to load your custom mappings

Example custom mapping file (`mappings/custom.json`):

```json
{
  "custom_category": {
    "canonical_value_1": ["variant1", "variant2", "variant3"],
    "canonical_value_2": ["variant4", "variant5", "variant6"]
  }
}
```

### Fuzzy Matching Thresholds

The system uses the following thresholds for fuzzy matching:

- **High confidence**: 90%+ similarity
- **Medium confidence**: 80-89% similarity  
- **Low confidence**: 70-79% similarity
- **No match**: <70% similarity

### Confidence Scoring

Confidence scores are calculated based on:

- **Exact match**: 1.0
- **Case-insensitive match**: 0.95
- **Fuzzy match (90%+)**: 0.9
- **Fuzzy match (80-89%)**: 0.8
- **Fuzzy match (70-79%)**: 0.7
- **No match**: 0.0

## Maintenance

### Updating Mappings

1. Review existing mappings for accuracy
2. Add new variants as they are discovered
3. Remove obsolete or incorrect mappings
4. Test mappings with sample data
5. Update documentation

### Validation

Regular validation of mappings:

```python
# Validate mapping completeness
def validate_mappings(mappings):
    for category, values in mappings.items():
        for canonical, variants in values.items():
            assert canonical not in variants, f"Canonical value in variants: {canonical}"
            assert len(variants) > 0, f"No variants for {canonical}"
```

### Performance Considerations

- Keep mapping dictionaries in memory for fast lookup
- Use case-insensitive matching for better coverage
- Implement caching for frequently accessed mappings
- Consider using trie data structures for large mapping sets
