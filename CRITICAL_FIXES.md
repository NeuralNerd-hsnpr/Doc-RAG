# Critical RAG System Fixes

## Issues Identified from Evaluation

### 1. Hallucination of Omission ❌
**Problem**: Agent claimed document doesn't address topics that are actually central chapters
- Example: "The document does not address... U.S. Dollar" when document has section "Is this the downfall of the U.S. dollar?"

**Root Cause**: 
- Retrieval too restrictive (threshold too high)
- Not retrieving enough chunks for general queries
- Chunking missing section headers

### 2. Temporal Confusion ❌
**Problem**: Answering retrospective questions with forward-looking text
- Example: "Which forecasted themes played out?" answered with forecasts instead of clarifying document is forward-looking

**Root Cause**: No temporal awareness in prompts

### 3. Formatting Artifacts ❌
**Problem**: Special tokens leaking through ([/ASS], <s>, etc.)
**Root Cause**: No post-processing to clean LLM output

## Fixes Implemented

### 1. Enhanced Retrieval for General Queries ✅

**Changes**:
- Detect general/topic questions automatically
- Increase `top_k` by 2x-4x for general questions
- Lower adaptive threshold for general queries (0.7x base threshold, min 0.15)
- Retrieve more chunks even if below threshold for better coverage

**Code**:
```python
is_general = any(word in query_lower for word in [
    "topic", "theme", "subject", "about", "main", "overview", 
    "summary", "what is", "discuss", "cover", "include", "all"
])

if is_general:
    expanded_top_k = int(top_k * 4)  # 4x expansion
    adaptive_threshold = max(config.SIMILARITY_THRESHOLD * 0.7, 0.15)
```

### 2. Negative Answer Guardrails ✅

**Changes**:
- System prompt explicitly prohibits claiming "document does not contain"
- Instructions to say "I could not find in retrieved sections" instead
- Post-processing detects and warns about negative claims
- Better handling of missing vs not found

**Prompt Addition**:
```
NEGATIVE ANSWER RULE: If you cannot find information in the RETRIEVED sections, 
say "I could not find information about [topic] in the retrieved sections" - 
DO NOT claim "the document does not address" or "the document does not contain" 
unless you are absolutely certain after reviewing ALL retrieved sections
```

### 3. Special Token Cleaning ✅

**Changes**:
- Post-processing regex to remove common special tokens
- Handles: [/ASS], [ASS], </s>, <s>, [INST], [|assistant|], etc.

**Code**:
```python
special_tokens = [
    r'\[/ASS\]', r'\[ASS\]', r'</s>', r'<s>', r'\[INST\]', r'\[/INST\]',
    r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
    r'### Assistant:', r'### User:', r'### System:',
    r'<|endoftext|>', r'<|end|>'
]
```

### 4. Improved Section Header Detection ✅

**Changes**:
- Better regex patterns for section headers
- Detects question-style headers ("Is this...?")
- Preserves headers in chunks
- Better section splitting

**Patterns Added**:
- Question patterns: "is this", "will...?", "what...?"
- Better capitalization detection
- Longer header support (up to 150 chars)

### 5. Temporal Awareness ✅

**Changes**:
- Detect retrospective questions
- Special prompt for temporal mismatch
- Clear instructions to identify forward-looking vs retrospective documents

**Detection**:
```python
is_retrospective = any(word in query_lower for word in [
    "played out", "happened", "occurred", "was", "were", 
    "did", "past", "previous", "retrospective", "review"
])
```

**Prompt**:
```
IMPORTANT: This question asks about past events. However, this document 
appears to be a forward-looking forecast. If the document makes forecasts 
rather than reviewing past events, clearly state: "This document contains 
forward-looking forecasts rather than a review of past events."
```

### 6. Comprehensive Coverage for Topic Questions ✅

**Changes**:
- Increased word limit for general questions (400 words)
- Explicit instruction to mention ALL topics found
- Better synthesis of multiple sections

**Prompt Addition**:
```
- For topic questions, list ALL major topics/themes found in retrieved sections
- Be comprehensive - if asking about topics/themes, mention ALL major topics
```

## Configuration Changes

```python
# config.py
RETRIEVAL_TOP_K: int = 10  # Base retrieval
SIMILARITY_THRESHOLD: float = 0.3  # Lower threshold
MAX_TOKENS_RESPONSE: int = 800  # Prevent repetition
TEMPERATURE: float = 0.3  # Reduce repetition
```

## Expected Improvements

1. ✅ **Better Coverage**: General queries retrieve 2-4x more chunks
2. ✅ **No False Negatives**: Won't claim document doesn't contain when it does
3. ✅ **Clean Output**: No special tokens in responses
4. ✅ **Temporal Clarity**: Recognizes forward-looking vs retrospective
5. ✅ **Better Headers**: Section headers preserved in chunks
6. ✅ **Comprehensive Answers**: Lists all topics for general questions

## Testing

Test with the problematic questions:

```bash
python main.py
```

1. "what is the topic of the document"
   - Should retrieve 20+ chunks
   - Should list ALL major topics
   - Should not miss sections

2. "Which forecasted themes played out as expected?"
   - Should detect retrospective question
   - Should clarify document is forward-looking
   - Should list forecasts mentioned

3. "What does the document say about the U.S. Dollar?"
   - Should retrieve U.S. Dollar section
   - Should not claim document doesn't address it
   - Should cite relevant sections

## Monitoring

Check logs for:
- `[RETRIEVAL] General query detected` - Shows expanded retrieval
- `[RETRIEVAL] Matches after threshold` - Shows filtering
- `[POST_PROCESS] Found X negative claims` - Warns about potential hallucinations
- `[RETRIEVAL] Retrospective question detected` - Shows temporal detection

## Next Steps

If issues persist:
1. Check retrieval logs - are chunks being retrieved?
2. Verify chunking - are section headers preserved?
3. Check similarity scores - are they too low?
4. Review post-processing - are duplicates being removed?

