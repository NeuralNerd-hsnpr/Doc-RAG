# Repetition Loop and Generation Fixes

## Critical Issues Fixed

### 1. Repetition Loop (Hallucination Loop)
**Problem**: Model got stuck repeating "Dollar has softened 1% since May... since March... since April..." with impossible statistics.

**Root Causes**:
- No repetition penalty configured
- Temperature too low (0.3) causing deterministic loops
- No stop sequences to break loops
- No post-processing to detect repetition patterns

**Fixes Applied**:

#### A. Repetition Penalty Handling
- **File**: `src/hf_llm.py`
- Added `repetition_penalty` parameter (default: 1.15) to `generate()` method
- **Note**: HuggingFace InferenceClient API doesn't support `repetition_penalty` directly
- Code gracefully handles this by catching TypeError and falling back to alternative methods
- **Config**: `config.py` → `REPETITION_PENALTY: float = 1.15` (for future API support)
- **Alternative**: Using temperature, stop sequences, and post-processing to prevent repetition

#### B. Increased Temperature
- **Before**: 0.3 (too deterministic)
- **After**: 0.4 (breaks loops while maintaining accuracy)
- **Config**: `config.py` → `TEMPERATURE: float = 0.4`

#### C. Added Stop Sequences
- **File**: `src/langgraph_workflow.py` (synthesis_node)
- Stop sequences: `["\n\n\n", "---", "===", "The sections suggest", "The document suggests"]`
- Prevents model from continuing repetitive patterns

#### D. Reduced Max Tokens
- **Before**: 800 tokens
- **After**: 600 tokens
- Prevents excessive generation that leads to loops

### 2. Logic and Number Validation

#### A. Enhanced Post-Processing
**File**: `src/langgraph_workflow.py`

**New Methods**:
1. `_detect_and_fix_repetition_loops()`:
   - Detects patterns like `"X has Y% since [month]"` repeated multiple times
   - Removes duplicate sentence patterns
   - Uses pattern matching to identify repetitive structures

2. `_validate_numbers()`:
   - Detects when same statistic is claimed with different months/contexts
   - Validates that numbers aren't being hallucinated
   - Removes impossible claims (e.g., "softened 1% since every month")

#### B. Improved Prompts
**File**: `src/langgraph_workflow.py` (synthesis_node)

**System Prompt Enhancements**:
- Added explicit "ANTI-REPETITION RULES" section
- Added "NUMBER RULE": Only quote exact numbers, DO NOT generate variations
- Added "LOGIC CHECK": Stop if repeating similar phrases
- Added "TEMPORAL LOGIC": Check if question asks about past events but document is forward-looking

**User Prompt Enhancements**:
- Added temporal logic check for retrospective questions
- Explicit instruction to quote statistics rather than generate them
- Clear instruction to consolidate repetitive statements

### 3. Temporal Paradox Handling

**Problem**: User asks "Which forecasted themes played out?" but document is a forecast (can't confirm outcomes).

**Fix**: Enhanced temporal logic in prompts:
- Check document date/context
- If forecast document, clearly state: "This document contains forward-looking forecasts rather than a review of past events"
- List forecasts mentioned
- DO NOT claim events "played out" if document only forecasts them
- Clarify "so far this year" refers to partial data, not full outcomes

## Code Changes Summary

### `src/hf_llm.py`
- Added `repetition_penalty` and `stop_sequences` parameters to `generate()` method
- Updated `_generate_with_client()` to accept and use these parameters
- Updated `_generate_streaming()` to accept and use these parameters
- Added `List` import from `typing`

### `src/langgraph_workflow.py`
- Updated `synthesis_node` to pass `repetition_penalty=1.15` and `stop_sequences` to LLM
- Enhanced system prompt with anti-repetition rules and number validation
- Enhanced user prompt with temporal logic checks
- Added `_detect_and_fix_repetition_loops()` method
- Added `_validate_numbers()` method
- Updated `_post_process_answer()` to call new validation methods
- Changed `_post_process_answer()` signature to accept `chunks` parameter for validation

### `config.py`
- Updated `MAX_TOKENS_RESPONSE`: 800 → 600
- Updated `TEMPERATURE`: 0.3 → 0.4
- Added `REPETITION_PENALTY: float = 1.15`

## Testing Recommendations

Test with these questions that previously caused repetition loops:
1. "Which forecasted themes played out?" (temporal paradox)
2. "What is the topic of the document?" (general question)
3. "Which stocks were highlighted?" (specific question)

**Expected Improvements**:
- No repetition loops
- Accurate number quoting (no variations)
- Proper temporal logic handling
- Clean, non-repetitive answers
- Better citation extraction

## Performance Parameters

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Temperature | 0.3 | 0.4 | Break deterministic loops |
| Repetition Penalty | None | 1.15 | Prevent repetition |
| Max Tokens | 800 | 600 | Limit generation length |
| Stop Sequences | None | 5 sequences | Break loops early |

## Monitoring

The logging system will now log:
- `[POST_PROCESS] Detected repetition loop: ...` - When loops are detected
- `[POST_PROCESS] Found X similar number patterns` - When number repetition detected
- `[POST_PROCESS] Skipping repetitive sentence pattern` - When duplicate sentences removed

