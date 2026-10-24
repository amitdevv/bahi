# Bahi

Bahi turns messy Indian money documents into verified ledgers, entirely on your own hardware.

The first track is **handwritten shagun registers**: Devanagari entries written in a printed diary, photographed on a phone. The second track (later) is **bank statement PDFs**, with personal data redacted before any model sees them.

## Principles

1. **Private by design.** Documents never leave your own machines. Any model call sees redacted data, or runs locally.
2. **Every number is checked.** Page totals must equal the sum of their rows, and bank running balances must add up on every row. A mismatch points to the exact row to fix.
3. **Built for Indian documents.** Devanagari handwriting, ditto marks (〃), Vikram Samvat dates and UPI narrations.

## Register pipeline

1. **Orient and crop.** Fix 180° photos, find the page and dewarp it.
2. **Separate ink.** Cluster colours in Lab space to keep the blue pen and drop the printed diary template, the watermark and show-through from the back of the page.
3. **Lines and rows.** Detect the ruled lines and cut one image per entry.
4. **Recognize.** Handwriting recognition with constrained decoding (an amount grammar, plus a trie of known names and villages).
5. **Reason across rows.** Resolve ditto marks, struck-out entries and cash-plus-gift rows.
6. **Verify.** Check each page total and send flagged rows to a quick human review.

## Layout

```
src/bahi/   pipeline code
data/       raw photos and ground truth (gitignored; private, server only)
out/        intermediate images and results (gitignored)
```

## Running

Heavy work runs on the compute server:

```bash
uv sync
uv run python -m bahi.preprocess data/raw out/preprocess
```
