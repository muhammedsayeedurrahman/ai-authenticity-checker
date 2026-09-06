# ProofyX Investor Pitch Deck

**Build:** `python scripts/make_pitch_figures.py && python scripts/build_investor_deck.py`
**Output:** `ProofyX_Investor_Deck.pptx` (15 slides, 16:9) and `.pdf`

Every chart is regenerated from `scripts/make_pitch_figures.py`. If an investor
questions a number, the source is on slide 15 and the full research is in
`docs/PROOFYX_COMPLETE_ANALYSIS.md`.

## Structure

Follows the ten slides investors screen for, plus a Why-Now slide, a business
model slide, and a sources appendix.

| # | Slide | Investor question it answers |
|---|-------|------------------------------|
| 1 | Title / one-line intro | What does this company do? |
| 2 | Problem | Is this a real, acute pain? |
| 3 | Why now | Why does this get bought *this year*? |
| 4 | Solution | What is the unique value proposition? |
| 5 | Product | What have you actually built? |
| 6 | Validation | What evidence do you have? |
| 7 | Market | How big can this get? |
| 8 | TAM / SAM / SOM | What share are you underwriting? |
| 9 | Competition | Who else is doing this, and why do you win? |
| 10 | Business model | How do you make money? |
| 11 | Financial projections | What is the trajectory? |
| 12 | Team | Are you the right people? |
| 13 | The ask | What do you want, and what will it buy? |
| 14 | Contact | How do I reach you? |
| 15 | Sources (appendix) | Where did every number come from? |

Speaker notes with timings are on every slide.

## Before this deck leaves the room

1. **Slide 14 — replace both placeholders.** `REPLACE-WITH-REAL@EMAIL` and
   `REPLACE-WITH-REAL-NUMBER`. The previous version of this deck went out with
   Canva's `reallygreatsite.com` placeholders still in it; that alone ends a
   conversation with an investor.
2. **Slide 5 — swap the concept render for a real screenshot.** The product is
   running; use it. The caption currently says the image is a concept render, so
   the deck is honest as-is, but a real screenshot is far stronger.
3. **Slide 12 — confirm the roles.** Roles were inferred from commit history and
   file ownership. Correct them, and add LinkedIn URLs or prior credentials if
   any team member has them.
4. **Slide 13 — confirm the raise range.** `$1.5M-$2.5M` and the 40/25/20/15
   split are a starting position, not a modelled number. Sanity-check against an
   actual 18-24 month burn plan before quoting it.
5. **Slide 11 — have the spreadsheet ready.** The chart is a projection. If an
   investor asks for the model, do not improvise; send the file.

## Claims that are deliberately hedged

These are labelled on the slides. Do not "upgrade" them in a later edit:

- **82.5% accuracy** is CorefakeNet on 332 held-out samples. It is not the
  ensemble, and it is not a public-benchmark number.
- **95%+ ensemble accuracy** is a *target*, shown as "In progress" on slide 6.
  The fusion calibration bug is disclosed rather than hidden.
- **97.9% audio accuracy** is the upstream Wav2Vec2 model card, not our
  measurement. Slide 6 says so.
- **Zero paying customers** is stated on slide 6. Volunteering this is what makes
  the other rows credible in diligence.
- **Financial projections** are labelled projections on both the chart and the
  headline.

The previous deck claimed "industry-leading accuracy" with nothing behind it.
That claim is gone.

## Assets

`assets/` holds the brand elements lifted from the original Canva deck (logo and
prism render, both re-extracted with their alpha channels intact) and the five
generated figures. Regenerate figures with `make_pitch_figures.py`; do not edit
the PNGs by hand.
