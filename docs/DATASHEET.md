# Datasheet: MBIC's word-level votes, by annotator ideology

This datasheet follows the questions of Gebru et al. (2021), "Datasheets for Datasets". It describes the label set in `data/derived/`, which `scripts/export_label_set.py` builds from MBIC's raw annotations. This label set is the proposal's first deliverable.

## Motivation

**Why was it created?** The paper (Spinde et al., 2021) reduces about ten crowd judgements per sentence to one yes/no label per word: biased if at least 4 annotators highlighted it. The counts behind that label carry more information:
- **How contested a word is:** a word picked by 4 of 10 annotators is not the same as one picked by 10 of 10.
- **Who picked it:** whether liberal and conservative annotators picked it differently.

This label set keeps those counts for the project's questions about contestedness (RQ1) and whose bias the models learn (RQ2).

**Who created it?** The project team, for the course "Trustworthy Artificial Intelligence". It is derived from MBIC, which the paper's authors collected and released.

**Funding.** None. It is a course project.

## Composition

**What is in it?** Two tables of English news sentences and their words:

| File | Rows | One row per | Columns |
| --- | --- | --- | --- |
| `mbic_sentences.csv` | 1,700 | Sentence | `sentence_id`, `text`, `outlet`, `outlet_leaning`, `topic`, `annotators`, `annotators_liberal`, `annotators_centre`, `annotators_conservative`, `said_biased` |
| `mbic_word_votes.csv.gz` | 56,909 | Word | `sentence_id`, `word_index`, `word`, `start`, `end` (character span in the sentence), `votes`, `votes_liberal`, `votes_centre`, `votes_conservative` |

A word's vote share is `votes / annotators`, and a group's share is, for example, `votes_liberal / annotators_liberal`. The paper's label is `votes >= 4`.

**The sentences:**
- **Topics:** 14, from abortion and the coronavirus to student debt and white nationalism.
- **Outlets:** 8 US outlets, rated by AllSides:

| Outlet leaning | Outlets | Sentences |
| --- | --- | --- |
| Left | Alternet, HuffPost, MSNBC | 694 |
| Centre | Reuters, USA Today | 315 |
| Right | Breitbart, The Federalist, Fox News | 691 |

**The annotators:**
- **Per sentence:** 9 to 12 annotators judged each sentence, 10.5 on average. Of those, 4.7 were liberal, 3.1 centre and 2.7 conservative.
- **Groups:** these follow the paper's cut on a self-reported ideology scale from −10 to 10. Liberal is −10 to −4, centre −3 to 3, and conservative 4 to 10.
- **Totals:** across the 888 survey records in the released data, 44% are liberal, 29% centre and 26% conservative. The paper reports 784 annotators.
- **Sentence verdicts:** `said_biased` counts the annotators who called the whole sentence biased. That is 60% of judgements on average.

**The words:**
- **Splitting:** words are split as in the rest of the project: runs of letters and digits that keep inner apostrophes and hyphens ("isn't", "pro-life").
- **Labels:** 3,235 words (5.7%) have at least 4 votes, and 16,955 (30%) were highlighted by at least one annotator. The paper's 9.5% is over its own list of 31,794 words, which leaves out stop words.

**Is it complete?** Every sentence and every word of MBIC's 1,700 sentences is included. Every annotator falls in one of the three groups.

**Does it identify anyone?** No. Only counts per sentence and per group are released, never an individual's judgements, demographics or survey ID. The sentence texts are published news. MBIC's raw files, which do have pseudonymous survey IDs, are in `data/raw/`. The MTurk IDs, some of which contained real names, were removed from those files earlier in the project.

**Is anything sensitive?** Political ideology appears only as a group label behind the counts.

**How reliable is it?** The rebuilt "at least 4 votes" label agrees with the paper's own label on 98.3% of the words that both lists share (Cohen's κ 0.906; `docs/SOFT_LABEL_MODELS.md`).

## Collection process

**How was the data collected?** By MBIC's authors, on Amazon Mechanical Turk. Each annotator read news sentences and, for each one, said whether it was biased and highlighted the words that made it so. A survey recorded their political ideology on the −10 to 10 scale, along with other questions. The sentences come from news articles of 2017 to 2020 (Spinde et al., 2021).

**How was this label set derived?**
1. **Highlights to words:** each annotator's free-text highlights are matched to the sentence's words (`src/data/annotations.py`, `match_highlights`). A highlighted phrase is matched as a run of consecutive words. Failing that, its words are matched one by one. A repeated word marks its first occurrence.
2. **Groups:** each annotator is put in a group from their ideology score (`ideology_groups`).
3. **Counts:** highlights are counted per word, overall and per group.

**Who did the work?** The MBIC authors collected the data. The project team wrote and ran the derivation.

## Preprocessing

- **Kept raw:** `data/raw/annotations.xlsx` and `data/raw/annotators.csv`, as released, apart from the removed MTurk IDs.
- **Known imprecision:** the first-occurrence rule can attach a highlight of a repeated word to the wrong occurrence. Highlights that match no word of the sentence are dropped.

## Uses

**Used for, in this project:**
- **Training:** M2, trained on vote shares; M3, trained on group shares; and the distilled student on the board.
- **Analysis:** the perspective analysis (`docs/PERSPECTIVE_ANALYSIS.md`), the audit of whose readers the models follow (`docs/AUDIT.md`), and the abstention analysis (`docs/SOFT_LABEL_MODELS.md`).

**Other suitable uses.** Research on learning with disagreement, and on how readers' politics shape what they call biased.

**Not suitable for:**
- **Inferring individuals' politics.**
- **Claims about what liberals or conservatives think in general.** The groups are small: a sentence has 4.7 liberal and 2.7 conservative annotators on average, and sometimes none of a group. They are self-selected US crowd workers, not representative samples.
- **Ground truth beyond MBIC's annotators.** Experts mark about half as many words as this crowd (`docs/BABE_TEST.md`).

**Risks.** A single sentence's group shares are noisy. Differences between groups are meaningful only aggregated over many sentences. `scripts/audit_group_balance.py` shows how to compare groups of equal size.

## Distribution

The label set lives in this repository, in `data/derived/`. `python scripts/export_label_set.py` rebuilds it from `data/raw/`; the output is the same each time. It is derived from MBIC, and MBIC's terms of use, set by its authors, apply to it.

## Maintenance

The project team maintains it for the length of the course. Errors can be reported as issues on the repository. If MBIC's raw files or the matching code change, rerunning the script regenerates both tables.

## References

- Spinde, T., Rudnitckaia, L., Mitrović, J., Hamborg, F., Granitzer, M., Gipp, B., and Donnay, K. (2021). Automated identification of bias inducing words in news articles using linguistic and context-oriented features. *Information Processing and Management*, 58, 102505.
- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
