# COMP 664 project update for November 12

**Question.** Does stronger crop and color augmentation improve transfer from controlled tomato-leaf photographs to field images?

**New work completed.** We independently trained ordinary and strong-augmentation MobileNetV3-Small models with fresh per-image views in each of eight epochs, under three paired seeds (664, 665, 666). All six runs used the same official source split, initialization within each seed, shuffled sample order and optimization budget. All checkpoint choices used source validation and were locked before either test was scored. The original two-fixed-view experiment is retained separately.

**Results.** Mean field accuracy was 18.78 ± 1.10% for ordinary and 20.13 ± 0.74% for strong augmentation; mean field macro-F1 was 7.76 ± 0.83% and 9.16 ± 0.73% (sample SD over seeds). Field accuracy changes were −0.27, +3.10 and +1.21 percentage points. Thus the average gain is modest and seed-dependent. Clean accuracy was 97.26 ± 0.16% versus 96.66 ± 0.58%.

**Evidence.** Six checkpoints, all 48 epoch records, per-image test probabilities, paired-budget/hash audits, per-class gaps, conditional bootstrap intervals and an executed Notebook are included. The final report uses LaTeX and the slides use a plain academic layout.

**Remaining limits.** Validation omits mosaic virus; 4,280 source-training and 1,060 clean-test images lack documented leaf IDs; exact-byte checks do not remove all near-duplicates; only two related field spider-mite entries exist. Three seeds do not establish broad deployment reliability. The follow-up was designed after inspecting legacy test results and remains exploratory evidence on that benchmark.

**Next research priority.** Independent, reliably labeled field samples with a predefined target-development/test separation, rather than adding unrelated architectures.
