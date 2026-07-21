# Paper compilation note

`main.tex` and the generated result fragments contain the final gated-v2
interpretation. Recompile `main.tex` from inside this `paper/` directory so
that the eight figures in `../figures/` are resolved correctly.

The compiled PDF must visibly contain:

- five stations, five seeds, and a 30-epoch maximum;
- 0/25 wins against the strongest comparator;
- pooled MAE 3.708 (hybrid), 4.078 (TimeMixer), and 3.271 (LSTM);
- eight main figures, with only two composite EDA figures;
- no SOTA or best-model claim for the hybrid.

The included `main.pdf` was compiled before the final prose revision. Treat
`main.tex` as authoritative and regenerate the PDF before submission.
