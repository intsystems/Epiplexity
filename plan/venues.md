# Venues

Checked on 23.09.2026. "Confirmed" means the date was taken from the official page, linked in the
same row. "Estimated" means the date is the previous edition's, because the 2027 page has not
published it yet. All deadlines are 23:59 AoE unless noted otherwise.

## Recommendation

**Primary: CPAL 2027, Proceedings Track** (Conference on Parsimony and Learning;
[call](https://cpal.cc/call_for_papers/)).

| Item | Detail |
|---|---|
| Abstract registration | **23.11.2026** (confirmed) |
| Paper | **05.12.2026** (confirmed) |
| Rebuttal | 11–17.01.2027 |
| Notification | 01.02.2027 |
| Camera-ready | 12.02.2027 |
| Format | 9 pages of main text in the CPAL template; references and appendix unlimited; double-blind |
| Proceedings | archival, published in PMLR |
| Conference | 23–26.03.2027, Tokyo |
| Presentation | one author must attend in person; Japanese visas for Russian citizens take about 5–10 working days |

The call lists "information-theoretic, minimum-description-length, and compression-based views of
learning" among its core topics. The deadline matches the end of both courses. CPAL is young (4th
edition) and has no CORE rank. Acceptance rates are not published; CPAL 2026 accepted 54
proceedings papers.

**Optional companion: ICLR 2027 Blogpost Track** (estimated early December; the 2026 deadline was
07.12.2025; [2026 call](https://iclr-blogposts.github.io/2026/call/)).
- It is a reviewed blog post, not a proceedings paper. A distinct post on the pitfalls (floor,
  seeds, multiple epochs) does not use up the paper.
- Attendance is optional.
- The BMM blog post (Member 3) can be its first draft.

## Other venues in the window

| Venue | Deadline | Format | Conference | Fit and constraints |
|---|---|---|---|---|
| [ESANN 2027](https://www.esann.org/) | 18.11.2026 (confirmed) | 6 pages; archival (Scopus); single-blind | Bruges, 21–23.04.2027; online presentation by video allowed | Good fit for a small, careful study. Needs the paper two weeks before CPAL. No other submission is allowed until the decision on 22.01 |
| [ICML 2027](https://icml.cc/) | late January 2027 (estimated; 2026: 23.01 abstract, 28.01 paper) | 8 pages; archival; double-blind | South America; author attendance was optional in 2026 | A* venue with 26.6% acceptance in 2026. Needs a clear methodological result, e.g. Bayesian and training-based estimators disagreeing in a predictable way |
| [ICLR 2027 workshops](https://iclr.cc/Conferences/2027/CallForWorkshops) | about 01.02.2027 (suggested date, confirmed); list published 29.11.2026 | usually 4 pages; non-archival | San Francisco, 29–30.04.2027 | Best series: the Science of Deep Learning workshop (Sci4DL). Choose a workshop that accepts posters without the authors present (US visa) |
| [TMLR](https://jmlr.org/tmlr/) | rolling; expected pause 02.12.2026–05.01.2027 | any length; journal | none; J2C certification allows a poster at ICML or NeurIPS | Reviews for correctness over novelty; has a reproducibility certification. Safest fallback |
| [ISIT 2027](https://2027.ieee-isit.org/) | 10.01.2027 (confirmed) | 5 pages; archival (IEEE); not anonymous | Sorrento, 27.06–02.07.2027; strictly in person | Only if the MDL and Bayesian-code part becomes the core. Needs a Schengen visa |
| [IJCNN 2027](https://ijcnn.org/2027) | 31.01.2027 (confirmed) | 6 pages; archival | Cape Town, 14–18.06.2027; visa-free for Russian citizens; video presentation allowed | Broad neural-network venue with lower prestige. Cannot overlap with CPAL review |
| UAI 2027 / ProbML 2027 (formerly AABI) | late February / March 2027 (estimated) | 8–9 pages archival; ProbML also has a non-archival workshop track | TBA | Best fit for the Bayesian half. Just after the window |

Closed or unsuitable:
- **Deadline passed:** ICLR 2027 main (25.09), AISTATS 2027 (06.10), all NeurIPS 2026 workshops,
  MLRC 2026, AI Journey / Doklady Mathematics (August).
- **No online presentation, and a Canadian visa is uncertain:** AAAI-27 workshops (20.11,
  Montréal).
- **Local and low-stakes:** the 69th MIPT conference takes 1–2 page abstracts in February–March
  2027.

## Rules that constrain the plan

1. **One archival venue at a time.** CPAL Proceedings, ESANN, ISIT and IJCNN each forbid parallel
   archival submission. Non-archival venues (ICLR workshops, the CPAL Spotlight track, the ICLR
   blog post) can run in parallel with anything.
2. **An archival CPAL paper cannot go to ICML afterwards.** Only a substantially extended version
   could.
   - To keep ICML 2027 open, use the non-archival route instead: the ICLR blog post in December,
     ICML in late January, and in parallel an ICLR workshop and the CPAL Spotlight track (18.01).
   - Decide between the two routes by 19.10.
3. **Double-blind review.**
   - The public repository shows the names of the authors. Link code through an anonymised copy
     (e.g. anonymous.4open.science).
   - Check the CPAL policy on preprints before the named Habr post (BMM, due 24.11) goes public.
4. **Two templates.** The R&D course asks for the arXiv style; CPAL asks for its own. Keep one
   source in `paper/` with two preambles.

## Fallback chain

- **CPAL rejects on 01.02.2027:** an ICLR 2027 workshop (about 01.02). Then UAI or ProbML 2027,
  or TMLR.
- **The paper is ready by 18.11:** ESANN instead of CPAL. If it is rejected on 22.01, go to IJCNN
  (31.01).

## To re-check before committing

- The ICLR 2027 Blogpost dates (the page is not live yet).
- The ICML 2027 dates and city.
- The ICLR 2027 workshop list (29.11).
- The ISIT 2027 page limit.
