# References checked 2026-11-20

Source of each row: the arXiv abstract page (all) and, for NCM and LCM, the full HTML text, read through a page-fetching helper; LLM-JEPA, VL-JEPA, HDLM, VAR and AURORA-LM through a web search that returned the arXiv records. The abstract pages were read by a helper agent, so quotes below are short and should be re-read once before submission. Nothing here is a number from our own runs.

| Key in draft | arXiv | Title (first author, v1 date) | What was checked | Result |
| --- | --- | --- | --- | --- |
| NCM | 2610.14162 | Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation (Kumbong and 8 others, 9 Oct 2026) | architecture, training-token claim, single decode, future work, TinyStories reference numbers, corruption recipe | matches the draft (below) |
| LCM | 2412.08821 | Large Concept Models: Language Modeling in a Sentence Representation Space (LCM team, 11 Dec 2024) | Base-LCM versus diffusion variants | matches: Sec. 2.4.1, "diffusion-based methods give clearly better results compared to all other models"; caveat: Base-LCM has the lowest l2 distance |
| H-JEPA | 2610.06805 | H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning (Zhang and 4 others, 5 Oct 2026) | hierarchy of action-conditioned JEPAs | matches; scope is visual planning |
| HiLP | 2608.05806 | Hierarchical Latent Prediction for Language Models (Shi and 4 others, 6 Aug 2026) | auxiliary higher-level latent alongside next-token prediction | partial: the abstract does not use the JEPA name, so the draft no longer calls it JEPA-style |
| Korchinski et al. | 2605.27734 | Learn from your own latents and not from tokens: A sample-complexity theory (Korchinski, Favero, Wyart, 26 May 2026) | "explicit stacking such as H-JEPA is largely redundant" | matches; the data model is a probabilistic context-free grammar, the abstract hedges with "suggests" |
| Balestriero and LeCun | 2610.39302 | Proving and Improving the Platonic Representation Hypothesis with JEPAs (29 Oct 2026) | same representation up to one rotation for good views; "42 times fewer training samples than next-token prediction" | matches; the 42x figure measures alignment with DINOv2 and CLIP encoders, not generation (stays out of the generation claim) |
| LLM-JEPA | 2509.14252 | LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures (Huang, LeCun, Balestriero, 11 Sep 2025; ICLR 2026) | JEPA-style loss added to the next-token objective | matches |
| VL-JEPA | 2512.10942 | VL-JEPA: Joint Embedding Predictive Architecture for Vision-language (Chen, Shukor and others, 11 Dec 2025; ICLR 2026) | predicts continuous text embeddings, a lightweight decoder maps them to text | matches |
| HDLM | 2510.08632 | Next Semantic Scale Prediction via Hierarchical Diffusion Language Models (Zhou and 6 others, 8 Oct 2025; NeurIPS 2025) | tiered vocabulary, coarse-to-fine reverse process | matches the draft's one-line description |
| VAR | 2404.02905 | Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction (Tian, Jiang, Yuan, Peng, Wang; NeurIPS 2024 oral) | next-scale prediction with a multi-scale VQ-VAE | matches |
| AURORA-LM | 2608.02602 | AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling (Aug 2026) | continuous latent sequence, block-causal diffusion transformer, flow matching | matches; note it uses flow matching on continuous latents, relevant to the novelty claim |

## NCM details used by the draft and SPEC (full text, HTML v1)
- Training tokens: "approximately 4-5x fewer training tokens than strong diffusion baselines on LM1B and OpenWebText" (Sec. 1, contribution iii; Sec. 4.6 gives about 4.9x against 66B on LM1B). Their totals include tokenizer training (LM1B: 13.36B) and repeated passes; they measure token exposure, not FLOPs (App. C.2). Our own accounting must say the same (SPEC 6.2, Q14).
- One decode: Sec. 3.2 and Algorithm 2 decode the final cumulative reconstruction once.
- Future work: App. I names end-to-end training of the tokenizer with the generator.
- TinyStories references: Table 8 (App. E.2) real text 9.67 perplexity and 3.96 entropy, NCM with 12 steps 15.41 and 3.98; Table 1 reconstruction accuracy 99.98 percent; GPT-2-large scorer. These are the SPEC section 1 numbers.
- Ablation budget: App. F.3 "88.6M NCM transformer ... 10,000 steps with a batch size of 1,024" and tokenizers 40k steps (App. F.1.1), as in our dev config. The main TinyStories configs differ (generator 50,000 updates at batch 2,048).
- Corruption: Sec. 3.4 and F.3 replace some input codes with entries sampled uniformly from the same level's codebook at a rate that decreases linearly across levels; targets stay clean; the selected recipe has mean rate 0.50 (App. F.2.1); Figure 13 plots perplexity against the mean rate, and the text does not give the swept values. SPEC 5.4 says NCM found moderate mean rates best; the text only supports "selected 0.50".
- Difference to note in our setup (from the helper's quote of the NCM sentence above, re-read App. F.3 before relying on it): NCM replaces codes with entries sampled uniformly from the level's codebook, while our `swap` mode takes the code of another position in the batch at the same level (SPEC 5.4), which is a draw from the data distribution, not uniform over the codebook. SPEC marks the corruption as a [DECISION]; recorded as QUESTIONS Q15 for the researcher.

## Still open
- Bibliography keys and venue entries (`.bib` is not written yet).
- Novelty search shortly before submission. Seen while checking, to read then: "Continuous Latent Diffusion Language Model" (arXiv 2605.06548), "Designing Transformers for Continuous Diffusion Language Models" (arXiv 2610.22330), "Learning Hierarchical Forward Processes For Discrete Diffusion Language Models" (NeurIPS 2026 poster), and follow-ups from the NCM authors.
- Choose the workshop after the Nov 29 decisions.
