```python
STRONG_YES = [
    # Direct usage / adaptation / implementation of MAE or masked autoencoding
    r'\b(masked\s+autoencod(?:er|ers?|ing)|MAE(?:s)?|M[Aa]E(?:-based|[- ]?style)?)\b.*\b(pretrain(?:ing)?|fine[- ]tune(?:d|ing)?|adapt(?:ed|ation)|extend(?:ed|s|ing)|apply(?:ied|ing)?|use(?:d|s)?|propos(?:e|ed|ing)|implement(?:ed|ation)|framework|model|architecture)\b',
    r'\b(pretrain(?:ing)?|fine[- ]tune(?:d|ing)?|adapt(?:ed|ation)|extend(?:ed|s|ing)|apply(?:ied|ing)?|use(?:d|s)?|propos(?:e|ed|ing)|implement(?:ed|ation)|framework|model|architecture)\b.*\b(masked\s+autoencod(?:er|ers?|ing)|MAE(?:s)?|M[Aa]E(?:-based|[- ]?style)?)\b',

    # Explicit reconstruction / self-supervised masked modeling with MAE phrasing
    r'\bself[- ]supervised\s+masked\s+auto[- ]?encod(?:ing|er|ers?)\b',
    r'\bmasked\s+auto[- ]?encod(?:ing|er|ers?)\b.*\b(reconstruct(?:ion|s|ed|ing)|mask(?:ed|ing)|pretrain(?:ing)?|representation(?:s)?|latent(?:s)?|downstream)\b',

    # Paper title / abstract signals that the method is the core contribution
    r'\b(MAE|masked\s+autoencod(?:er|ers?))\b.*\b(based|framework|model|method|architecture|pretraining|pre[- ]training|fine[- ]tuning)\b',
    r'\b(based|framework|model|method|architecture|pretraining|pre[- ]training|fine[- ]tuning)\b.*\b(MAE|masked\s+autoencod(?:er|ers?))\b',

    # Strong benchmark / comparison language when paired with MAE
    r'\b(MAE|masked\s+autoencod(?:er|ers?))\b.*\b(benchmark(?:ing|s)?|compare(?:d|s|ing)?|state[- ]of[- ]the[- ]art|SOTA|outperform(?:s|ed|ing)?|evaluation|ablation)\b',
    r'\b(benchmark(?:ing|s)?|compare(?:d|s|ing)?|state[- ]of[- ]the[- ]art|SOTA|outperform(?:s|ed|ing)?|evaluation|ablation)\b.*\b(MAE|masked\s+autoencod(?:er|ers?))\b',

    # Domain-specific but clearly MAE-derived named methods in the context
    r'\b(?:hBehaveMAE|RamanMAE|PiMAE|scMMAE|SeqMAE|Echo-Vision-FM|ViT-MAE|Mask(?:ed)?\s*Autoencoder(?:-based)?|Masked\s+Image\s+Model(?:ing|led|ing))\b',
]

WEAK_YES = [
    # Title/abstract mentions MAE as the explicit basis for a new method
    r'\b(masked\s+autoencod(?:er|ers?)|MAE(?:s)?)\b',
    r'\b(masked\s+autoencod(?:er|ers?)|MAE(?:s)?)\b.*\b(new|novel|propos(?:e|ed|ing)|framework|model|method|architecture|pipeline|system)\b',
    r'\b(new|novel|propos(?:e|ed|ing)|framework|model|method|architecture|pipeline|system)\b.*\b(masked\s+autoencod(?:er|ers?)|MAE(?:s)?)\b',

    # Common usage cues in the context
    r'\b(pretrain(?:ing)?|pre[- ]training|fine[- ]tune(?:d|ing)?|transfer(?:ability|red)?|downstream tasks?|representation(?:s)?|latent(?:s)?|reconstruct(?:ion|s|ed|ing)|mask(?:ed|ing))\b',
    r'\b(state[- ]of[- ]the[- ]art|SOTA|benchmark(?:ing|s)?|compare(?:d|s|ing)?|evaluation|ablation|outperform(?:s|ed|ing)?)\b',

    # Title phrases from the supplied target vocabulary seeds
    r'\bmasked\s+autoencoders?\b',
    r'\bautoencoders?\s+are\s+scalable\s+vision\s+learners\b',
    r'\bvision\s+learners?\b',
    r'\bmasked\s+image\s+model(?:ing|led|ing)\b',
]

BACKGROUND_ONLY = [
    # Generic background-only cues around self-supervised/vision without target evidence
    r'\brelated\s+work\b',
    r'\bbackground\b',
    r'\bself[- ]supervised\s+learning\b',
    r'\bvision\s+transformer(?:s)?\b',
    r'\bmasked\s+image\s+model(?:ing|led|ing)\b.*\b(back(?:ground)?|pretrain(?:ed|ing)?|benchmark(?:s|ing)?)\b',
    r'\bpretrained\b.*\b(vision\s+transformer|ViT|self[- ]supervised)\b',
    r'\b(compare(?:d|s|ing)?\s+with|based\s+on)\b.*\b(baseline(?:s)?|existing\s+methods?|prior\s+work)\b',
]

AMBIGUOUS = [
    # Missing title/abstract or insufficiently informative citation role
    r'^\s*abstract\s+missing\s*$',
    r'\bno\s+abstract\b',
    r'\bmissing\b',
    r'\bto\s+be\s+defined\b',
    r'\bunclear\b',
    r'\bambiguous\b',
    r'\bpossible\b',
    r'\bmay\s+be\b',
    r'\bcould\s+be\b',
    r'\bappears?\s+to\s+be\b',
    r'\bexplore(?:s|d|ing)?\b.*\b(masked\s+autoencod(?:er|ers?)|MAE(?:s)?)\b',
]

# Decision priority:
# 1) If any STRONG_YES regex matches => label yes
# 2) Else if any AMBIGUOUS regex matches OR title/abstract missing => label dont_know
# 3) Else if any WEAK_YES regex matches AND no BACKGROUND_ONLY regex matches => label yes
# 4) Else if any BACKGROUND_ONLY regex matches => label no
# 5) Else => dont_know
#
# Precision/recall tradeoff:
# - STRONG_YES is intentionally narrow and should fire only when the abstract/title
#   clearly describes MAE usage, adaptation, pretraining, benchmarking, or a named
#   MAE-derived method.
# - WEAK_YES captures more cases where MAE is the central technique but wording is
#   less explicit; this improves recall but can produce some false positives.
# - BACKGROUND_ONLY helps suppress papers that merely mention self-supervised learning,
#   vision transformers, or masked image modeling as generic context.
# - AMBIGUOUS keeps uncertain or missing-abstract cases out of yes/no, favoring manual review.
#
# Examples from the provided context:
# 1) "Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders"
#    - matches STRONG_YES via "masked autoencoders" + "framework" + "learns" + "benchmark"
#    - label: yes
# 2) "Optical implementation and robustness validation for multi-scale masked autoencoder"
#    - matches STRONG_YES via MAE + "implementation" + "deploy" + reconstruction
#    - label: yes
# 3) "RamanMAE: Masked Autoencoders Enable Efficient Molecular Imaging..."
#    - matches STRONG_YES via named method "RamanMAE" and masked autoencoders
#    - label: yes
# 4) "Remote sensing scene classification with masked image modeling"
#    - matches BACKGROUND_ONLY / weak context, but no MAE-specific evidence
#    - label: no
# 5) "Data Masking Analysis Based on Masked Autoencoders Architecture for Leaf Diseases Classification"
#    - abstract missing => matches AMBIGUOUS
#    - label: dont_know
#
# Notes for manual alias expansion:
# - Add project-specific names if encountered: e.g., "ImageMAE", "VideoMAE",
#   "BEiT/BEVT-like masked modeling" only if the paper explicitly connects them to MAE use.
# - Add any author-defined variant names ending in "MAE" (e.g., "FooMAE") if they are
#   known to be direct adaptations of masked autoencoder pretraining.
# - The target-specific phrase "Masked Autoencoders Are Scalable Vision Learners" is usually
#   too exact for runtime matching; prefer alias-based matching on MAE, masked autoencoder(s),
#   and explicit usage verbs.
```