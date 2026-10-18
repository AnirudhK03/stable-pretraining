"""HCL: Contrastive Learning with Hard Negative Samples.

Work in progress. SimCLR-style contrastive learning where negatives are
reweighted toward harder (more similar) samples via an importance weight
``beta``, with the debiasing correction ``tau_plus`` from Debiased Contrastive
Learning. With ``beta=0`` the loss reduces to Debiased Contrastive Learning,
and with ``beta=0, tau_plus=0`` it reduces to NT-Xent (SimCLR).

References:
    Robinson et al. "Contrastive Learning with Hard Negative Samples."
    ICLR 2021. https://arxiv.org/abs/2010.04592

    Chuang et al. "Debiased Contrastive Learning." NeurIPS 2020.
    https://arxiv.org/abs/2007.00224
"""
