import numpy as np
from scipy.stats import norm, kendalltau, spearmanr

def fisher_z(r: float) -> float:
    """Fisher's z-transformation for Pearson's r."""
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_spearman_rhos(n, r1, r2):
    # Fisher's z-transform
    z1, z2 = fisher_z(r1), fisher_z(r2)
    # Standard error assuming similar sample size
    se = np.sqrt(1.060 / (n - 3))
    # Difference statistic
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return  {"z_stat": z_stat, "p_value": p_value}

def tau_to_r(tau: float) -> float:
    """
    Approximate conversion from Kendall's tau to Pearson's r.
    This uses a meta-analysis formula (e.g., Walker, 2003).
    """
    return np.sin(np.pi * tau / 2)

def compare_kendall_taus(n, tau1, tau2):
    """
    Compare two Kendall's tau correlations via Fisher's z on
    approximate Pearson r values.
    
    Parameters:
    - n: length of the two arrays
    - tau1, tau2: Kendall's tau values
    Return:
    - z_stat: test statistic
    - p_value: two-sided p-value
    """
    # Convert to Pearson r via approximation
    r1, r2 = tau_to_r(tau1), tau_to_r(tau2)

    return compare_spearman_rhos(n, r1, r2)
