# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Function for generating association rules
#
# Author: Joshua Goerner <https://github.com/JoshuaGoerner>
#         Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
import numpy as np
import pandas as pd


def association_rules(df, metric="confidence",
                      min_threshold=0.8, support_only=False, custom_metrics={}):
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', and 'conviction'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.

    Returns
    ----------
    pandas DataFrame with columns "antecedents" and "consequents"
      that store itemsets, plus the scoring metric columns:
      "antecedent support", "consequent support",
      "support", "confidence", "lift",
      "leverage", "conviction"
      of all rules for which
      metric(rule) >= min_threshold.
      Each entry in the "antecedents" and "consequents" columns are
      of type `frozenset`, which is a Python built-in type that
      behaves similarly to sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    def conviction_helper(sAC, sA, sC):
        confidence = sAC / sA
        if confidence < 1.:
            return (1. - sC) / (1. - confidence)
        else:
            return np.inf

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, antecedent, __, dict: dict[antecedent]["support"],
        "consequent support": lambda _, __, consequent, dict: dict[consequent]["support"],
        "support": lambda k, _, __, dict: dict[k]["support"],
        "confidence": lambda k, antecedent, _, dict: dict[k]["support"] / dict[antecedent]["support"],
        "lift": lambda k, antecedent, consequent, dict: metric_dict["confidence"](k, antecedent, consequent, dict) /
                                                  dict[consequent]["support"],
        "leverage": lambda k, antecedent, consequent, dict: metric_dict["support"](
            k, antecedent, consequent, dict) - dict[antecedent]["support"] * dict[consequent]["support"],
        "conviction": lambda k, antecedent, consequent, dict: conviction_helper(dict[k]["support"],
                                                                          dict[antecedent]["support"],
                                                                          dict[consequent]["support"])
    }

    columns_ordered = ["antecedent support", "consequent support",
                       "support",
                       "confidence", "lift",
                       "leverage", "conviction"]

    columns_ordered = columns_ordered + sorted(set(custom_metrics.keys()).difference(metric_dict.keys()))
    metric_dict.update(custom_metrics)

    # check for metric compliance
    if support_only:
        metric = 'support'
    else:
        if metric not in metric_dict.keys():
            raise ValueError("Metric must be 'confidence' or 'lift', got '{}'"
                             .format(metric))

    # get dict of {frequent itemset} -> support
    keys = df['itemsets'].values
    values = df.to_dict('records')
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_ks = []
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]["support"]
        # to find all possible combinations
        for idx in range(len(k) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    # support doesn't need these,
                    # hence, placeholders should suffice
                    sA = None
                    sC = None

                else:
                    try:
                        sA = frequent_items_dict[antecedent]["support"]
                        sC = frequent_items_dict[consequent]["support"]
                    except KeyError as e:
                        s = (str(e) + 'You are likely getting this error'
                                      ' because the DataFrame is missing '
                                      ' antecedent and/or consequent '
                                      ' information.'
                                      ' You can try using the '
                                      ' `support_only=True` option')
                        raise KeyError(s)
                    # check for the threshold

                score = metric_dict[metric](k, antecedent, consequent, frequent_items_dict)
                if score >= min_threshold:
                    rule_ks.append(k)
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedents", "consequents"] + columns_ordered)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_ks, rule_antecedents, rule_consequents)),
            columns=["ks", "antecedents", "consequents"])

        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res['support'] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = [metric_dict[m](x[0], x[1], x[2], frequent_items_dict) for x in zip(rule_ks, rule_antecedents, rule_consequents)]
        df_res.drop(columns=["ks"], inplace=True)
        return df_res
