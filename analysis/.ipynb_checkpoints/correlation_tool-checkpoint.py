from scipy import stats
import matplotlib.pyplot as plt

def correlator(gt_dict, pred_dict):
    correlated_lists = [[], []]
    print('should be estimated: ', len(gt_dict), '; number of estimations: ', len(pred_dict), '; really estimated: ', len(set(gt_dict.keys()).intersection(set(pred_dict.keys()))))
    for qid in set(gt_dict.keys()).intersection(set(pred_dict.keys())):
        correlated_lists[0].append(gt_dict[qid])
        correlated_lists[1].append(pred_dict[qid])

    print('r:', stats.pearsonr(correlated_lists[0], correlated_lists[1])[0])
    print('tau:', stats.kendalltau(correlated_lists[0], correlated_lists[1])[0])
    
    plt.scatter(correlated_lists[0], correlated_lists[1])
    plt.show()
    return {'r': stats.pearsonr(correlated_lists[0], correlated_lists[1]), 'tau': stats.kendalltau(correlated_lists[0], correlated_lists[1]), 'rho': stats.spearmanr(correlated_lists[0], correlated_lists[1])}