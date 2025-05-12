from scipy import stats
import matplotlib.pyplot as plt

def correlator(gt_dict, pred_dict, colour_dict={}):

    if(colour_dict == {}):
        for qid in gt_dict:
            colour_dict.update({qid: 1})
    
    correlated_lists = [[], []]
    colour_list = []
    print('should be estimated: ', len(gt_dict), '; number of estimations: ', len(pred_dict), '; really estimated: ', len(set(gt_dict.keys()).intersection(set(pred_dict.keys()))))
    for qid in set(gt_dict.keys()).intersection(set(pred_dict.keys())):
        correlated_lists[0].append(gt_dict[qid])
        correlated_lists[1].append(pred_dict[qid])
        colour_list.append(colour_dict[qid])

    print('r:', stats.pearsonr(correlated_lists[0], correlated_lists[1])[0])
    print('tau:', stats.kendalltau(correlated_lists[0], correlated_lists[1])[0])
    
    plt.scatter(correlated_lists[1], correlated_lists[0], c=colour_list, cmap='viridis')
    plt.show()
    return {'r': stats.pearsonr(correlated_lists[0], correlated_lists[1]), 'tau': stats.kendalltau(correlated_lists[0], correlated_lists[1]), 'rho': stats.spearmanr(correlated_lists[0], correlated_lists[1])}

def correlator_simple(gt_dict, pred_dict):

    correlated_lists = [[], []]
    
    # print('should be estimated: ', len(gt_dict), '; number of estimations: ', len(pred_dict), '; really estimated: ', len(set(gt_dict.keys()).intersection(set(pred_dict.keys()))))
    for qid in set(gt_dict.keys()).intersection(set(pred_dict.keys())):
        correlated_lists[0].append(gt_dict[qid])
        correlated_lists[1].append(pred_dict[qid])

    r, p1_r = stats.pearsonr(correlated_lists[0], correlated_lists[1])
    tau, p1_tau = stats.kendalltau(correlated_lists[0], correlated_lists[1])
    # print('r:', r)
    # print('tau:', tau)
    
    return [r, p1_r], [tau, p1_tau]