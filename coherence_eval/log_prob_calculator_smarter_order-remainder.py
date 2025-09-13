from prob_estimator import ProbEstimator
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='nq_test')
    args = parser.parse_args()

    # calculator = ProbEstimator(_task=args.task, _ret='e5')
    # print('set retriever as E5')
    
    # for k in [3, 2, 5, 10, 4, 6, 7, 8, 9, 11, 12]:
    #     print(f'k={k}')
    #     calculator.concatenated_context_probs(k)
    
    calculator = ProbEstimator(_task=args.task, _ret='bm25')
    print('set retriever as bm25')
    
    for k in [9, 11, 12]:
        print(f'k={k}')
        calculator.concatenated_context_probs(k)
    
    calculator.change_retriever('mt5')
    print('set retriever as monoT5')
    
    for k in [3, 2, 5, 10, 4, 6, 7, 8, 9, 11, 12]:
        print(f'k={k}')
        calculator.concatenated_context_probs(k)
    
    del calculator