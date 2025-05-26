from prob_estimator import ProbEstimator
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='nq_test')

    calculator = ProbEstimator(_task=args.task, _ret='e5')
    print('set retriever as E5')
    
    for k in range(2, 13):
        print(f'k={k}')
        calculator.concatenated_context_probs(k)
    
    calculator.change_retriever('bm25')
    print('set retriever as bm25')
    
    for k in range(2, 13):
        print(f'k={k}')
        calculator.concatenated_context_probs(k)
    
    calculator.change_retriever('mt5')
    print('set retriever as monoT5')
    
    for k in range(2, 13):
        print(f'k={k}')
        calculator.concatenated_context_probs(k)
    
    del calculator