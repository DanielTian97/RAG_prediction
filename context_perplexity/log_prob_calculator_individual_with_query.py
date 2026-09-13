from prob_estimator import ProbEstimator
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='nq_test')
    parser.add_argument("--device_number", type=int, default=0)
    args = parser.parse_args()

    calculator = ProbEstimator(_task=args.task, _ret='e5', load_on_device=args.device_number)
    print('set retriever as E5')
    
    calculator.individual_doc_probs_with_query(20)
    
    calculator.change_retriever('bm25')
    print('set retriever as bm25')
    
    calculator.individual_doc_probs_with_query(20)
    
    calculator.change_retriever('mt5')
    print('set retriever as monoT5')
    
    calculator.individual_doc_probs_with_query(20)
    
    del calculator