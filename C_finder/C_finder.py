def train_cbox_model(fasta_fname=None, model_fname='eden_model_Cbox',window = 3, neg_size_factor=5, train_test_split=0.7, n_jobs=4, n_iter=40):
    
    #transform sequences in a linear graph
    def pre_process_graph(iterator):
        from eden.converter.fasta import sequence_to_eden
        graphs = sequence_to_eden(iterator)
        return graphs

    #extract box sequence with annotaded header information
    def extract_box(data,window=3,box_type='C'):
        import re
        from eden.converter.fasta import fasta_to_sequence
        seqs = fasta_to_sequence(data)
        
        for seq in seqs:
            header = seq[0].split('_')
            cbox = header[-4]
            cpos = int(header[-3])
            dbox = header[-2]
            dpos = int(header[-1])
            
            nts = re.sub('\n','',seq[1])
            if box_type == 'C':
                if (not((len(nts) < cpos+6+window) or (cpos-1-window < 0))):
                    #box = nts[cpos-1-window:cpos+6+window]
                    box = nts[cpos-1-window:cpos-1]+'x'+nts[cpos-1:cpos+6]+'y'+nts[cpos+6:cpos+6+window]
                    yield seq[0],box
    
    #Choose the vectorizer
    from eden.graph import Vectorizer
    vectorizer = Vectorizer()
        
    #Choose the estimator
    from sklearn.linear_model import SGDClassifier
    estimator = SGDClassifier(class_weight='auto', shuffle = True, average=True )
    
    import random
    from eden import util
    
    #Generate positive samples
    seqs_c_pos = extract_box(fasta_fname,window,box_type='C')

    #from itertools import islice
    #for h,seq in islice(seqs_c_pos,3):
    #   print h,"\t",seq
    

    from itertools import tee
    seqs_c_pos,seqs_c_pos_ = tee(seqs_c_pos)
    
    #shuffle negative samples
    from eden.modifier.seq import seq_to_seq, shuffle_modifier
    seqs_c_neg = seq_to_seq( seqs_c_pos_, modifier=shuffle_modifier, times=neg_size_factor, order=2 )
    
    #split train/test
    from eden.util import random_bipartition_iter
    iterable_pos_train, iterable_pos_test = random_bipartition_iter(seqs_c_pos, relative_size=train_test_split)
    iterable_neg_train, iterable_neg_test = random_bipartition_iter(seqs_c_neg, relative_size=train_test_split)
    

    iterable_pos_train = list(iterable_pos_train)
    iterable_pos_test = list(iterable_pos_test)
    iterable_neg_train = list(iterable_neg_train)
    iterable_neg_test = list(iterable_neg_test)

    print "training pos ",len(iterable_pos_train)
    print "training neg ",len(iterable_neg_train)
    print "test pos ",len(iterable_pos_test)
    print "test neg ",len(iterable_neg_test)



    #make predictive model
    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel(pre_processor=pre_process_graph, 
                                                    estimator=estimator, 
                                                    vectorizer=vectorizer,
                                                    n_jobs=n_jobs)
    #optimize hyperparameters and fit modelter=40):
    from numpy.random import randint
    from numpy.random import uniform
    
    vectorizer_parameters={'complexity':[2,3]}
    
    estimator_parameters={'n_iter':randint(5, 100, size=n_iter),
                          'penalty':['l1','l2','elasticnet'],
                          'l1_ratio':uniform(0.1,0.9, size=n_iter), 
                          'loss':['log'],#'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                          'power_t':uniform(0.1, size=n_iter),
                          'alpha': [10**x for x in range(-8,0)],
                          'eta0': [10**x for x in range(-4,-1)],
                          'learning_rate': ["invscaling", "constant", "optimal"],
                          'n_jobs':[n_jobs]}
    
    model.optimize(iterable_pos_train, iterable_neg_train, 
                   model_name=model_fname,
                   max_total_time=60*60*3, n_iter=n_iter, 
                   cv=10,
                   score_func=lambda avg_score,std_score : avg_score - std_score * 2,
                   scoring='roc_auc', 
                   vectorizer_parameters=vectorizer_parameters, 
                   estimator_parameters=estimator_parameters)

    #estimate predictive performance
    
    print model.get_parameters()
    
    ###############using different dataset for test#######

    
    result,text = model.estimate( iterable_pos_test, iterable_neg_test )
    
    rss=0
    i = 0

    for prob in result:
        i=i+1
        if (prob[1] == 1):
            rss = rss + ((1 - prob[0][1])**2)
        else:
            rss = rss + ((1 - prob[0][0])**2)

    avg_rss= rss/i;
    text.append('RSS: %.2f' % rss)
    text.append('avg RSS: %2f' % avg_rss)

    for t in text:
        print t

import sys

if (len(sys.argv) > 2):
	fasta = sys.argv[1]
	model_c = sys.argv[2]
	#print fasta,model_c
	train_cbox_model(fasta_fname=fasta, model_fname=model_c,window=3, neg_size_factor=4, train_test_split=0.7, n_jobs=3, n_iter=40)

