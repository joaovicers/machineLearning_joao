###########################################################
# AUTHOR: JOAO VICTOR DE ARAUJO OLIVEIRA
#
#	This version uses RNAshapes instead RNAfold -C
###########################################################
import logging
logging.basicConfig(level=logging.DEBUG)
########################## Arguments#####################################
fasta = "../CD_sim_dataset1.fa"
fasta_test = "../CD_sim_dataset2.fa"
model_stem_name = 'stem_model_1'
model_c_name = "../../C_finder/model_Cbox_1" 
model_d_name = "../../D_finder/model_Dbox_1"
train_test_split=0.7
neg_size_factor = 4
n_jobs=2
n_iter=60
window_c = 3
window_d = 3
flank_size_l=4
flank_size_r=3

############Train stem finder#####################################

def train_stem_finder_model(fasta,model_stem_name,window_c,model_c_name,window_d, model_d_name,flank_size_l,flank_size_r, train_test_split=0.7,neg_size_factor = 4, n_jobs=4, n_iter=40,fasta_test=None):

    ########### Pre processor ####################
    def pre_process_graph(iterator):    
        from eden.converter.rna.rnasubopt import rnasubopt_to_eden
        graphs = rnasubopt_to_eden(iterator, energy_range=10, max_num=10, split_components=False)
        return graphs
    ########## Vectorizer ########################
    from eden.graph import Vectorizer
    vectorizer = Vectorizer()

    ######### Estimator #########################
    from sklearn.linear_model import SGDClassifier
    estimator = SGDClassifier(class_weight='auto', shuffle = True, average=True )

    
    def get_Cbox(seqs,window_c):
		import re
		for seq in seqs:
			header = seq[0].split('_')
			cpos = int(header[-3])
			nts = re.sub('\n','',seq[1])
			if (not((len(nts) < cpos+6+window_c) or (cpos-1-window_c < 0))):
				box = nts[cpos-1-window_c:cpos-1]+'x'+nts[cpos-1:cpos+6]+'y'+nts[cpos+6:cpos+6+window_c]
				yield [seq,cpos],box

    def get_Dbox(seqs_c,window_d):
	    import re 
	    for [seq,cbox],pred in seqs_c:
			header = seq[0][0].split('_')
			dpos = int(header[-1])
			nts = re.sub('\n','',seq[0][1])
			if (not((len(nts) < dpos+3+window_d) or (dpos-1-window_d < 0))):
				box = nts[dpos-1-window_d:dpos-1]+'x'+nts[dpos-1:dpos+3]+'y'+nts[dpos+3:dpos+3+window_d]
				yield [seq,cbox,pred,dpos],box
    
                    
    ######### Get stem #########################
    def get_stem(seqs,window_c,model_c_name,window_d, model_d_name,flank_size_l,flank_size_r):
        from itertools import izip 
        import re
        from itertools import tee,islice
       
        #1)c_finder
        seqs_c = get_Cbox(seqs,window_c)
		

        #2)submit the Cbox candidates to the model
        from eden.model import ActiveLearningBinaryClassificationModel
        model = ActiveLearningBinaryClassificationModel()
        model.load(model_c_name)
        
        seqs_c_pred = list()
        cands_c = list()
        max_count = 0        
        
        for seq_c in seqs_c:
            max_count +=1
            cands_c.append(seq_c)
            if (max_count == 10000): #in order to not generate memory leak I've restricted the number of samples to be submited to the model
				preds = model.decision_function(cands_c)
				seqs_c_pred = seqs_c_pred + zip(cands_c,preds)
				cands_c = list()
				max_count = 0
        if (max_count != 0):
			preds = model.decision_function(cands_c)
			seqs_c_pred = seqs_c_pred + zip(cands_c,preds)
        
        #discard sequences with pred < 0
        seqs_c = list()
        for cand in seqs_c_pred:
			if (cand[1] >= 0.0):
				seqs_c.append(cand)
        
        
        #D_finder
        seqs_cd = get_Dbox(seqs_c,window_d)
        #submit Dboxes candidate to its model
        model = ActiveLearningBinaryClassificationModel()
        model.load(model_d_name)
        
        seqs_d_pred = list()
        cands_d = list()
        max_count = 0        
        
        for seq_d in seqs_cd:
            max_count +=1
            cands_d.append(seq_d)
            if (max_count == 10000): #in order to not generate memory leak I've restricted the number of samples to be submited to the model
				preds = model.decision_function(cands_d)
				seqs_d_pred = seqs_d_pred + zip(cands_d,preds)
				cands_d = list()
				max_count = 0
        if (max_count != 0):
			preds = model.decision_function(cands_d)
			seqs_d_pred = seqs_d_pred + zip(cands_d,preds)
		
	#Get the stem region from the sequences
        stem_cands=[]
        stem_info =[]
        #(([[(header, seq), pos_c], cand_c, pred_c, pos_d], 'UAAxCUGAyGAU'), 77.000434164559792)

        for ([[(header,nts),pos_c],cand_c,pred_c,pos_d],cand_d),pred_d in seqs_d_pred:
			#print header,'\t',seq,pos_c,'\t',cand_c,'\t',pred_c,'\t',cand_d,'\t',pred_d,"\n---\n" 
			if ( int(pos_c) - 10 < 0):
				if (int(pos_d)+10 > len(nts)):
					stem_cands.append([[header,pos_c,pos_d],nts[0:int(pos_c)+6]+"&"+nts[int(pos_d)-1:len(nts)]])
				else:
					stem_cands.append([[header,pos_c,pos_d],nts[0:int(pos_c)+6]+"&"+nts[int(pos_d)-1:int(pos_d)+3+10]])
					
			else:
				if (int(pos_d)+10 > len(nts)):
					stem_cands.append([[header,pos_c,pos_d],nts[int(pos_c)-10:int(pos_c)+6]+"&"+nts[int(pos_d)-1:len(nts)]])
					
				else:
					stem_cands.append([[header,pos_c,pos_d],nts[int(pos_c)-10:int(pos_c)+6]+"&"+nts[int(pos_d)-1:int(pos_d)+3+10]])
					
		
        return stem_cands
            
			
    
    #get positive data
    pos_cds=[]

    from eden.converter.fasta import fasta_to_sequence
    seqs = fasta_to_sequence(fasta)

    train_pos = get_stem(seqs,window_c,model_c_name,window_d, model_d_name,flank_size_l,flank_size_r)
    train_pos = list(train_pos)

    

    #for h,seq in stems_cds:
	#	print h[0][0:10],'\t',seq

    #Generate Negative Dataset
    from eden.modifier.seq import seq_to_seq, shuffle_modifier
    train_neg = seq_to_seq(train_pos, modifier=shuffle_modifier, times=neg_size_factor, order=2 )
    train_neg = list(train_neg)
 
    
    #######Split the data into training and test
    if (fasta_test == None):
        print "Training and Test with the same dataset (different sequences)"
        #split train/test
        from eden.util import random_bipartition_iter
        iterable_pos_train, iterable_pos_test = random_bipartition_iter(train_pos, relative_size=train_test_split)
        iterable_neg_train, iterable_neg_test = random_bipartition_iter(train_neg, relative_size=train_test_split)
        
        
        iterable_pos_train = list(iterable_pos_train)
        iterable_neg_train = list(iterable_neg_train)
        
        iterable_pos_test = list(iterable_pos_test)
        iterable_neg_test = list(iterable_neg_test)
        
    
        
      
        

    else:        
        print "test dataset = ",fasta_test,"\n"
        pos_test_cds=[]
        neg_test_cds=[]

        from eden.converter.fasta import fasta_to_sequence
        seqs = fasta_to_sequence(fasta_test)

        test_pos = get_stem(seqs,window_c,model_c_name,window_d, model_d_name,flank_size_l,flank_size_r)
        test_pos = list(test_pos)

        #Generate Negative test data
        test_neg = seq_to_seq(test_pos, modifier=shuffle_modifier, times=neg_size_factor, order=2 )
        test_neg = list(test_neg)
        
        iterable_pos_train = list(train_pos)
        iterable_neg_train = list(train_neg)
        iterable_pos_test  = list(test_pos)
        iterable_neg_test  = list(test_neg)
        
    print "Positive training samples: ",len(iterable_pos_train)
    print "Negative training samples: ",len(iterable_neg_train)
    print "--------\nPositive test samples: ",len(iterable_pos_test)
    print "Negative test samples: ",len(iterable_neg_test)



    #make predictive model
    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel(pre_processor=pre_process_graph, 
                                                    estimator=estimator, 
                                                    vectorizer=vectorizer,
                                                    n_jobs=n_jobs)
    #optimize hyperparameters and fit model:
    from numpy.random import randint
    from numpy.random import uniform


    pre_processor_parameters={'energy_range':[3,4,5],
							   'max_num_subopts':randint(100,200,size=n_iter),
							   'max_num': [3,4,5,6]}


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
                   model_name=model_stem_name,
                   max_total_time=60*60*4, n_iter=n_iter, 
                   n_active_learning_iterations=3,
                   cv=10,
                   score_func=lambda avg_score,std_score : avg_score - std_score * 2,
                   scoring='roc_auc', 
                   vectorizer_parameters=vectorizer_parameters, 
                   estimator_parameters=estimator_parameters)

    #estimate predictive performance
    print model.get_parameters()

    result,text = model.estimate( iterable_pos_test, iterable_neg_test )

    rss=0
    i = 0

    for prob in result:
        i=i+1
        #print prob
        if (prob[1] == 1):
            rss = rss + ((1 - prob[0][1])**2)
        else:
            rss = rss + ((1 - prob[0][0])**2)

    avg_rss= rss/i;
    text.append('RSS: %.2f' % rss)
    text.append('avg RSS: %2f' % avg_rss)

    for t in text:
        print t        

train_stem_finder_model(fasta,model_stem_name,window_c,model_c_name,window_d, model_d_name,flank_size_l,flank_size_r,train_test_split,neg_size_factor, n_jobs, n_iter,fasta_test)
