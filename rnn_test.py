import theano
import theano.tensor as T
import pickle
import numpy as np
from collections import OrderedDict


class model(object):       
     def __init__(self, nh, nc, ne, de, cs, bs): 
         ''' 
         nh :: dimension of the hidden layer 
         nc :: number of classes 
         ne :: number of word embeddings in the vocabulary 
         de :: dimension of the word embeddings 
         cs :: word window context size  
         bs :: batch size (number of samples)
         ''' 
         # parameters of the model 
         self.emb = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end 
         self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(de * cs, nh)).astype(theano.config.floatX)) 
         self.Wh  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX)) 
         self.W   = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX)) 
         self.bh  = theano.shared(np.zeros(nh, dtype=theano.config.floatX)) 
         self.b   = theano.shared(np.zeros(nc, dtype=theano.config.floatX)) 
         self.h0  = theano.shared(np.zeros((bs,nh), dtype=theano.config.floatX)) 
 
 
         # bundle 
         self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ] 
         self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0'] 
         idxs = T.itensor3() # time->samples->features as many columns as context window size/lines as words in the sentence 
         x = self.emb[idxs].reshape((idxs.shape[0], bs, de*cs))          
         y_sentence = T.imatrix('y_sentence')
 
         def recurrence(x_t, h_tm1): 
             h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh) 
             s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b) 
             return [h_t, s_t] 
 
 
         [h, s], _ = theano.scan(fn=recurrence,sequences=x, outputs_info=[self.h0, None], n_steps=x.shape[0]) 
 
 
         p_y_given_x_lastword = s[-1,0,:] 
         #p_y_given_x_sentence = s[:,0,:] 
         #p_y_given_x_sentence = s.reshape((x.shape[0]* bs, nc)) 
         p_y_given_x_sentence = s
         #y_pred = T.argmax(p_y_given_x_sentence, axis=1) 
         y_pred = T.argmax(p_y_given_x_sentence, axis=2) 
 
 
         # cost and gradients and learning rate 
         lr = T.scalar('lr') 

         sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]).dimshuffle(0,'x'),T.arange(bs).dimshuffle('x', 0), y_sentence])
                  
         #sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]*bs), y_sentence.flatten()])
         sentence_error = T.sum(T.neq(y_pred, y_sentence))
         sentence_gradients = T.grad(sentence_nll, self.params)
         sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))


         # theano functions 
         self.classify = theano.function(inputs=[idxs], outputs=y_pred) 

         self.train_sentence = theano.function( inputs  = [idxs, y_sentence, lr], 
                                       outputs = sentence_nll, 
                                       updates = sentence_updates ) 
 
         self.errors = theano.function(inputs=[idxs, y_sentence], outputs=sentence_error)             

 
         self.normalize = theano.function( inputs = [], 
                          updates = {self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})  


def loadData():
    dataPath = r"../Data/atis.pkl"

    with open(dataPath, "rb") as f:
        train, test, dicts = pickle.load(f, encoding="latin1")


    converted = dict()
    for ds in dicts.items():
        dict_name = ds[0]#.decode("utf-8")    
        inner_dict = dict()    
        converted[dict_name] = inner_dict
        for v in ds[1].items():
             key = v[0]#.decode("utf-8")
             inner_dict[key] = v[1]

    return converted["words2idx"], converted["labels2idx"], converted["tables2idx"], train, test

    
def contextwin(l, win):  
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def minibatch(l, bs): 
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )] 
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ] 
    assert len(l) == len(out) 
    return out 


words2idx, labels2idx, tables2idx, train, test = loadData()

train_lex, train_ne, train_y = train
test_lex, test_ne, test_y = test

vocsize = len(words2idx) 
nclasses = len(labels2idx) 
nsentences = len(train_lex) 
win = 7
bs = 32
lr = 0.0627142536696559 

rnn = model(    nh = 100, 
                     nc = nclasses, 
                     ne = vocsize, 
                     de = 100, 
                     cs = win,
                     bs = bs) 

for e in range(50):     
    print("Epoch {0}".format(e))    
    nBatches = nsentences // bs
    #for i in range(nsentences): 
    for b in range(nBatches): 
        
        #if i % 100 == 0:
        #    print("{0}%".format(i*100/nsentences))
        s = 0
        r = []
        for i in range(b*bs, (b+1)*bs):
            cw = contextwin(train_lex[i], win)
            s = np.max([s, len(cw)])
            r.append(cw)
        input = np.ones((s, bs, win), dtype='int32')*(-1)
        startIndex = b*bs
        for c in range(bs):            
            cw = contextwin(train_lex[c+startIndex], win)
            input[:len(cw), c , :] = np.array(cw, dtype='int32')
              
                                
        #cwords = [contextwin(train_lex[i], win) for i in range(b*batchSize, (b+1)*batchSize)] 
        #words  = map(lambda x: np.asarray(x).astype('int32'), minibatch(cwords, bs))         
        labels = [train_y[i] for i in range(b*bs, (b+1)*bs)]
        targets = np.ones((s, bs), dtype='int32')*(-1)
        for i in range(len(labels)):
            v = labels[i]
            targets[:len(v), i] = v
        
        #np.zeros((s, bs), dtype='int32')
        #rnn.train_sentence(cwords, labels, lr)
        rnn.train_sentence(input, targets, lr)
        rnn.normalize()
        #for word_batch , label_last_word in zip(words, labels): 
            #rnn.train(word_batch, label_last_word, lr) 
            #rnn.normalize() 
    
    cum_errors = 0
    nTestBatches = len(test_lex) // bs
    for b in range(nTestBatches):
        s = 0
        r = []
        for i in range(b*bs, (b+1)*bs):
            cw = contextwin(test_lex[i], win)
            s = np.max([s, len(cw)])
            r.append(cw)
        input = np.zeros((s, bs, win), dtype='int32')        
        startIndex = b*bs
        for c in range(bs):            
            cw = contextwin(test_lex[c+startIndex], win)
            input[:len(cw), c , :] = np.array(cw, dtype='int32')

        labels = [test_y[i] for i in range(b*bs, (b+1)*bs)]
        targets = np.ones((s, bs), dtype='int32')*(-1)
        for i in range(len(labels)):
            v = labels[i]
            targets[:len(v), i] = v

        cum_errors += rnn.errors(input, targets)

    print("Test errors {0}".format(cum_errors))