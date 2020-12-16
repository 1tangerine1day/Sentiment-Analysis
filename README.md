
# Sentiment Analysis
* bert
* xlnet

## BERT
### introduction
* Transformer : https://arxiv.org/pdf/1706.03762.pdf
* Bert : https://arxiv.org/pdf/1810.04805.pdf
    * word-piece embeddings
        * 30k
        * 768 dim
        * trained on a combination of BOOKCOR-PUS plus English WIKIPEDIA,which totals 16GB of uncompressed text
    * pre-training
        * MLM : The training data generator chooses 15% of the token positions at random for prediction.  If the i-th token is chosen, we replace the i-th token with (1) the "MASK" token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time.
        * NSP :  Choosing the sentences A and B for each pre-training example, and 50% of the time B is the actual next sentence that follows A(labeled as IsNext),and 50% of the time it is a random sentence fromthe  corpus  (labeled  as NotNext).
    * Fine-tune
        * The input embeddings are the sum of the token embeddings, the segmenta-tion embeddings and the position embeddings.
        
* Bert for sentiment analysis : https://arxiv.org/pdf/1905.05583.pdf
    * **head+tail** : empirically select the first 128 and the last 382 tokens.
    * **learning rate** : 
    ![](https://i.imgur.com/xctYfy2.png)
    * **layer selection** : 
    ![](https://i.imgur.com/g6TmvQQ.png)

### Model
    class bert_classifier(nn.Module):
        def __init__(self, num_labels=5, pretrain_model="bert-base-uncased"):
            super(bert_classifier, self).__init__()
            self.bert = BertModel.from_pretrained(pretrain_model)
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask=None):
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            drop_output = self.drop(pooled_output)
            linear_output = self.out(drop_output)
            return linear_output

1. index tensor (batch, length) -> Bert -> Bert output (batch, length, 768)
2. Bert output (batch, length, 768) -> pooler output (batch, 768)
3. pooler output (batch, 768) -> Dropout(0.3) ->  Dropout output (batch, 768)
4. Dropout output (batch, 768) -> Linear -> Linear output (batch, 3)

### training
* head+tail : first 200 and the last 300 tokens
* learning rate : 2e-5
* no position_ids 
* use pretrain model "bert-base-cased"
* training batch: 5 
* epoch: 6
* Wall time: 18h 31min 17s 
* Dataset
    * amazon clawer
    * sst5 : https://github.com/prrao87/fine-grained-sentiment/tree/master/data/sst
    * amazon 5 Electronics (to balance data): https://jmcauley.ucsd.edu/data/amazon/
![](https://i.imgur.com/8Hf4WI3.png)
![](https://i.imgur.com/fkzNk52.png)
![](https://i.imgur.com/Q1NYgba.png)
 
### Performance-five classes

* train data (sst5)
    ```
    Wall time: 1min 58s
                  precision    recall  f1-score   support

               0       0.99      0.99      0.99      1092
               1       0.98      0.99      0.99      2218
               2       0.95      0.98      0.97      1624
               3       0.99      0.97      0.98      2322
               4       1.00      0.99      0.99      1288

        accuracy                           0.98      8544
       macro avg       0.98      0.98      0.98      8544
    weighted avg       0.98      0.98      0.98      8544
    ```
    ![](https://i.imgur.com/NRs1jog.jpg)
    
* test data (sst5)

    ```
    Wall time: 33.4 s
                  precision    recall  f1-score   support

               0       0.55      0.32      0.41       279
               1       0.57      0.59      0.58       633
               2       0.31      0.54      0.40       389
               3       0.51      0.53      0.52       510
               4       0.74      0.36      0.49       399

        accuracy                           0.49      2210
       macro avg       0.54      0.47      0.48      2210
    weighted avg       0.54      0.49      0.49      2210
    ```
    ![](https://i.imgur.com/XOcEVy6.jpg)

<!-- * train sst5
    ```
    Wall time: 2min
                  precision    recall  f1-score   support

               0       0.92      0.81      0.86      1092
               1       0.88      0.82      0.85      2218
               2       0.76      0.80      0.78      1624
               3       0.87      0.62      0.73      2322
               4       0.59      0.99      0.74      1288

        accuracy                           0.79      8544
       macro avg       0.81      0.81      0.79      8544
    weighted avg       0.82      0.79      0.79      8544
    ```
    ![](https://i.imgur.com/cv8pdHb.jpg) -->
    
<!-- * test data (sst5->amazon)
    ```
    Wall time: 33.8 s
                      precision    recall  f1-score   support

                   0       0.53      0.24      0.33       279
                   1       0.54      0.46      0.50       633
                   2       0.31      0.45      0.37       389
                   3       0.46      0.36      0.40       510
                   4       0.52      0.76      0.62       399

            accuracy                           0.46      2210
           macro avg       0.47      0.45      0.44      2210
        weighted avg       0.48      0.46      0.45      2210
    ```
    ![](https://i.imgur.com/eeIlDZ8.jpg) -->

<!-- * train amazon 
    ``` 
    Wall time: 32.4 s
                  precision    recall  f1-score   support

               0       1.00      0.99      1.00       150
               1       1.00      1.00      1.00       150
               2       1.00      1.00      1.00       150
               3       0.99      1.00      0.99       150
               4       0.99      0.99      0.99       150

        accuracy                           1.00       750
       macro avg       1.00      1.00      1.00       750
    weighted avg       1.00      1.00      1.00       750
    ```
    ![](https://i.imgur.com/a1xOdzW.jpg) -->
    
<!-- * test data (sst5->amazon)
    ```
    Wall time: 1min 24s
                  precision    recall  f1-score   support

               0       0.90      0.57      0.70       311
               1       0.04      0.42      0.07        12
               2       0.10      0.36      0.16        53
               3       0.33      0.59      0.42       301
               4       0.94      0.64      0.76      1190

        accuracy                           0.61      1867
       macro avg       0.46      0.52      0.42      1867
    weighted avg       0.81      0.61      0.68      1867
    ```
    ![](https://i.imgur.com/oSz4nps.jpg) -->

##  Performance-3 classes
* train data (sst5 + amazon)
    ```
        Wall time: 54min 32s
                  precision    recall  f1-score   support

               0       1.00      0.98      0.99      8034
               1       0.98      0.99      0.98      8052
               2       0.99      1.00      0.99      8045

        accuracy                           0.99     24131
       macro avg       0.99      0.99      0.99     24131
    weighted avg       0.99      0.99      0.99     24131
    ```
    
    ![](https://i.imgur.com/KcJvWoD.png)

* test data (amazon)
    ```
        Wall time: 7min 54s
                  precision    recall  f1-score   support

               0       0.89      0.84      0.87       778
               1       0.40      0.56      0.47       179
               2       0.94      0.92      0.93      1395

        accuracy                           0.86      2352
       macro avg       0.74      0.77      0.75      2352
    weighted avg       0.88      0.86      0.87      2352
    ```
    ![](https://i.imgur.com/6UItgVi.png)

* test data (sst5)
    ```
        Wall time: 35.9 s
                  precision    recall  f1-score   support

               0       0.82      0.65      0.73       912
               1       0.31      0.45      0.36       389
               2       0.81      0.82      0.81       909

        accuracy                           0.68      2210
       macro avg       0.64      0.64      0.63      2210
    weighted avg       0.72      0.68      0.70      2210
    ```
    ![](https://i.imgur.com/TWUeuXb.png)


<!-- 
* train sst5
    ```
    Wall time: 1min 59s
                  precision    recall  f1-score   support

               0       0.91      0.98      0.94      3310
               1       0.90      0.69      0.78      1624
               2       0.94      0.97      0.96      3610

        accuracy                           0.92      8544
       macro avg       0.92      0.88      0.89      8544
    weighted avg       0.92      0.92      0.92      8544
    ```
    ![](https://i.imgur.com/oZhVQpP.png)

* test sst5
    ```
    Wall time: 33.4 s
                  precision    recall  f1-score   support

               0       0.76      0.78      0.77       912
               1       0.33      0.31      0.32       389
               2       0.83      0.82      0.83       909

        accuracy                           0.72      2210
       macro avg       0.64      0.64      0.64      2210
    weighted avg       0.71      0.72      0.71      2210
    ```
    ![](https://i.imgur.com/WKw5yQc.png)

* train amazon
    ```
    Wall time: 32.4 s
                  precision    recall  f1-score   support

               0       0.98      1.00      0.99       300
               1       1.00      0.93      0.96       150
               2       0.98      1.00      0.99       300

        accuracy                           0.99       750
       macro avg       0.99      0.98      0.98       750
    weighted avg       0.99      0.99      0.99       750
    ```
    ![](https://i.imgur.com/VjYP7ZW.png)

* test amazon
    ```
    Wall time: 1min 23s
                  precision    recall  f1-score   support

               0       0.71      0.93      0.80       323
               1       0.11      0.28      0.15        53
               2       0.99      0.87      0.92      1491

        accuracy                           0.86      1867
       macro avg       0.60      0.69      0.63      1867
    weighted avg       0.92      0.86      0.88      1867
    ```
    ![](https://i.imgur.com/vsuJWN8.png)

 -->
## improve
* bert large
* Improving BERT Performance for Aspect-Based Sentiment Analysis : https://arxiv.org/pdf/2010.11731.pdf
* The Importance of Neutral Class in Sentiment Analysis of Arabic Tweets : https://www.researchgate.net/publication/302979046_The_Importance_of_Neutral_Class_in_Sentiment_Analysis_of_Arabic_Tweets

## ps
* Stopword : According to https://arxiv.org/pdf/1904.07531.pdf. Surprisingly, the stopwords received as much  attention as non-stop words, but removing them has no effect in MRR performances. (https://stackoverflow.com/questions/63633534/is-it-necessary-to-do-stopwords-removal-stemming-lemmatization-for-text-classif)
* huggingface bert has "pooled_output" to automatical choose the first output
* postition encoding need to do by yourself(postion_ids = [])
* has 512 limit
* Higher precision means less false positives.
* Higher recall means less false negatives.
---

## xlnet
### introduction
* transformer-xl : https://arxiv.org/pdf/1901.02860.pdf

    * Segment-Level Recurrence with StateReuse
    ![](https://i.imgur.com/mp5g8yB.png)
    ![](https://i.imgur.com/dznETZc.png)
        *  long  context、faster evaluation

    *  Relative Positional Encodings
        ![](https://i.imgur.com/cNO7IjB.png)
        ![](https://i.imgur.com/BWjR2fC.png)
        * replace all appearances of the absolute positional embedding $U_j$ for computing key vectors in term (b) and (d) with its relative counterpart $R_{i−j}$. (This essentially reflects the prior that only the relative distance matters for where to attend.)
        * introduce  a  trainable  parameter $u∈R^d$ to replace the query $U^T_i W^T_q$ in term (c) and $v∈R^d$ is added to substitute $U^T_iW^T_q$ in term (d). (In this case, since the query vector is the same for all query positions, it suggests that the attentive bias towards different words should remain the same regardless of the query position.)
        * deliberately separate the two weigh tmatrices $W_{k,E}$ and $W_{k,R}$ for  producing  the content-based  key  vectors  and  location-based key vectors respectively.

* xlnet : https://arxiv.org/pdf/1906.08237.pdf 
    * The pros and cons ofthe two pretraining objectives are compared in the following aspects (1) Independence Assumption (2) Input  noise (3) Context dependency problems

    * Permutation Language Modeling : AR model but knows the context (3)
    ![](https://i.imgur.com/qwJDtrk.png)

    * Two-Stream Self-Attention for Target-Aware Representations : [MASK] in Bert has (1) Independence Assumption (2) Input  noise
    ![](https://i.imgur.com/EElenPc.png)

    *  Incorporating Ideas from Transformer-XL : deal with the long text problem

### Model
```
    class xlnet_classifier(nn.Module):
        def __init__(self, num_class, pretrain_model):
            super(xlnet_classifier, self).__init__()
            self.pretrain_model = pretrain_model
            self.num_class = num_class

            self.xlnet = XLNetModel.from_pretrained(pretrain_model)
            self.drop = nn.Dropout(p=0.3)#??
            self.out = nn.Linear(self.xlnet.config.hidden_size, num_class)

        def forward(self, input_ids, attention_mask=None):
            last_hidden_states, _ = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = self.pool_hidden_state(last_hidden_states)
            drop_output = self.drop(pooled_output)
            linear_output = self.out(drop_output)
            return linear_output

        def pool_hidden_state(self, last_hidden_state):
            # last_hidden_state = last_hidden_state[0]
            mean_last_hidden_state = torch.mean(last_hidden_state, 1)
            return mean_last_hidden_state
```
1. input_ids(batch, length) -> xlnet -> xlnet output (batch, length, 768)
2. xlnet output (batch, length, 768) -> pool(summery) -> pooled output (batch, 768)
3. pooled output (batch, 768) -> drop_output -> droup output (batch, 768)
4. droup output (batch, 768) -> linear -> Linear output (batch, 3)

### training
* Wall time: 1d 3h 44min 27s
* learning rate : 2e-5
* use pretrain model "xlnet-base-cased" 
* training batch: 5 
* epoch: 6
* Dataset
    * amazon clawer
    * sst5 : https://github.com/prrao87/fine-grained-sentiment/tree/master/data/sst
    * amazon 5 Electronics (to balance data): https://jmcauley.ucsd.edu/data/amazon/
![](https://i.imgur.com/8Hf4WI3.png)
![](https://i.imgur.com/fkzNk52.png)
![](https://i.imgur.com/Q1NYgba.png)



###  Performance-3 classes
* train data (amazon + sst5)
    ```
            Wall time: 1h 58min 57s
                  precision    recall  f1-score   support

               0       0.99      0.99      0.99      8034
               1       0.99      0.97      0.98      8052
               2       0.98      1.00      0.99      8045

        accuracy                           0.99     24131
       macro avg       0.99      0.99      0.99     24131
    weighted avg       0.99      0.99      0.99     24131
    ```

    ![](https://i.imgur.com/mQisJv9.png)


* test data (amazon)
    ```
            Wall time: 17min 37s
                  precision    recall  f1-score   support

               0       0.88      0.90      0.89       778
               1       0.46      0.58      0.51       179
               2       0.96      0.91      0.94      1395

        accuracy                           0.89      2352
       macro avg       0.77      0.80      0.78      2352
    weighted avg       0.90      0.89      0.89      2352
    ```
    ![](https://i.imgur.com/ulwv87t.png)

* test data (sst5)
    ```
            Wall time: 40.9 s
                  precision    recall  f1-score   support

               0       0.81      0.73      0.77       912
               1       0.40      0.30      0.34       389
               2       0.76      0.92      0.83       909

        accuracy                           0.73      2210
       macro avg       0.66      0.65      0.65      2210
    weighted avg       0.72      0.73      0.72      2210
    ```
    ![](https://i.imgur.com/PoKAEIZ.png)

### ps
* https://medium.com/ai-academy-taiwan/2019-nlp%E6%9C%80%E5%BC%B7%E6%A8%A1%E5%9E%8B-xlnet-ac728b400de3
* use sequence_summary(xlnet) not pooler(bert):
    * https://huggingface.co/transformers/_modules/transformers/models/xlnet/modeling_xlnet.html
    ![](https://i.imgur.com/rRGTf9M.png)
    * https://discuss.huggingface.co/t/why-is-there-no-pooler-representation-for-xlnet-or-a-consistent-use-of-sequence-summary/2357
<!--     ![](https://i.imgur.com/dgaIAN2.png) -->
* XLNet is one of the few models that has no sequence length limit. (but your GPU memory has)--> head 200 & tail 300
    

