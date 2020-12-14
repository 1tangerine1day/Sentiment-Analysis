# XLNet-Sentiment Analysis
## xlnet
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

## Model
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
4. droup output (batch, 768) -> linear -> Linear output (batch, 5)

##  Performance-3 classes
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

## ps
* https://medium.com/ai-academy-taiwan/2019-nlp%E6%9C%80%E5%BC%B7%E6%A8%A1%E5%9E%8B-xlnet-ac728b400de3
* use sequence_summary(xlnet) not pooler(bert):
    * https://huggingface.co/transformers/_modules/transformers/models/xlnet/modeling_xlnet.html
    ![](https://i.imgur.com/rRGTf9M.png)
    * https://discuss.huggingface.co/t/why-is-there-no-pooler-representation-for-xlnet-or-a-consistent-use-of-sequence-summary/2357
<!--     ![](https://i.imgur.com/dgaIAN2.png) -->
* XLNet is one of the few models that has no sequence length limit. (but your GPU memory has)--> head 200 & tail 300
    

