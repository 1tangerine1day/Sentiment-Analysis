# Sentiment Analysis
## Model
* Bert: https://hackmd.io/@h5v7rKYIT4GGOr9b0g0k7g/SyKjPsasv
* XLNet: https://hackmd.io/@h5v7rKYIT4GGOr9b0g0k7g/SyKjPsasv
* RoBerta: https://hackmd.io/@h5v7rKYIT4GGOr9b0g0k7g/H1osvVOnv
## Dataset
* amazon clawer
<br>![](https://i.imgur.com/8Hf4WI3.png )
* sst5 : https://github.com/prrao87/fine-grained-sentiment/tree/master/data/sst
<br>![](https://i.imgur.com/fkzNk52.png )
* amazon 5 Electronics (to balance data): https://jmcauley.ucsd.edu/data/amazon/
<br>![](https://i.imgur.com/Q1NYgba.png )

## Performance

|                | 0 -- f1  | 1 -- f1  | 2 -- f1  |
| --------       | -------- | -------- | -------- |
| BERT           | 0.87     | 0.47     | 0.93     |
| XLNet          | 0.89     | 0.51     | 0.94     |
| RoBERTa(sum)   | 0.86     | 0.43     | 0.91     |
| RoBERTa(pool)  | 0.89     | 0.48     | 0.93     |

## case study

A: `Not as extreme as touted.` (3/5)
B: `IT WAS DIFFERENT TYPE MOVIE BUT IT WAS GOOD. SUSPENSEFUL.` (5/5)
C: `Not impressed with this movie, too many unknown variables.` (1/5)
D: `The first half is outstanding. Lupita Nyongo is excellent as always -- shes a moody complex character which makes complete sense after the big reveal.Once that big reveal happens, suspending disbelief is difficult, even for a horror movie. It is almost as if they came up with the main premise, than spit-balled an explanation. The explanation / back story is so absurd it requires way too much exposition. Heres hoping Peele gets more grounded in his third act.` (3/5)
<!-- E: Not impressed with this movie, but the actors performance well. -->

|                | A        | B        | C        | D        |
| --------       | -------- | -------- | -------- | -------- |
| BERT           | 0        | 2        | 0        |2         |
| XLNet          | 1        | 2        | 1        |1         |
| RoBERTa(sum)   | 1        | 1        | 0        |2         |
| RoBERTa(pool)  | 1        | 1        | 0        |1         |

* class:
    * 0-> 1、2 (stars)
    * 1-> 3 (stars)
    * 2-> 4、5 (stars)

## improve
* use "large" model not "base"
* Aspect-Based Sentiment Analysis
* more data ?
