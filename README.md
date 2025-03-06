# Varying Sentence Representations via Condition-Specified Routers

## Updates

- (2024.09.20) Our Paper have been accepted by **EMNLP 2024**🔥🔥.

## 🚀 Overview

In this paper, we propose a **C**onditioned **S**entence **R**epresentation (**CSR**) method based on the tri-encoder architecture, with the goal of enhancing its performance without introducing external parameters while maintaining computational efficiency. Condition semantics ought to play the role of influencing which tokens in the sentence should contribute to the final condition-specific sentence embedding. Our approach obtains different score distributions for a sentence based on different conditions, thereby generating varied conditioned sentence representations.

We evaluate our method on the C-STS task and Knowledge Graph Completion (KGC) task, demonstrating significant improvement over previous frameworks while maintaining memory and computational efficiency.

## ⚙️ Installation

```bash
# clone project
git clone https://github.com/T0nglinziyong/C-STS.git

# install requirements
pip install -r requirements.txt

```

## 💡 How to run

You can download all the data from [https://github.com/princeton-nlp/c-sts] following the insrtuctions in the repository.

Train model

```bash
source env_for_tri_encoder.sh

bash run_sts.sh
```

Evaluate model

```python
python make_test_submission.py
```

## ⚽ Evaluation Results

***C-STS***

![image](https://github.com/user-attachments/assets/ea8add18-9c46-4f12-93dc-c9d3ef3abd25)

***KGC***

![image](https://github.com/user-attachments/assets/eebbe7a1-50e0-4819-8ee9-142d3e652a2c)


## Acknowledgement

Data / Code: 
- [C-STS](https://github.com/princeton-nlp/c-sts)

## 📜 Citation

```tex
@inproceedings{lin2024varying,
  title={Varying Sentence Representations via Condition-Specified Routers},
  author={Lin, Ziyong and Wang, Quansen and Jia, Zixia and Zheng, Zilong},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={17390--17401},
  year={2024}
}
```
