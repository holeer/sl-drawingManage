from config.logger import Logger
from config.config import config
import transformers
import numpy as np
import time
from sklearn import metrics
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM
from utils import read_dataset_prompt


LOG_FILENAME = config.log_dir + "Prompt_" + str(int(time.time())) + ".log"
print(30 * "=",
      "Training log in file: {}".format(LOG_FILENAME),
      30 * "=")
logging = Logger(filename=LOG_FILENAME).logger


class LecCallTag():

    # model, tokenizer
    def create_model_tokenizer(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        return model, tokenizer

    # 构建dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        X_train, X_test, Y_train, Y_test = train_test_split(text, label, test_size=0.2, random_state=1)
        logging.info('训练集：%s条，测试集：%s条' % (len(X_train), len(X_test)))
        train_dict = {'text': X_train, 'label_text': Y_train}
        test_dict = {'text': X_test, 'label_text': Y_test}
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding=True, truncation=True, max_length=max_len)
            text_token['labels'] = np.array(
                tokenizer(examples['label_text'], padding=True, truncation=True, max_length=max_len)[
                    "input_ids"])  # 注意数据类型
            return text_token

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, test_dataset

    # 构建trainer
    def create_trainer(self, model, train_dataset, test_dataset, checkpoint_dir, batch_size):
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=15,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            labels = pred.label_ids[:, 3]
            preds = pred.predictions[:, 3].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            # tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        return trainer


def main():
    lct = LecCallTag()
    checkpoint_dir = "output/"
    text, label = read_dataset_prompt(config.train_file, config.max_length)
    # print(text[0])
    # print(label[0])
    model, tokenizer = lct.create_model_tokenizer("model/bert-base-chinese")
    train_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, config.max_length)
    trainer = lct.create_trainer(model, train_dataset, test_dataset, checkpoint_dir, config.batch_size)
    trainer.train()


if __name__ == '__main__':
    main()
