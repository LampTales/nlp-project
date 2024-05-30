import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

from config import LoraConfig, PromptConfig


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


SENTIMENT_BATCH_MUL = 1
PARAPHRASE_BATCCH_MUL = 4
SIMILARITY_BATCH_MUL = 1

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

PARAPHRASE_EMB = 768

SIMILARITY_EMB = 768

outer_args = None

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if config.option == 'pretrain':
            self.frozen_bert()
        elif config.option == 'finetune':
            self.unfrozen_bert()
        elif config.option == 'lora':
            self.frozen_bert()
            self.lora_config = LoraConfig(
                lora_rank=8,
                lora_dropout=0.1,
                lora_scaling=0.5
            )
            self.bert.lora_init(lora_config=self.lora_config)
        elif config.option == 'prompt':
            self.frozen_bert()
            self.prompt_config = PromptConfig(
                single_prompt_length=4,
                # stack_prompt=True,
                # batch_size=args.batch_size
            )
            self.bert.prompt_init(prompt_config=self.prompt_config)
            
        ### TODO
        # raise NotImplementedError
        
        # Sentiment classification
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, N_SENTIMENT_CLASSES)
        )

        # Paraphrase detection
        # self.paraphrase_tower = nn.Sequential(
        #     nn.Linear(BERT_HIDDEN_SIZE, 256),
        #     nn.ReLU(),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(256, PARAPHRASE_EMB)
        # )
        self.paraphrase_tower = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, 256),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, PARAPHRASE_EMB),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(PARAPHRASE_EMB, 2)
        )

        # Similarity detection
        self.similarity_tower = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, 256),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, SIMILARITY_EMB),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(SIMILARITY_EMB, 1)
        )
        
    
    def frozen_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
            
            
    def unfrozen_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # raise NotImplementedError
        return self.bert(input_ids, attention_mask)


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # raise NotImplementedError
        cls_emb = self.forward(input_ids, attention_mask)['pooler_output']
        logits = self.sentiment_classifier(cls_emb)
        return logits
        


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # raise NotImplementedError
        # emb_1 = self.forward(input_ids_1, attention_mask_1)['last_hidden_state'].mean(dim=1)
        # emb_1 = self.paraphrase_tower(emb_1)
        # emb_2 = self.forward(input_ids_2, attention_mask_2)['last_hidden_state'].mean(dim=1)
        # emb_2 = self.paraphrase_tower(emb_2)
        # # calculate cosine similarity
        # sim = F.cosine_similarity(emb_1, emb_2, dim=-1)
        # return sim

        emb_1 = self.forward(input_ids_1, attention_mask_1)['last_hidden_state'].mean(dim=1)
        emb_2 = self.forward(input_ids_2, attention_mask_2)['last_hidden_state'].mean(dim=1)
        emb = torch.cat((emb_1, emb_2), dim=-1)
        x = self.paraphrase_tower(emb)
        return x

        


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        # raise NotImplementedError
        emb_1 = self.forward(input_ids_1, attention_mask_1)['last_hidden_state'].mean(dim=1)
        emb_2 = self.forward(input_ids_2, attention_mask_2)['last_hidden_state'].mean(dim=1)
        emb = torch.cat((emb_1, emb_2), dim=-1)
        sim = self.similarity_tower(emb)
        return sim




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size*SENTIMENT_BATCH_MUL,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size*SENTIMENT_BATCH_MUL,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # paraphrase data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size*PARAPHRASE_BATCCH_MUL,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size*PARAPHRASE_BATCCH_MUL,
                                        collate_fn=para_dev_data.collate_fn)
      
    # similarity data
    simi_train_data = SentencePairDataset(sts_train_data, args)
    simi_dev_data = SentencePairDataset(sts_dev_data, args)
    
    simi_train_dataloader = DataLoader(simi_train_data, shuffle=True, batch_size=args.batch_size*SIMILARITY_BATCH_MUL,
                                        collate_fn=simi_train_data.collate_fn)
    simi_dev_dataloader = DataLoader(simi_dev_data, shuffle=False, batch_size=args.batch_size*SIMILARITY_BATCH_MUL,
                                        collate_fn=simi_dev_data.collate_fn)
    

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        sent_train_loss = 0
        para_train_loss = 0
        simi_train_loss = 0

        # train each task with one batch each time
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size*SENTIMENT_BATCH_MUL,
                                            collate_fn=sst_train_data.collate_fn)
        para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size*PARAPHRASE_BATCCH_MUL,
                                            collate_fn=para_train_data.collate_fn)
        simi_train_dataloader = DataLoader(simi_train_data, shuffle=True, batch_size=args.batch_size*SIMILARITY_BATCH_MUL,
                                           collate_fn=simi_train_data.collate_fn)
        
        num_batches = 0
        zip_iter = zip(sst_train_dataloader, para_train_dataloader, simi_train_dataloader)
        total_len = min(len(sst_train_dataloader), len(para_train_dataloader), len(simi_train_dataloader))
        
        for sent_batch, para_batch, simi_batch in tqdm(zip_iter, desc=f'train-{epoch}', disable=TQDM_DISABLE, total=total_len):
            
            optimizer.zero_grad()
            
            # sentiment
            b_ids, b_mask, b_labels = (sent_batch['token_ids'],
                                       sent_batch['attention_mask'], sent_batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            sent_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / (args.batch_size*SENTIMENT_BATCH_MUL)


            # paraphrase
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (para_batch['token_ids_1'],
                                                              para_batch['attention_mask_1'],
                                                              para_batch['token_ids_2'],
                                                              para_batch['attention_mask_2'],
                                                              para_batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)
            
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            para_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / (args.batch_size*PARAPHRASE_BATCCH_MUL)
            

            # similarity
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (simi_batch['token_ids_1'],
                                                              simi_batch['attention_mask_1'],
                                                              simi_batch['token_ids_2'],
                                                              simi_batch['attention_mask_2'],
                                                              simi_batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)
            
            b_labels = b_labels.float().view(-1).unsqueeze(-1)

            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            simi_loss = F.mse_loss(logits, b_labels, reduction='sum') / (args.batch_size*SIMILARITY_BATCH_MUL)
            
            
            total_loss = sent_loss + para_loss + simi_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            sent_train_loss += sent_loss.item()
            para_train_loss += para_loss.item()
            simi_train_loss += simi_loss.item()
            num_batches += 1
            
            
        train_loss = train_loss / (num_batches)
        sent_train_loss = sent_train_loss / num_batches
        para_train_loss = para_train_loss / num_batches
        simi_train_loss = simi_train_loss / num_batches


        # evaluate
        para_train_acc, para_train_y_pred, _, sent_train_acc, sent_train_y_pred, _, simi_train_corr, simi_train_y_pred, *_ = model_eval_multitask(
            sst_train_dataloader, para_train_dataloader, simi_train_dataloader, model, device)
        para_dev_acc, para_dev_y_pred, _, sentiment_dev_acc, sent_dev_y_pred, _, simi_dev_corr, simi_dev_y_pred, *_ = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, simi_dev_dataloader, model, device)
        
        print(f"Task: Sentiment, Epoch {epoch}: train loss :: {sent_train_loss :.3f}, train acc :: {sent_train_acc :.3f}, dev acc :: {sentiment_dev_acc :.3f}")
        print(f"Task: Paraphrase, Epoch {epoch}: train loss :: {para_train_loss :.3f}, train acc :: {para_train_acc :.3f}, dev acc :: {para_dev_acc :.3f}")
        print(f"Task: Similarity, Epoch {epoch}: train loss :: {simi_train_loss :.3f}, train corr :: {simi_train_corr :.3f}, dev corr :: {simi_dev_corr :.3f}")
        
        save_model(model, optimizer, args, config, args.filepath)

        if epoch == 0:
            nvi_out = os.popen('nvidia-smi')
            print(nvi_out.read())
            nvi_out.close()
        



def test_model(args):

    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune', 'lora', 'prompt'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'trained/{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    print(f'{args.option} :: batch_size: {args.batch_size}, para_mul: {PARAPHRASE_BATCCH_MUL}')
    outer_args = args
    train_multitask(args)
    test_model(args)
