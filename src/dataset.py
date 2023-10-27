import config
import torch
class BERTDataset:
    def __init__(self, review , target):
        self.review = review
        self.target = target
        self.tokernizer = config.TOKENIZER
        self.Max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,item):
        review = str(self.review)
        review = " ".join(review.split())

        input = self.tokenizer.encode_plus(
            review,
            None,
            add_special_token = True,
            max_length = self.Max_len

        )

        ids = input["input_ids"]
        mask = input["attention_mask"]
        token_type_ids = input["token_type_ids"]

        padding_length  = self.Max_len - len(ids)
        ids = ids +([0]* padding_length)
        mask = mask +([0]* padding_length)
        token_type_ids = token_type_ids +([0]* padding_length)

        return{
            'ids' : torch.Tensor(ids , dtype=torch.long),
           'mask': torch.Tensor(mask , dtype=torch.long),
           'token_type_ids': torch.Tensor(token_type_ids, dtype=torch.long),
           'target' :  torch.Tensor(self.target[item], dtype= torch.float)           
        }