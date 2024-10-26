import torch
from torch._C import NoopLogger
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.models.t5.modeling_t5 import T5PreTrainedModel
from transformers import AutoModelForSequenceClassification,T5EncoderModel,T5Config
from transformers.modeling_outputs import SequenceClassifierOutput
from math import sqrt
import logging
from Classification_Head import Conv2D_Layer1,TransformerModel,ClassifierModel

class ProtT5PrefixForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = CustomT5EncoderModel(config)
        dropout=config.classifier_dropout
        self.dropout = torch.nn.Dropout(dropout)
        # self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.d_model, config.num_labels)
        self.classifier = ClassifierModel()

        for param in self.transformer.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_layers
        self.n_head = config.num_heads
        self.n_embd = config.d_kv

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoderForT5(config)

        all_param=0
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                all_param += param.numel()
        print('total param is {}'.format( all_param)) 
         
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer.device)
        # prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_feature(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
        # prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len)
        # print(attention_mask.device,prefix_attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # print(attention_mask.shape)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        # hidden_state = outputs.last_hidden_state
        # # print(hidden_state.shape)
        # pooled_output = torch.mean(hidden_state[:,:200,:],dim=1)

        hidden_states = outputs.hidden_states
        # print(hidden_states[1].shape)
        last_hidden_state = outputs.last_hidden_state
        # print(len(hidden_states))
        contextual_embedding_last = last_hidden_state[:,:200,:]
        contextual_embedding_first = outputs.hidden_states[1][:,:200,:]
        contextual_embedding = contextual_embedding_last+contextual_embedding_first
        
        input_features = torch.mean(contextual_embedding,dim=1)
        # input_features = torch.mean( contextual_embedding_last,dim=1)
        
        return input_features#shape batch_size,max_length,config.d_model
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_features = self.get_feature(
           input_ids=input_ids,
        attention_mask= attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask= head_mask,
        inputs_embeds= inputs_embeds,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states= output_hidden_states,
        return_dict=return_dict,
        )

        logits = self.classifier(input_features)
        # logits = self.classifier(pooled_output)
        # print("logits in the forward",logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # hidden_states=pooled_output,
            # attentions=outputs.attentions,
        )


class CustomT5EncoderModel(T5EncoderModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        past_key_values = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.ERROR)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        return encoder_outputs


class PrefixEncoderForT5(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.d_kv*config.num_heads)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_layers * 2 * config.d_kv*config*num_heads)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_layers * 2 * config.d_kv*config.num_heads)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values