import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.models.t5.modeling_t5 import T5PreTrainedModel
from transformers import T5EncoderModel,T5Config
from transformers.modeling_outputs import SequenceClassifierOutput
from math import sqrt
from Classification_Head import ClassifierModel

class ProtT5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = T5EncoderModel(config)
        dropout=config.classifier_dropout
        self.dropout = torch.nn.Dropout(dropout)
        # self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.d_model, config.num_labels)
        self.model_parallel = False
        # self.gradient_checkpointing = True
        # dropout=0.1
        # self.classifier =  nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(config.d_model, config.num_labels),
        #     nn.Tanh(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(config.ff_units, config.num_labels)
        # )
        self.classifier = ClassifierModel()

        for param in self.transformer.parameters():
            param.requires_grad = False
        
        bert_param = 0
        for name, param in self.transformer.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
         
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


        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,

        )


        
        hidden_state = outputs.hidden_states
        # print(hidden_state.shape)
        last_hidden_state = outputs.last_hidden_state
        contextual_embedding_last = last_hidden_state[:,:200,:]
        contextual_embedding_first = outputs.hidden_states[1][:,:200,:]
        contextual_embedding = contextual_embedding_last+contextual_embedding_first
        input_features = torch.mean(contextual_embedding,dim=1)

        logits = self.classifier(input_features)

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
            attentions=outputs.attentions,
        )