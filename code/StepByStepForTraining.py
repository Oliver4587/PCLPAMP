import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from accelerate import Accelerator
from accelerate import dispatch_model
import os
from accelerate import notebook_launcher
from torch import autograd

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# This guide can only be run with the torch backend.



def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    if step_mode == 'linear':
        factor = (end_lr / start_lr - 1) / num_iter
        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter    
        def lr_fn(iteration):
            return np.exp(factor)**iteration    
    return lr_fn

class StepByStepForTraining(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class
        
        # We start by storing the arguments as attributes 
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.triple_loss_fn=None
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tcl_optimizer = None
        self.config = self.model.config
        # Let's send the model to the specified device right away
        # self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        self.test_loader = None
        self.writer = None
        self.scheduler = None
        self.is_batch_lr_scheduler = False
        self.clipping = None
        self.accelerater=None
        self.gradient_accumulation_steps=None
        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.total_epochs = 0
        
        self.visualization = {}
        self.handles = {}
        self.temp_max=2.0
        self.temp_min=0.01
        self.epochs_to_anneal=20
        self.temp=1.0
        self.best_test = 0.9300
        self.best_val = 0.9300

        # Creates the train_step function for our model, 
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()
        
    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_accelerater(self,gradient_accumulation_steps=None):
        # os.environ["NCCL_IB_DISABLE"] = "1"
        # os.environ['NCCL_P2P_DISABLE']="1"
        # self.accelerater=Accelerator(mixed_precision="fp16",gradient_accumulation_steps=gradient_accumulation_steps)
        self.accelerater=Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        # self.accelerater=Accelerator()
        self.gradient_accumulation_steps=gradient_accumulation_steps
        
    def set_tripletloss(self,loss_fn=None):
        self.triple_loss_fn=loss_fn
        # if self.triple_loss_fn.centers is not None:
        #     self.triple_loss_fn.centers.to(self.device)
            

    
    def set_loaders(self, train_loader, val_loader=None,test_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    
    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(input_ids,attention_mask,labels):
            # Sets model to TRAIN mode
            self.model.train()
            # self.model.to(self.device)


            # Step 1 - Computes our model's predicted output - forward pass
#             yhat = self.model(x)
            # with self.accelerator.accumulate(self.model):
            outputs = self.model(input_ids, attention_mask)
            # Step 2 - Computes the loss
            loss = self.loss_fn(outputs.logits, labels)
            # Step 3 - Computes gradients
            self.accelerater.backward(loss)
#             loss.backward()
            
            if callable(self.clipping):
                self.clipping()
            
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(input_ids,attention_mask,labels):
            # Sets model to EVAL mode
            self.model.eval()
            # self.model.to(self.device)
            
            # Step 1 - Computes our model's predicted output - forward pass
            outputs = self.model(input_ids, attention_mask)
            # Step 2 - Computes the loss
            loss = self.loss_fn(outputs.logits, labels)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn
            

    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None
            
        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []

        if validation:
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                mini_batch_losses.append(mini_batch_loss)
        else:
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                with self.accelerater.accumulate(self.model):
                    mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                    mini_batch_losses.append(mini_batch_loss)
                    
            
            
            

        loss = np.mean(mini_batch_losses)
        return loss

    def train_classifier(self, n_epochs, best_acc=0.9367):
        # self.model.transformer.gradient_checkpointing_enable()
        # self.model.gradient_checkpointing_enable()
        self.model, self.optimizer ,self.train_loader, self.val_loader, self.test_loader = self.accelerater.prepare(self.model, self.optimizer, self.train_loader,self.val_loader, self.test_loader)
        best_acc=best_acc
        for epoch in range(n_epochs):
            print(f"epoch{epoch+1}:",end=', ',flush=True)
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1
            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
                #compute accuracy
                train_acc = self.evaluate_accuracy(self.train_loader)
                val_acc = self.evaluate_accuracy(self.val_loader)
                test_acc = self.evaluate_accuracy(self.test_loader)

            if self.total_epochs>=20:
                if val_acc > best_acc:
                    best_acc = val_acc
                    filename = f"../saved_model/Margin{self.model.config.margin}_Contrasive_ProtT5_Prompt{self.model.config.pre_seq_len}_lambda{self.model.config.lam}_acc:"
                    print(f"saved val_acc {val_acc:.4f} test_acc:{test_acc:.4f}")
                    self.save_checkpoint(filename)
            
#             self._epoch_schedulers(val_loss)
            print(f"loss:{loss:.4f},val_loss{val_loss:.4f},train_acc{train_acc:.4f},val_acc{val_acc:.4f},test_acc{test_acc:.4f}")
            
            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            # Closes the writer
            self.writer.close()


                    

    
    def _batch_matrix(self,data_loader):
        prelabel, relabel = [], []
        for input_ids,attention_mask,label in data_iter:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            label = label.to(self.device)
            outputs = self.model(input_ids,attention_mask)
            prelabel.append(outputs.argmax(dim=1).cpu().numpy())
            relabel.append(y.cpu().numpy())
            
        prelabel = [np.concatenate(prelabel)]
        relabel = [np.concatenate(relabel)]
        prelabel = np.array(prelabel)
        relabel = np.array(relabel)
        prelabel = prelabel.reshape(-1, 1)
        relabel = relabel.reshape(-1, 1)
    
        df1 = pd.DataFrame(prelabel, columns=['prelabel'])
        df2 = pd.DataFrame(relabel, columns=['realabel'])
        df4 = pd.concat([df1, df2], axis=1)
    
    
        acc_sum, n = 0.0, 0
        outputs = []
        for input_ids,attention_mask,label in data_iter:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            label = label.to(self.device)
            output = torch.softmax(self.model(input_ids,attention_mask), dim=1)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        pre_pro = outputs[:, 1]
        pre_pro = np.array(pre_pro.cpu().detach().numpy())
        pre_pro = pre_pro.reshape(-1)
        df3 = pd.DataFrame(pre_pro, columns=['pre_pro'])
        df5 = pd.concat([df4, df3], axis=1)
        real1 = df5['realabel']
        pre1 = df5['prelabel']
        pred_pro1 = df5['pre_pro']
        metric1, roc_data1, prc_data1 = self.caculate_metric(pre1, real1, pred_pro1)
        return metric1, roc_data1, prc_data1


    def _make_triplelet_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(input_ids,attention_mask,labels):
            # Sets model to TRAIN mode
            self.model.train()
            # self.model.to(self.device)


            # Step 1 - Computes our model's predicted output - forward pass
#             yhat = self.model(x)
            # with self.accelerator.accumulate(self.model):
            outputs = self.model(input_ids, attention_mask)
            # features = self.model.get_feature(input_ids, attention_mask)
            # Step 2 - Computes the loss
            loss = self.triple_loss_fn(outputs, labels)
            # loss = self.triple_loss_fn(features, labels)
            # Step 3 - Computes gradients
            self.accelerater.backward(loss)
#             loss.backward()
            
            if callable(self.clipping):
                self.clipping()
            
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_triplelet_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(input_ids,attention_mask,labels):
            # Sets model to EVAL mode
            self.model.eval()
            # self.model.to(self.device)
            
            # Step 1 - Computes our model's predicted output - forward pass
            outputs = self.model(input_ids, attention_mask)
            # features = self.model.get_feature(input_ids, attention_mask)
            # Step 2 - Computes the loss
            loss = self.triple_loss_fn(outputs, labels)
            # loss = self.triple_loss_fn(features, labels)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn
            

    def _triple_mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self._make_triplelet_val_step_fn()
        else:
            data_loader = self.train_loader
            step_fn =  self._make_triplelet_train_step_fn()

        if data_loader is None:
            return None
            
        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []

        if validation:
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                mini_batch_losses.append(mini_batch_loss)
        else:
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                with self.accelerater.accumulate(self.model):
                    mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                    mini_batch_losses.append(mini_batch_loss)
                    
            
            
            

        loss = np.mean(mini_batch_losses)
        return loss




    
    def train_accelerate_triplelet(self,n_epochs):
        self.model, self.optimizer, self.train_loader ,self.val_loader  = self.accelerater.prepare(self.model, self.optimizer, self.train_loader,self.val_loader)
        # best_acc=0.91
        for epoch in range(n_epochs):
            print(f"epoch{epoch+1}:",end=', ',flush=True)
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self. _triple_mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self ._triple_mini_batch(validation=True)
                self.val_losses.append(val_loss)
                
            self._epoch_schedulers(val_loss) 

            print(f"loss:{loss:.4f},val_loss{val_loss:.4f}")
                      
            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)
        # self.model.save_pretrained(f"../saved_model/TripleletProtBert_Prompt_{self.model.config.pre_seq_len}_{self.model.config.margin}")
        # self.model.save_pretrained(f"../saved_model/TripleletProtT5_Prompt_{self.model.config.pre_seq_len}_{self.model.config.margin}")
        self.model.save_pretrained(f"../saved_model/TripleletProtT5_Prompt_{self.model.config.pre_seq_len}_{self.model.config.margin}_stage1")
        if self.writer:
            # Closes the writer
            self.writer.close()


    def _make_combined_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(input_ids,attention_mask,labels):
            # Sets model to TRAIN mode
            self.model.train()
            # self.model.to(self.device)


            # Step 1 - Computes our model's predicted output - forward pass
#             yhat = self.model(x)
            # with self.accelerator.accumulate(self.model):
            outputs = self.model(input_ids, attention_mask)
            features = self.model.get_feature(input_ids, attention_mask)
            # Step 2 - Computes the loss
            # print(outputs.logits,labels)
            triple_loss = self.triple_loss_fn(features, labels)
            # print(triple_loss)
            binary_loss = self.loss_fn(outputs.logits,labels)
            # print(binary_loss)
            # loss = (triple_loss + binary_loss)
            # loss = (triple_loss*0.3 + binary_loss)
            loss = (triple_loss*self.config.lam + binary_loss)
            # Step 3 - Computes gradients
            self.accelerater.backward(loss)
#             loss.backward()
            
            if callable(self.clipping):
                self.clipping()
            
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.tcl_optimizer.step()
            self.optimizer.zero_grad()
            self.tcl_optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_combined_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(input_ids,attention_mask,labels):
            # Sets model to EVAL mode
            self.model.eval()
            # self.model.to(self.device)
            
            # Step 1 - Computes our model's predicted output - forward pass
            outputs = self.model(input_ids, attention_mask)
            features = self.model.get_feature(input_ids, attention_mask)
            # Step 2 - Computes the loss
            triple_loss = self.triple_loss_fn(features, labels)
            binary_loss = self.loss_fn(outputs.logits,labels)
            loss = (triple_loss*self.config.lam + binary_loss)
            # loss = binary_loss
            return loss.item()

        return perform_val_step_fn
            

    def _combined_mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self._make_combined_val_step_fn()
        else:
            data_loader = self.train_loader
            step_fn =  self._make_combined_train_step_fn()

        if data_loader is None:
            return None
            
        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []

        if validation:
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                mini_batch_losses.append(mini_batch_loss)
        else:
            # with autograd.detect_anomaly():
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                with self.accelerater.accumulate(self.model):
                    mini_batch_loss = step_fn(input_ids, attention_mask,labels)
                    mini_batch_losses.append(mini_batch_loss)
                    
                    
                #Calls the learning rate scheduler at the end of every mini-batch update
                self._mini_batch_schedulers(i / n_batches)
                    
            
            

        loss = np.mean(mini_batch_losses)
        return loss



    
    def train_accelerate_combined(self,n_epochs,best_acc=None):
        self.model, self.triple_loss_fn, self.optimizer, self.tcl_optimizer, self.train_loader ,self.val_loader, self.test_loader  = self.accelerater.prepare(self.model, self.triple_loss_fn, self.optimizer, self.tcl_optimizer,self.train_loader,self.val_loader, self.test_loader)
        # device = accelerator.device
        # self.model = dispatch_model(self.model, device_map="auto")
        if best_acc is not None:
            best_acc = best_acc
            self.best_val = best_acc
            self.best_test = best_acc
        else:
            best_acc=0.9283
        
        for epoch in range(n_epochs):
            print(f"epoch{epoch+1}:",end=', ',flush=True)
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self. _combined_mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self ._combined_mini_batch(validation=True)
                self.val_losses.append(val_loss)
                train_acc = self.evaluate_accuracy(self.train_loader)
                val_acc = self.evaluate_accuracy(self.val_loader)
                test_acc = self.evaluate_accuracy(self.test_loader)
            
            self._epoch_schedulers(val_loss)

            #save the model
            if self.total_epochs>=20:
                if val_acc > best_acc:
                    best_acc = val_acc
                    filename = f"../saved_model/Margin{self.model.config.margin}_Contrasive_ProtT5_Prompt{self.model.config.pre_seq_len}_lambda{self.model.config.lam}_acc:"
                    print(f"saved val_acc {val_acc:.4f} test_acc:{test_acc:.4f}")
                    # filename=f"../saved_model/Margin{self.model.config.margin}_Contrasive_ProtT5_Prompt{self.model.config.pre_seq_len}_lambda{self.model.config.lambda}_acc:{best_acc:.4f}/TCL_loss.pth"
                    # torch.save(checkpoint, filename)
                    # self.save_checkpoint(filename)
                    self.save_prefix_checkpoint(filename)
                
            # self._epoch_schedulers(val_loss)
            print(f"loss:{loss:.4f},val_loss{val_loss:.4f},train_acc{train_acc:.4f},val_acc{val_acc:.4f},test_acc{test_acc:.4f}")
            
                      
            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)
        # self.model.save_pretrained(f"../saved_model/TripleletProtBert_Prompt_{self.model.config.pre_seq_len}_{self.model.config.margin}")
        if self.writer:
            # Closes the writer
            self.writer.close()
        
   def save_prefix_checkpoint(self, filename):
            # Builds dictionary with all elements for resuming training
    
        self.model.config.save_pretrained(filename)
        checkpoint = {'epoch': self.total_epochs,
                      'model_prefix_state_dict': self.model.prefix_encoder.state_dict(),
                      'model_classifier_state_dict': self.model.classifier.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      "tcl_optimizer_state_dict":self.tcl_optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses,
                      "TCL_loss":self.triple_loss_fn.state_dict()
                     }
        # self.model.save_pretrained(filename)
        checkpoint_filename = filename + "/checkpoint.pth"


        torch.save(checkpoint, checkpoint_filename)

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      # 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      "tcl_optimizer_state_dict":self.tcl_optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses,
                      "TCL_loss":self.triple_loss_fn.state_dict()
                     }
        self.model.save_pretrained(filename)
        checkpoint_filename = filename + "/checkpoint.pth"

        torch.save(checkpoint, checkpoint_filename)


    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint_filename = filename + "/checkpoint.pth"
        checkpoint = torch.load(checkpoint_filename)

        # Restore state for model and optimizer
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.prefix_encoder.load_state_dict(checkpoint['model_prefix_state_dict'])
        self.model.classifier.load_state_dict(checkpoint['model_classifier_state_dict'])
        self.triple_loss_fn.load_state_dict(checkpoint['TCL_loss'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tcl_optimizer.load_state_dict(checkpoint["tcl_optimizer_state_dict"])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train() # always use TRAIN for resuming training   

    def predict(self, input_ids,attention_mask):
        # Set is to evaluation mode for predictions
        self.model.eval() 
        # Takes a Numpy input and make it a float tensor
        input_ids_tensor = torch.as_tensor(input_ids)
        attention_mask_tensor=torch.as_tensor(attention_mask)
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(input_ids_tensor.to(self.device),attention_mask_tensor.to(self.device)).logits
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def correct(self, x1,x2, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x1.to(self.device),x2.to(self.device)).logits
        y = y.to(self.device)
        # _,label=torch.max(y,1)
        label = y
        self.model.train()

        # We get the size of the batch and the number of classes 
        # (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:        
            # In a multiclass classification, the biggest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax, 
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, we NEED to check if the
            # last layer is a sigmoid (and then it produces probs)
            if isinstance(self.model, nn.Sequential) and \
               isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            # or something else (logits), which we need to convert
            # using a sigmoid
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (label == c).sum().item()
            n_correct = (predicted[label == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)


    def evaluate_accuracy(self,data_loader):
        # self.model.eval()
        device = self.device
        acc_sum, n = 0.0, 0
        for i,batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids,attention_mask).logits 

            acc_sum += (outputs.argmax(dim=1) == labels).float().sum().item()
            n += labels.shape[0]
        return acc_sum / n

    def caculate_metric(self,pred_y, labels, pred_prob):
    
        test_num = len(labels)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
    
        for index in range(test_num):
            if int(labels[index]) == 1:
                if labels[index] == pred_y[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if labels[index] == pred_y[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1
    
    
        ACC = float(tp + tn) / test_num
    
        # precision
        if tp + fp == 0:
            Precision = 0
        else:
            Precision = float(tp) / (tp + fp)
    
        # SE
        if tp + fn == 0:
            Recall = Sensitivity = 0
        else:
            Recall = Sensitivity = float(tp) / (tp + fn)
    
        # SP
        if tn + fp == 0:
            Specificity = 0
        else:
            Specificity = float(tn) / (tn + fp)
    
        # MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    
        # F1-score
        if Recall + Precision == 0:
            F1 = 0
        else:
            F1 = 2 * Recall * Precision / (Recall + Precision)
    
        # ROC and AUC
        labels = list(map(int, labels))
        pred_prob = list(map(float, pred_prob))
        fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
        AUC = auc(fpr, tpr)
    
        # PRC and AP
        precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
        AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    
        metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])
    
        # ROC(fpr, tpr, AUC)
        # PRC(recall, precision, AP)
        roc_data = [fpr, tpr, AUC]
        prc_data = [recall, precision, AP]
        return metric, roc_data, prc_data
    
    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(batch['input_ids'],batch['attention_mask'],batch['labels']) for i, batch in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)
        return results


    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(self.model.state_dict()), 
                           'optimizer': deepcopy(self.optimizer.state_dict())}
        # Retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If there are more iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        self.model.to(self.device)
        while (iteration < num_iter):
            # That's the typical mini-batch inner loop
            for i,batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # Step 1
                yhat = self.model(input_ids, attention_mask)
                features = self.model.get_feature(input_ids, attention_mask)
                # Step 2 - Computes the loss
                loss_triplelet = self.triple_loss_fn(features, labels)
                # Step 2
                loss_binary = self.loss_fn(yhat.logits, labels)
                loss = (loss_triplelet*0.3 + loss_binary)
                # loss = (loss_triplelet + loss_binary)
                # Step 3
                # print(loss)
                loss.backward()

                # Here we keep track of the losses (smoothed)
                # and the learning rates
                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1-alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                # Number of iterations reached
                if iteration == num_iter:
                    break

                # Step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # Restores the original states
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_tcl_optimizer(self,tcl_optmizer):
        self.tcl_optimizer = tcl_optmizer
        

    def set_lr_scheduler(self, scheduler):
        # Makes sure the scheduler in the argument is assigned to the
        # optimizer we're using in this class
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or
                isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)
        
    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)