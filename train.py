import argparse
import torch
import os
import pickle
import random
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from utils.loader import load_logger, load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.preprocess_data import preprocess_dataset
from accelerate.utils import InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import default_data_collator
from easy_context import prepare_dataloader
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
)


class CD_POS_TRAIN:
    """
    Train LLMs with CD_Pos recipe.
    """
    def __init__(
        self, 
        output_path, 
        log_path, 
        model_path, 
        data_path,
        rts_path, 
        rss_path, 
        learning_rate, 
        gradient_accumulate_every, 
        epoch, 
        batch_size, 
        parallel_mode, 
        setting, 
        stage,  
        seq_length, 
        target_length, 
        num_proc, 
        seed=None,
        
    ):
        # logger
        self.logger, self.tqdm_out = load_logger(log_path)
        # set_seed
        self.seed = seed
        set_seed(self.seed)
        
        # training setting
        self.setting = setting
        
        # paths
        self.model_path = model_path
        self.output_path = output_path
        self.data_path = data_path
        self.rss_path = rss_path
        self.rts_path = rts_path
        
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
        
        # parallel_mode
        self.parallel_mode = parallel_mode
    
        # training params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.epoch = epoch
        self.stage = stage

        # config
        self.seq_length = seq_length
        self.target_length = target_length
        self.num_proc = num_proc
        
    
    def load_dataset_util(self, stage, data_path, tokenizer, batch_size, accelerator, model_type):
        try:
            train_dataset = load_dataset(data_path)
        except:
            train_dataset = load_dataset('json', data_files=data_path)['train']


        if stage == 0:
            train_dataset = preprocess_dataset(train_dataset, 
                                                tokenizer=tokenizer,
                                                model_type=model_type, 
                                                setting_choice=self.setting,
                                                seq_length_=self.seq_length, 
                                                target_len=self.target_length,
                                                rts_file=self.rts_path, 
                                                rss_s_file=self.rss_path, 
                                                num_proc=self.num_proc
                                            )
            print("Dataset Size:", len(train_dataset))
            
            train_loader = DataLoader(
                train_dataset,
                collate_fn=default_data_collator,
                shuffle=False,
                batch_size=batch_size,
            )
            
            train_dataset_loader = prepare_dataloader(self.parallel_mode, train_loader, accelerator)


        if stage == 1:
            train_dataset = preprocess_dataset(train_dataset, 
                                                tokenizer=tokenizer,
                                                setting_choice='full_str',
                                                model_type=self.model_path, 
                                                seq_length_=self.seq_length, 
                                                target_len=self.target_length,
                                                rts_file=self.rts_path, 
                                                rss_s_file=self.rss_path, 
                                                num_proc=self.num_proc
                                            )
            print("Dataset Size:", len(train_dataset))
        
            train_loader = DataLoader(
                train_dataset,
                collate_fn=default_data_collator,
                shuffle=False,
                batch_size=batch_size,
            )
            train_dataset_loader = prepare_dataloader(self.parallel_mode, train_loader, accelerator)
        
        return train_dataset_loader
        
    
    def prepare_accelerator(self, stage):

        # accelerator
        timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
        
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulate_every if stage == 0 else self.gradient_accumulate_every,
            mixed_precision="bf16",
            kwargs_handlers=[timeout],
            # fsdp_plugin=fsdp_plugin,
        )        
        return accelerator
        
    def prepare_model_and_tokenizer(self, stage, accelerator):
        if stage == 0:
            model, tokenizer = load_model_and_tokenizer(self.model_path, accelerator)

        elif stage == 1:
            # model, tokenizer = load_model_and_tokenizer(self.model_path, accelerator)
            model, tokenizer = load_model_and_tokenizer(self.output_path + '/stage_' + str(stage-1) , accelerator)


        assert isinstance(
            model, (transformers.LlamaForCausalLM, transformers.MistralForCausalLM)
        ), "Only support llama and mistral model"
        model_type = (
            "llama" if isinstance(model, transformers.LlamaForCausalLM) else "mistral"
        )
        apply_seq_parallel_monkey_patch(self.parallel_mode, model_type)
        
        return model, tokenizer
    
    def prepare_scheduler(self, stage, optim, num_training_steps):
        # scheduler
        scheduler = DummyScheduler(
            optim,
            num_training_steps=num_training_steps,
            total_num_steps=num_training_steps,
        )
        return scheduler

    
    def prepare_optimizer(self, stage, model, learning_rate):
        
        optim = DummyOptim(model.parameters(), lr=learning_rate)
        return optim
        
    def prepare_and_check_params(self, stage, epoch, train_dataset_loader, gradient_accumulate_every):
        
        # calculate training steps
        num_training_steps = math.ceil(epoch * train_dataset_loader.dataset.shape[0] / gradient_accumulate_every)
        print(num_training_steps)
        return num_training_steps
        
    def prepare_loss_fn(self):
        loss_func = CrossEntropyLoss(inplace_backward=True)
        return loss_func

    def prepare_modules(self, stage, lr, data_path, epoch, batch_size, gradient_accumulate_every):
        """
        prepare accelerator, model, tokenizer, dataloader, optimizer, scheduler
        """
        
        # if stage == 0:
        accelerator = self.prepare_accelerator(stage=stage)
        model, tokenizer = self.prepare_model_and_tokenizer(stage, accelerator)
        train_data_loader = self.load_dataset_util(stage=stage, data_path=data_path, tokenizer=tokenizer, batch_size=batch_size, accelerator=accelerator, model_type=self.model_path)
        optim = self.prepare_optimizer(stage=stage, model=model, learning_rate=lr)
        num_training_steps = self.prepare_and_check_params(stage=stage, epoch=epoch, train_dataset_loader=train_data_loader, gradient_accumulate_every=gradient_accumulate_every)
        scheduler = self.prepare_scheduler(stage, optim, num_training_steps)
        model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
        
        model.gradient_checkpointing_enable()
        accelerator.register_for_checkpointing(scheduler)
        accelerator.print(f"Max train epoches: {epoch}, Max train steps: {num_training_steps}")

        progress_bar = tqdm(
            range(num_training_steps), file=self.tqdm_out, mininterval=1, disable=not accelerator.is_local_main_process
        )
        
        loss_func = self.prepare_loss_fn()
        
        return model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar

        
    def train(self, stage, model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar):
        completed_steps = 0
        for idx, batch in enumerate(train_data_loader):
            
            input_ids = batch["input_ids"][0][..., :-1].unsqueeze(dim=0)
            target_ids = batch["input_ids"][0][..., 1:].unsqueeze(dim=0)
            if stage != 1:
                try:
                    position_ids = batch["position_ids"][0][..., : input_ids.shape[-1]].unsqueeze(dim=0)
                except:
                    print('Position idx error, check your pkl in train.py')
                    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)    

                prepared = prepare_seq_parallel_inputs(
                    self.parallel_mode,
                    input_ids,
                    position_ids,
                    target_ids,
                    accelerator.process_index,
                    accelerator.num_processes,
                    accelerator.device,
                )

            else:
                position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
                prepared = prepare_seq_parallel_inputs(
                    self.parallel_mode,
                    input_ids,
                    position_ids,
                    target_ids,
                    accelerator.process_index,
                    accelerator.num_processes,
                    accelerator.device,
                )

            local_input_ids = prepared["local_input_ids"]
            local_position_ids = prepared["local_position_ids"]
            local_target_ids = prepared["local_target_ids"]

            loss_log = None
            
            with accelerator.accumulate(model):
                logits = model(
                    local_input_ids,
                    position_ids=local_position_ids,
                ).logits
                loss = loss_func(
                    logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
                )
                try:
                    accelerator.backward(loss)
                except:
                    self.logger.log(msg='wrong step', level=0)

                if accelerator.sync_gradients:
                    gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                    loss_log = {
                        "loss": gathered_loss.item(),
                        "ppl": math.exp(gathered_loss.item()),
                    }
                    accelerator.log(loss_log, step=completed_steps)

                optim.step()
                scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if loss_log is not None:
                    progress_bar.set_postfix(loss_log)
                completed_steps += 1
        return model, accelerator
    
    def finish_training(self, stage, model, accelerator):
        accelerator.print(f"Training Finished")
        accelerator.end_training()

        if self.output_path is not None:
            accelerator.print(f"Saving model to {self.output_path}, stage: {stage}")

            accelerator.wait_for_everyone()

            state_dict = accelerator.get_state_dict(model)

            accelerator.unwrap_model(model).save_pretrained(
                f"{self.output_path + '/stage_' + str(stage) + '/'}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )

            accelerator.print(f"Saving Finished")
    
    def merge_model(self, stage, model_1_path, model_2_path):
        
        model_1 = AutoModelForCausalLM.from_pretrained(model_1_path)
        model_2 = AutoModelForCausalLM.from_pretrained(model_2_path)
        
        model_1_params = list(model_1.named_parameters())
        model_2_params = list(model_2.named_parameters())

        assert len(model_1_params) == len(model_2_params), "The two models do not have the same number of parameters"

        delta_params = {}

        for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
            delta_params[name_1] = (param_2.data + param_1.data) / 2


        new_model = AutoModelForCausalLM.from_pretrained(model_1_path)


        for name, param in new_model.named_parameters():
            if name in delta_params:
                param.data = delta_params[name]


        new_model.save_pretrained(self.output_path + '/stage_' + str(stage) + '/',)

    
    def train_with_stage(self):
        stage = self.stage
    
        if stage == 0:
            
            model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar = self.prepare_modules(stage=stage, lr=self.learning_rate, data_path=self.data_path, epoch=self.epoch, batch_size=self.batch_size, gradient_accumulate_every=self.gradient_accumulate_every)                
            model.train()
            model, accelerator = self.train(stage, model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar)
            self.finish_training(stage, model, accelerator)
        
        elif stage == 1:
            model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar = self.prepare_modules(stage=stage, lr=self.learning_rate, data_path=self.data_path, epoch=self.epoch, batch_size=self.batch_size, gradient_accumulate_every=self.gradient_accumulate_every)                
            model.train()
            model, accelerator = self.train(stage, model, accelerator, train_data_loader, loss_func, optim, scheduler, progress_bar)
            self.finish_training(stage, model, accelerator)
            
            
        elif stage == 2:
            self.merge_model(stage, self.output_path + '/stage_' + str(stage-1) + '/', self.model_path)
        



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--epoch", type=int, default=1)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--data_path",type=str, default=None)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument("--model-base-length", type=int, default=8192)
    args.add_argument("--target-length", type=int, default=128000)
    args.add_argument("--setting", type=str, default=None)
    args.add_argument("--rts-path", type=str, default=None)
    args.add_argument("--rss-path", type=str, default=None)
    args.add_argument("--log-path", type=str, default=None)
    args.add_argument("--stage", type=int, default=3)
    args.add_argument("--num_proc", type=int, default=1)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    
    args = args.parse_args()
    
    cd_pos_train = CD_POS_TRAIN(
        output_path=args.output_dir,  
        log_path=args.log_path, 
        model_path=args.model, 
        data_path=args.data_path, 
        rts_path=args.rts_path, 
        rss_path=args.rss_path, 
        seed=args.seed,   
        learning_rate=args.learning_rate, 
        gradient_accumulate_every=args.gradient_accumulate_every, 
        epoch=args.epoch, 
        batch_size=args.batch_size, 
        parallel_mode=args.parallel_mode, 
        stage=args.stage, 
        setting=args.setting, 
        seq_length=args.seq_length, 
        target_length=args.target_length, 
        num_proc=args.num_proc, 
        )

    cd_pos_train.train_with_stage()