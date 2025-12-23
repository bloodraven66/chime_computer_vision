# from transformers.trainer import *
# from typing import Callable, Dict, List, Optional, Tuple, Union, Type

# class AVSRTrainer(Trainer):
    
#     def __init__(
#         self,
#         model: Union[PreTrainedModel, nn.Module] = None,
#         args: TrainingArguments = None,
#         data_collator: any = None,
#         valid_data_collator: any = None,
#         train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
#         eval_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
#         processing_class: Optional[
#             Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
#         ] = None,
#         model_init: Optional[Callable[[], PreTrainedModel]] = None,
#         compute_loss_func: Optional[Callable] = None,
#         compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
#         callbacks: Optional[List[TrainerCallback]] = None,
#         optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
#         optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
#         preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
#         max_eval_samples: Optional[int] = None,
#         print_predictions: bool = False,
#     ):
#         super().__init__(
#             model=model,
#             args=args,
#             data_collator=data_collator,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             tokenizer=processing_class,
#             model_init=model_init,
#             compute_loss_func=compute_loss_func,
#             compute_metrics=compute_metrics,
#             callbacks=callbacks,
#             optimizers=optimizers,
#             optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
#             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#         )
#         self.valid_data_collator = valid_data_collator
#         self.max_eval_samples = max_eval_samples
#         self.print_predictions = print_predictions
#         self.eval_sample_count = 0

#     def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
#         """
#         Returns the evaluation [`~torch.utils.data.DataLoader`].

#         Subclass and override this method if you want to inject some custom behavior.

#         Args:
#             eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
#                 If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
#         """
#         if eval_dataset is None and self.eval_dataset is None:
#             raise ValueError("Trainer: evaluation requires an eval_dataset.")

#         # If we have persistent workers, don't do a fork bomb especially as eval datasets
#         # don't change during training
#         dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
#         if (
#             hasattr(self, "_eval_dataloaders")
#             and dataloader_key in self._eval_dataloaders
#             and self.args.dataloader_persistent_workers
#         ):
#             return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

#         eval_dataset = (
#             self.eval_dataset[eval_dataset]
#             if isinstance(eval_dataset, str)
#             else eval_dataset
#             if eval_dataset is not None
#             else self.eval_dataset
#         )
#         if self.max_eval_samples is not None:
#             if hasattr(eval_dataset, 'take'):
#                 eval_dataset = eval_dataset.take(self.max_eval_samples)
#             elif hasattr(eval_dataset, 'select'):
#                 eval_dataset = eval_dataset.select(range(min(self.max_eval_samples, len(eval_dataset))))
        

#         data_collator = self.valid_data_collator

#         if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
#             eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
#         else:
#             data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

#         dataloader_params = {
#             "batch_size": self.args.eval_batch_size,
#             "collate_fn": data_collator,
#             "num_workers": self.args.dataloader_num_workers,
#             "pin_memory": self.args.dataloader_pin_memory,
#             "persistent_workers": self.args.dataloader_persistent_workers,
#         }

#         if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
#             dataloader_params["drop_last"] = self.args.dataloader_drop_last
#             dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

#         # accelerator.free_memory() will destroy the references, so
#         # we need to store the non-prepared version
#         eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
#         if self.args.dataloader_persistent_workers:
#             if hasattr(self, "_eval_dataloaders"):
#                 self._eval_dataloaders[dataloader_key] = eval_dataloader
#             else:
#                 self._eval_dataloaders = {dataloader_key: eval_dataloader}

#         return self.accelerator.prepare(eval_dataloader)


from transformers.trainer import *
from typing import Callable, Dict, List, Optional, Tuple, Union, Type
import torch

class AVSRTrainer(Trainer):
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: any = None,
        valid_data_collator: any = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_eval_samples: Optional[int] = None,
        print_predictions: bool = False,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.valid_data_collator = valid_data_collator
        self.max_eval_samples = max_eval_samples
        self.print_predictions = print_predictions

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override to add printing during eval
        """
        args = self.args
        
        # Call parent's evaluation_loop but wrap it to print step-by-step
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        if self.print_predictions:
            print("\n" + "="*60)
            print(f"Starting {description}")
            print("="*60)
        
        # Store original compute_loss
        original_compute_loss = self.compute_loss
        step_counter = [0]  # Use list to allow modification in nested function
        
        def compute_loss_with_logging(model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            if self.print_predictions and self.args.process_index == 0:
                step_counter[0] += 1
                print(f"\n--- Eval Step {step_counter[0]} ---")
                print(f"  Loss: {loss.item():.4f}")
                if hasattr(outputs, 'loss_ctc') and outputs.loss_ctc is not None:
                    print(f"  Loss CTC: {outputs.loss_ctc.item():.4f}")
                if hasattr(outputs, 'loss_att') and outputs.loss_att is not None:
                    print(f"  Loss ATT: {outputs.loss_att.item():.4f}")
                if hasattr(outputs, 'acc') and outputs.acc is not None:
                    print(f"  Accuracy: {outputs.acc:.4f}")
                            
            return (loss, outputs) if return_outputs else loss
        
        # Temporarily replace compute_loss
        self.compute_loss = compute_loss_with_logging
        
        try:
            result = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )
        finally:
            # Restore original compute_loss
            self.compute_loss = original_compute_loss
        
        if self.print_predictions:
            print("\n" + "="*60)
            print(f"Finished {description}")
            print("="*60 + "\n")
        
        return result

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        
        if self.max_eval_samples is not None:
            if hasattr(eval_dataset, 'take'):
                eval_dataset = eval_dataset.take(self.max_eval_samples)
            elif hasattr(eval_dataset, 'select'):
                eval_dataset = eval_dataset.select(range(min(self.max_eval_samples, len(eval_dataset))))

        data_collator = self.valid_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)