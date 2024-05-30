import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

import sys
sys.path.append("../")
from models.PAMStereo.PASMnet import PASMnet
import json

from tqdm import tqdm
import argparse
import accelerate
from HuggingFace_Trainer.dataset_configuration import prepare_dataset,covered_resultion
import math

import ast
from diffusers.optimization import get_scheduler
from utils.AverageMeter import AverageMeter
from losses.pam_loss import PAMStereoLoss,PAMLoss_OutSide,PAMLoss_Center_All,PAMLoss_Center_All_in_All


# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger = get_logger(__name__, log_level="INFO")
import  matplotlib.pyplot as plt

from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss,Disaprity_EPE_OOF,Disparity_EPE_Occ


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Stereo Matching Training")
    
    parser.add_argument(
        "--datapath",
        type=str,
        default="/data1/liu/Sintel/",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--trainlist",
        type=str,
        default="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Train_Sub_List.txt",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )
    parser.add_argument(
        "--vallist",
        type=str,
        default="/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Val_Sub_List.txt",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        required=True,
        help="Specify the dataset name used for training/validation.",
    )

    parser.add_argument(
        "--datathread",
        type=int,
        default=4,
        required=True,
        help="Specify the dataset name used for training/validation.",
    )

    parser.add_argument(
        "--visible_list",
        type=str,
        default="sdsada",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )

    parser.add_argument(
        "--maxdisp",
        type=int,
        default=192,
        required=True,
        help="Specify the dataset name used for training/validation.",
    )
    
    # training epochs and steps
    parser.add_argument("--num_train_epochs", type=int, default=70)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")

    # accumulation steps
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),)


    
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),)

    parser.add_argument(
        "--seed",
        type=int,
        default=10086,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),)

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),)

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.")


    parser.add_argument(
        "--loss_type",
        type=str,
        default="select from 'simple', 'plusoutside','pluscenter','plusall'",
        help="The output directory where the model predictions and checkpoints will be written.")

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args


def Inference_on_MPI_Test(stereo_matching_network,
                                 test_loader,
                                 args,accelerator,
                                 weight_dtype,epoch):
    
    stereo_matching_network=accelerator.unwrap_model(stereo_matching_network)    
    
    EPE_OP = Disparity_EPE_Loss
    EPE_OCC_OP = Disparity_EPE_Occ
    EPE_OFF_OP = Disaprity_EPE_OOF
    
    EPEs_Meter = AverageMeter()
    EPEs_OCC_Meter = AverageMeter()
    EPEs_OFF_Meter  = AverageMeter()

    img_nums = 0
    for i, sample_batched in enumerate(test_loader):
        left_image_data = sample_batched['final_left'] # left image
        right_image_data = sample_batched['final_right'] # right pose
        
        gt_disp_data = sample_batched['disp'].unsqueeze(1)
        gt_occlusion_data = sample_batched['occlusions'].unsqueeze(1)
        gt_outof_frame_data = sample_batched['outofframe'].unsqueeze(1)
        
        gt_disp_data = covered_resultion(gt_disp_data)
        gt_occlusion_data = covered_resultion(gt_occlusion_data)
        gt_outof_frame_data = covered_resultion(gt_outof_frame_data)
        
        left_image_data = covered_resultion(left_image_data)
        right_image_data = covered_resultion(right_image_data)
        

        with torch.no_grad():
            
            
            output = stereo_matching_network(left_image_data,right_image_data)

            
            img_nums += left_image_data.shape[0]
            
            # print(target_disp.shape)
            EPE_Value = EPE_OP(output, gt_disp_data)
            EPE_OCC_Value = EPE_OCC_OP(output, gt_disp_data,gt_occlusion_data)
            EPE_OFF_Value = EPE_OFF_OP(output, gt_disp_data,gt_outof_frame_data)
            

        if EPE_Value.data.item() == EPE_Value.data.item():
            EPEs_Meter.update(EPE_Value.data.item(), left_image_data.size(0))
            
        if EPE_OCC_Value.data.item() == EPE_OCC_Value.data.item():
            EPEs_OCC_Meter.update(EPE_OCC_Value.data.item(), left_image_data.size(0))
            
        if EPE_OFF_Value.data.item() == EPE_OFF_Value.data.item():
            EPEs_OFF_Meter.update(EPE_OFF_Value.data.item(), left_image_data.size(0))

        if i % 1 == 0:
            logger.info('Test: [{0}/{1}]\t EPE {2}'
                    .format(i, len(test_loader), EPEs_Meter.val))

    logger.info(' * DISP EPE {:.3f}'.format(EPEs_Meter.avg))
    logger.info(' * EPE Occ {:.3f}'.format(EPEs_OCC_Meter.avg))
    logger.info(' * EPE OOF {:.3f}'.format(EPEs_OFF_Meter.avg))

    results_dict = dict()
    results_dict['epe'] = EPEs_Meter.avg
    results_dict['epe_occ'] = EPEs_OCC_Meter.avg
    results_dict['epe_off'] = EPEs_OFF_Meter.avg


    
    os.makedirs(os.path.join(args.output_dir,'Performance'),exist_ok=True)
    saved_json = os.path.join(os.path.join(args.output_dir,'Performance'),'epoch_{}.json'.format(epoch))
    # Writing JSON data
    with open(saved_json, 'w') as file:
        json.dump(results_dict, file, indent=4)
        
    return results_dict
    


def main():
    ''' ------------------------Configs Preparation----------------------------'''
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
            
    #--------------------------  Loaded the Pretrained Models -------------------------------#
    stereo_matching_network = PASMnet()
    logger.info("initial stereo matching network",main_process_only=True)
    
    
    if args.resume_from_checkpoint:
        ckpts = torch.load(args.resume_from_checkpoint)
        model_ckpt = ckpts['model_state']
        stereo_matching_network.load_state_dict(model_ckpt)
    

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    # set to train
    stereo_matching_network.train()


    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # optimizer settings
    optimizer = optimizer_cls(
        stereo_matching_network.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    with accelerator.main_process_first():
        
        train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath=args.datapath,
                                                                         trainlist=args.trainlist,
                                                                         vallist=args.vallist,
                                                                         batch_size=args.batch_size,
                                                                         datathread=args.datathread,
                                                                         targetHW=(440,1024),
                                                                         visible_list=ast.literal_eval(args.visible_list))
        
        logger.info("Loading the train loader, test loader",main_process_only=True)

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

    # Prepare everything with our `accelerator`.
    stereo_matching_network, optimizer, train_loader, test_loader,lr_scheduler = accelerator.prepare(
        stereo_matching_network, optimizer, train_loader, test_loader,lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        
    
    stereo_matching_network = stereo_matching_network.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Here is the DDP training: actually is 4
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    
    initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # directly inference here
    if accelerator.is_main_process:
        stereo_matching_network.eval()
        # inference at here
        results = Inference_on_MPI_Test(stereo_matching_network=stereo_matching_network,
                                     test_loader=test_loader,args=args,
                                     accelerator=accelerator,weight_dtype=weight_dtype,
                                     epoch=0)


    losses_meter = AverageMeter()
    disp_epe_meter = AverageMeter()
    disp_epe_occ_meter = AverageMeter()
    disp_epe_out_of_frame_meter = AverageMeter()
    
    Best_EPE = 100
    Best_EPE_OCC = 100
    Best_EPE_OFF = 100
    
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        stereo_matching_network.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(stereo_matching_network):

                left_image_data = batch['final_left'] # left image
                right_image_data = batch['final_right'] # right pose
                
                gt_disp_data = batch['disp'].unsqueeze(1)
                gt_occlusion_data = batch['occlusions'].unsqueeze(1)
                gt_outof_frame_data = batch['outofframe'].unsqueeze(1)
                
                

                if args.loss_type=='plus_outside':
                    left_left_image_data = batch["rendered_left_left"]
                    right_right_image_data = batch["rendered_right_right"]
                
                if args.loss_type=='plus_center_all':
                    med_image_data = batch['rendered_med']
                    one_third_data = batch['rendered_one_third']
                    two_third_data = batch['rendered_two_third']

                if args.loss_type=='plus_center_all_in_all':
                    med_image_data = batch['rendered_med']
                    one_third_data = batch['rendered_one_third']
                    two_third_data = batch['rendered_two_third']
                    left_left_image_data = batch["rendered_left_left"]
                    right_right_image_data = batch["rendered_right_right"]

            
                # inference here
                output,attn_list,att_cycle,valid_mask = stereo_matching_network(left_image_data,right_image_data,192)
                
                if args.loss_type=='simple':
                    # loss here
                    loss, loss_P, loss_S, loss_PAM = PAMStereoLoss(left_image_data,
                                                                   right_image_data,
                                                                   disp=output,
                                                                   att=attn_list,
                                                                   att_cycle=att_cycle,
                                                                   valid_mask=valid_mask,
                                                                   disp_gt=None)

                if args.loss_type=='plus_outside':
                    loss, loss_P, loss_S, loss_PAM = PAMLoss_OutSide(img_left=left_image_data,
                                                                    img_right=right_image_data,
                                                                    disp=output,
                                                                    att=attn_list,
                                                                    att_cycle=att_cycle,
                                                                    valid_mask=valid_mask,
                                                                    disp_gt=None,
                                                                    img_left_left=left_left_image_data,
                                                                    img_right_right=right_right_image_data)
                if args.loss_type=='plus_center_all':
                    loss, loss_P, loss_S, loss_PAM = PAMLoss_Center_All(img_left=left_image_data,
                                                                        img_right=right_image_data,
                                                                        disp=output,
                                                                        att=attn_list,
                                                                        att_cycle=att_cycle,
                                                                        valid_mask=valid_mask,
                                                                        disp_gt=None,
                                                                        img_center=med_image_data,
                                                                        img_one_third=one_third_data,
                                                                        img_two_one_third=two_third_data)

                if args.loss_type=='plus_center_all_in_all':
                    loss, loss_P, loss_S, loss_PAM = PAMLoss_Center_All_in_All(img_left=left_image_data,
                                                                               img_right=right_image_data,
                                                                               disp=output,
                                                                               att=attn_list,
                                                                               att_cycle=att_cycle,
                                                                               valid_mask=valid_mask,
                                                                               disp_gt=None,
                                                                               
                                                                               img_center=med_image_data,
                                                                               img_one_third=one_third_data,
                                                                               img_two_one_third=two_third_data,
                                                                               img_left_left=left_left_image_data,
                                                                               img_right_right=right_right_image_data)                 
                    
                    
                    # only for metric
                    disp_epe_value = Disparity_EPE_Loss(output,gt_disp_data)
                    disp_epe_occ_value = Disparity_EPE_Occ(output,gt_disp_data,gt_occlusion_data)
                    disp_epe_oof_value = Disaprity_EPE_OOF(output,gt_disp_data,gt_outof_frame_data)
                    
                
                
    
                losses_meter.update(loss.data.item(), left_image_data.size(0))
                disp_epe_meter.update(disp_epe_value.data.item(),left_image_data.size(0))
                disp_epe_occ_meter.update(disp_epe_occ_value.data.item(),left_image_data.size(0))
                disp_epe_out_of_frame_meter.update(disp_epe_oof_value.data.item(),left_image_data.size(0))
                
                
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(stereo_matching_network.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
  
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % 10 == 0:
                    accelerator.log({"train_loss": losses_meter.val}, step=global_step)
                    logger.info("current_loss: {},\t disp epe: {}, \t disp_occ_epe: {} \t disp_oof_epe: {}".format(losses_meter.val,
                                                                                                                   disp_epe_meter.val,
                                                                                                                   disp_epe_occ_meter.val,
                                                                                                                   disp_epe_out_of_frame_meter.val),
                                main_process_only=True)
                train_loss = 0.0
                

                        
            
            # logger.info(" step loss:{} \t learning rate {}".format(loss.detach().item(),
            #                                                        lr_scheduler.get_last_lr()[0]),main_process_only=True)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break

        
        # do inference here
        if accelerator.is_main_process:
            logger.info("#################################")
            logger.info("Inference Here At Epoch {}".format(epoch))
            stereo_matching_network.eval()
            # inference at here
            results = Inference_on_MPI_Test(stereo_matching_network=stereo_matching_network,
                                        test_loader=test_loader,args=args,
                                        accelerator=accelerator,weight_dtype=weight_dtype,
                                        epoch=epoch)
            
            current_epe = results['epe']
            current_epe_off = results['epe_off']
            current_epe_occ = results['epe_occ']
            
            if current_epe<Best_EPE:
                Best_EPE = current_epe
                unwarped_model = accelerator.unwrap_model(stereo_matching_network)
                torch.save(
                    {"model_state":unwarped_model.state_dict(),
                    "epoch":epoch,
                    "best_score": current_epe,
                        },
                    os.path.join(args.output_dir,f"best_EPE_model.pt"))

            if current_epe_off<Best_EPE_OFF:
                Best_EPE_OFF = current_epe_off
                unwarped_model = accelerator.unwrap_model(stereo_matching_network)
                torch.save(
                    {"model_state":unwarped_model.state_dict(),
                    "epoch":epoch,
                    "best_score": current_epe_off,
                        },
                    os.path.join(args.output_dir,f"best_EPE_OFF_model.pt"))

            if current_epe_occ<Best_EPE_OCC:
                Best_EPE_OCC = current_epe_occ
                unwarped_model = accelerator.unwrap_model(stereo_matching_network)
                torch.save(
                    {"model_state":unwarped_model.state_dict(),
                    "epoch":epoch,
                    "best_score": current_epe_occ,
                        },
                    os.path.join(args.output_dir,f"best_EPE_OCC_model.pt"))
    
            
            
            


    
    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()

                
                
              

if __name__=="__main__":
    main()


