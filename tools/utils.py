# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-04-07 15:51:18
#
# Created on Wed Oct 20 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import nibabel as nb
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import subprocess

from einops.layers.torch import Rearrange
from einops import rearrange

from tools.dataloader import *

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from warmup_scheduler import GradualWarmupScheduler


# no need to get data path, as we are working in a bounded subspace of just HCPYA
'''
def get_data_path(config):

    dataloader = config['data']['dataloader']
    dataset = config['data']['dataset']
    task = config['data']['task']
    configuration = config['data']['configuration']
    modality = config['data']['modality']
    sampling = config['mesh_resolution']['sampling']
    registration = config['data']['registration']

    if str(dataloader) == 'metrics':
        if dataset == 'dHCP':
            data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/ico_6_{}'.format(sampling),'base','regression_{}_space_features'.format(configuration))
        elif dataset == 'UKB':
            if modality == 'cortical_metrics':
                if task == 'scan_age_msmall' or task == 'sex_msmall':
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_msmall/')
                else:
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample/')
            elif modality == 'fMRI':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_timeseries/clean_vn/')
        elif dataset == 'HCP':
            if modality == 'cortical_metrics':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'merged_metrics/ico6_{}_{}'.format(sampling,configuration), modality)
            elif modality == 'memory_task':
                data_path = os.path.join(config['data']['path_to_data'],dataset,'merged_metrics/ico6_{}_{}'.format(sampling,configuration), '{}_{}'.format(modality, registration))
    elif str(dataloader) == 'bold':
        if dataset == 'HCP':
            if modality == 'tfMRI':
                if config['training']['runtime']:
                    data_path = os.path.join(config['data']['path_to_data'],'HCP','movie_task/fmri_7T_1.6')
                else:
                    data_path = os.path.join(config['data']['path_to_data'],'HCP','movie_frames')
            elif modality == 'rfMRI' or 'smooth_rfMRI':
                if config['training']['runtime']:
                    data_path = os.path.join(config['data']['path_to_data'],'HCP','rest')
                else:
                    data_path = os.path.join(config['data']['path_to_data'],'HCP','rest_frames')

    else:
        raise('not implemented yet')
    
    return data_path

'''

def get_dataloaders(config, 
                    data_path):

    dataloader = config['data']['dataloader']
    #sampler = config['training']['sampler']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    modality = config['data']['modality']
    runtime = config['training']['runtime']

    #if str(dataloader)=='metrics':
    #    if str(modality) == 'cortical_metrics' or str(modality) == 'memory_task':
    #        train_loader, val_loader, test_loader = loader_metrics(data_path,sampler,config)
    #        return train_loader, val_loader, test_loader
    if str(dataloader)=="numpy":
            
        train_data = np.load(os.path.join(data_path,'train_data.npy'))
        train_label = np.load(os.path.join(data_path,'train_labels.npy'))

        mean = train_data.mean()
        std = train_data.std()
        print(f'train mean: {mean:.4f} | std: {std:.4f}\n')

            

        val_data = np.load(os.path.join(data_path,'validation_data.npy'))
        val_label = np.load(os.path.join(data_path,'validation_labels.npy'))


        test_data = np.load(os.path.join(data_path,'test_data.npy'))
        test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)
        

        ## normalization
        eps = 1e-8
        train_data = (train_data - mean) / (std + eps)
        val_data   = (val_data   - mean) / (std + eps)
        test_data  = (test_data  - mean) / (std + eps)

        print('training data: {}'.format(train_data.shape))
        print('validation data: {}'.format(val_data.shape))
        print('testing data: {}'.format(test_data.shape))
        print('')

        train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                            torch.from_numpy(train_label).float())

        train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                        batch_size = bs,
                                                        shuffle=True,
                                                        num_workers=16)
            
        val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                          torch.from_numpy(val_label).float())

    
        val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                                batch_size = bs_val,
                                                shuffle=False,
                                                num_workers=16)

        test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                           torch.from_numpy(test_label).float())

        test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                                batch_size = bs_val,
                                                shuffle=False,
                                                num_workers=16)
        
        return train_loader, val_loader, test_loader

    else:
            raise('not implemented yet')
    '''
    elif str(dataloader)=='bold' or str(dataloader)=='fmri':
        print('loading functional data')
        if modality == 'tfMRI':
            if runtime:
                train_loader = loader_tfmri_runtime(data_path,config)
            else:
                train_loader = loader_tfmri(data_path,config)
        elif (modality == 'rfMRI') or (modality == 'smooth_rfMRI'):
            if runtime: 
                train_loader = loader_rfmri_runtime(data_path,config)
            else:
                train_loader = loader_rfmri(data_path,config)
        return train_loader
    '''
    
    
def get_dimensions(config):

    modality = config['data']['modality']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    if config['MODEL'] in ['sit','ms-sit']:    
        channels = config['transformer']['channels']
    elif config['MODEL']== 'spherical-unet':
        channels = config['spherical-unet']['channels']
    num_channels = len(channels)

    
    if config['MODEL'] in ['sit','ms-sit'] and (modality == 'cortical_metrics' ):

        T = num_channels
        N = num_patches
        V = num_vertices

        use_bottleneck = False
        bottleneck_dropout = 0.0

        print(f'Number of channels {T}; Number of patches {N}; Number of vertices {V}')
        print(f'Using bottleneck {use_bottleneck}; Dropout bottleneck {bottleneck_dropout}')
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout

def get_scheduler(config, max_iterations ,optimizer):

    if config['optimisation']['use_scheduler']:

        print('Using learning rate scheduler')

        if config['optimisation']['scheduler'] == 'StepLR':

            scheduler = StepLR(optimizer=optimizer,
                                step_size= config['StepLR']['stepsize'],
                                gamma= config['StepLR']['decay'])
        
        elif config['optimisation']['scheduler'] == 'CosineDecay':

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                    T_max = config['CosineDecay']['T_max'],
                                                                    eta_min= config['CosineDecay']['eta_min'],
                                                                    )

        elif config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                            mode='max',
                                            factor=0.5,
                                            patience=2,
                                            cooldown=0,
                                            min_lr=0.0000001
                                                )

        if config['optimisation']['warmup']:

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
     
    else:
        # to use warmup without fancy scheduler
        if config['optimisation']['warmup']:
            scheduler = StepLR(optimizer,
                                step_size=max_iterations)

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
        else:

            return None
            
    return scheduler


'''
def load_weights_imagenet(state_dict,state_dict_imagenet,nb_layers):

    state_dict['mlp_head.0.weight'] = state_dict_imagenet['norm.weight'].data
    state_dict['mlp_head.0.bias'] = state_dict_imagenet['norm.bias'].data

    # transformer blocks
    for i in range(nb_layers):
        state_dict['transformer.layers.{}.0.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm1.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm2.bias'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_qkv.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.qkv.weight'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_out.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.fn.to_out.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.3.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.3.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.bias'.format(i)].data

    return state_dict
'''


def save_gifti(data, filename):
    gifti_file = nb.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(data))
    nb.save(gifti_file,filename)


def logging_sit(config, pretraining=False):
     #{dataset},{modality},{task},{grid resolution},{configuration}
    if pretraining:
        folder_to_save_model = f"{config['logging']['folder_to_save_model']}/pretraining"
    else:
        folder_to_save_model = f"{config['logging']['folder_to_save_model']}/train"
    
    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)

    if config['transformer']['dim'] == 192:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 384:
        folder_to_save_model = folder_to_save_model + '-small'
    elif config['transformer']['dim'] == 768:
        folder_to_save_model = folder_to_save_model + '-base'
    elif config['transformer']['dim'] == 96:
        folder_to_save_model = folder_to_save_model + '-very-tiny'
    elif config['transformer']['dim'] == 48:
        folder_to_save_model = folder_to_save_model + '-ultra-tiny'
    

    if config['training']['init_weights']!=False:
        folder_to_save_model = folder_to_save_model + f'-{config['training']['init_weights']}'

    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    if config['SSL'] == 'smae':
        folder_to_save_model = folder_to_save_model + f'-mask-{config['pretraining_smae']['mask_prob']}'
        
    folder_to_save_model = folder_to_save_model + f'-bs-{config['training']['bs']}'

    if config['training']['restart']:
        folder_to_save_model = folder_to_save_model + '-restart'
        
    return folder_to_save_model

'''
def plot_regression_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([20,40],[20,40],color='r')
    plt.savefig(path_to_save)


def plot_handedness_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([-100,100],[-100,100],color='r')
    plt.savefig(path_to_save)

def plot_iq_results_HCP(predictions, targets,folder_to_save, epoch):


    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    path_to_save = os.path.join(folder_to_save,'results_{}.png'.format(epoch))

    plt.figure(figsize=(10,10))
    plt.plot(predictions,targets,'*',color='b')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.plot([5,27],[5,27],color='r')
    plt.savefig(path_to_save)
'''

def save_reconstruction_mae(reconstructed_batch,
                            reconstructed_batch_unmasked,
                            inputs, 
                            num_patches,
                            num_vertices,
                            ico_grid,
                            num_channels,
                            masked_indices,
                            unmasked_indices,
                            epoch,
                            folder_to_save_model,
                            split,
                            path_to_workdir,
                            id,
                            server,
                            ):


    recon_folder = os.path.join(folder_to_save_model, 'reconstruction', f'{split}')
    if not os.path.exists(recon_folder):
        os.mkdir(recon_folder)
        print(f'Creating folder: {recon_folder}')



    ### reconstructing the original ico-6 sphere from the learned embeddings
    indices = pd.read_csv('{}/patch_extraction/msm/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,ico_grid))

    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    new_inputs = np.transpose(inputs.cpu().numpy(),(0,2,1,3))

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(recon_folder, f'original_sphere_{epoch}_{id}.shape.gii'))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstruction', split, 'original_sphere_{}_{}.shape.gii'.format(epoch,id)), 'CORTEX_LEFT'])
        p1.wait()

    B, num_masked_patch, V = reconstructed_batch.shape
    rearrange_layer_masked = Rearrange('b m (v c) -> b c m v', b=B, m=num_masked_patch, c=num_channels, v=num_vertices)

    B, num_unmasked_patch, V = reconstructed_batch_unmasked.shape
    rearrange_layer_unmasked = Rearrange('b m (v c) -> b c m v', b=B, m=num_unmasked_patch, c=num_channels, v=num_vertices)

    batch = rearrange_layer_masked(reconstructed_batch).cpu().numpy()
    batch_unmasked = rearrange_layer_unmasked(reconstructed_batch_unmasked).cpu().numpy()

    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        #import pdb;pdb.set_trace()
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch[0,:,ind,:].transpose()
        elif i in unmasked_indices:
            ind =  (unmasked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch_unmasked[0,:,ind,:].transpose()
        else:
            print(f'issue with indices: {i}')

    #import pdb;pdb.set_trace()
    save_gifti(reconstructed_sphere, os.path.join(recon_folder, f'recon_{split}_{epoch}_it_{id}.shape.gii'))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstruction', split, 'recon_{}_{}_it_{}.shape.gii'.format(split,id,epoch)), 'CORTEX_LEFT'])
        p1.wait()

    reconstructed_sphere_mask_only = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere_mask_only[indices_to_extract,:] = batch[0,:,ind,:].transpose()

    #import pdb;pdb.set_trace()
    save_gifti(reconstructed_sphere_mask_only, os.path.join(recon_folder, f'recon_mask_{split}_{epoch}_it_{id}.shape.gii'))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstruction', split, 'recon_mask_{}_{}_it_{}.shape.gii'.format(split,id,epoch)), 'CORTEX_LEFT'])
        p1.wait()

    sphere_patched = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            sphere_patched[indices_to_extract,:] = 0
        else:
            sphere_patched[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    #import pdb;pdb.set_trace()
    save_gifti(sphere_patched, os.path.join(recon_folder, f'sphere_patched_{epoch}_{id}.shape.gii'))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstruction', split, 'sphere_patched_{}_{}.shape.gii'.format(epoch,id)), 'CORTEX_LEFT'])
        p1.wait()

def save_reconstruction_mae_test(reconstructed_batch,
                                reconstructed_batch_unmasked,
                                inputs, 
                                num_patches,
                                num_vertices,
                                ico_grid,
                                num_channels,
                                masked_indices,
                                unmasked_indices,
                                folder_to_save_model,
                                id,
                                split,
                                path_to_workdir,
                                hemi,
                                server,
                                ):

    try:
        os.makedirs(os.path.join(folder_to_save_model, 'reconstructions_test_time', '{}'.format(split)),exist_ok=False)
        print('Creating folder: {}'.format(os.path.join(folder_to_save_model, 'reconstructions_test_time', '{}'.format(split))))
    except OSError:
        pass

    indices = pd.read_csv('{}/patch_extraction/msm/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,ico_grid))

    original_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    new_inputs = np.transpose(inputs.cpu().numpy(),(0,2,1,3))

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        original_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(original_sphere, os.path.join(folder_to_save_model,'reconstructions_test_time','{}'.format(split), '{}_{}.sphere.shape.gii'.format(id,hemi)))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.sphere.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
        p1.wait()

    B, num_masked_patch, V = reconstructed_batch.shape
    rearrange_layer_masked = Rearrange('b m (v c) -> b c m v', b=B, m=num_masked_patch, c=num_channels, v=num_vertices)

    B, num_unmasked_patch, V = reconstructed_batch_unmasked.shape
    rearrange_layer_unmasked = Rearrange('b m (v c) -> b c m v', b=B, m=num_unmasked_patch, c=num_channels, v=num_vertices)

    batch = rearrange_layer_masked(reconstructed_batch).cpu().numpy()
    batch_unmasked = rearrange_layer_unmasked(reconstructed_batch_unmasked).cpu().numpy()

    reconstructed_sphere = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch[0,:,ind,:].transpose()
        elif i in unmasked_indices:
            ind =  (unmasked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere[indices_to_extract,:] = batch_unmasked[0,:,ind,:].transpose()
        else:
            print('issue with indices: {}'.format(i))

    #import pdb;pdb.set_trace()
    save_gifti(reconstructed_sphere, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split),'{}_{}.reconstruction.shape.gii'.format(id,hemi)))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.reconstruction.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
        p1.wait()

    reconstructed_sphere_mask_only = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            ind =  (masked_indices[0] == i).nonzero(as_tuple=True)[0][0].cpu().numpy()
            reconstructed_sphere_mask_only[indices_to_extract,:] = batch[0,:,ind,:].transpose()

    save_gifti(reconstructed_sphere_mask_only, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split),'{}_{}.reconstruction_mask_only.shape.gii'.format(id,hemi)))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.reconstruction_mask_only.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
        p1.wait()

    sphere_patched = np.zeros((40962,num_channels),dtype=np.float32)

    for i in range(num_patches):
        indices_to_extract = indices[str(i)].values
        if i in masked_indices:
            sphere_patched[indices_to_extract,:] = 0
        else:
            sphere_patched[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()

    save_gifti(sphere_patched, os.path.join(folder_to_save_model,'reconstructions_test_time', '{}'.format(split), '{}_{}.sphere_masked.shape.gii'.format(id,hemi)))

    if not server:
        p1 = subprocess.Popen(['/home/sd20/software/workbench/bin_linux64/wb_command', '-set-structure',os.path.join(folder_to_save_model, 'reconstructions_test_time', split, '{}_{}.sphere_masked.shape.gii'.format(id,hemi)), 'CORTEX_LEFT'])
        p1.wait()

