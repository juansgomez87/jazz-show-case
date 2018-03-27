# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 09:40:32 2018

@author: Juan Sebastián Gómez
"""

import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
eps = np.finfo(np.float32).eps

def concatenate_files(path_to_files, list_inst):
    full_audio = list()
    for num, inst in enumerate(list_inst):
        for root, dirs, files in os.walk(path_to_files):
            for file in files:
                if file.startswith(inst):
                    full_file = os.path.join(path_to_files, file)
                    data, sr = librosa.load(full_file, sr=22050, mono=True)
                    data /= np.max(np.abs(data))
                    print(file)
                    full_audio.extend(data.tolist())
    return np.array(full_audio)

def create_spectrogram(audio):
    # normalize data
    audio /= np.max(np.abs(audio))
    audio = np.squeeze(audio)
    # short time fourier transform
    D = np.abs(librosa.stft(audio, win_length=1024, hop_length=512, center=True))
    # mel frequency representation
    S = librosa.feature.melspectrogram(S=D, sr=22050, n_mels=128)
    # natural logarithm
    ln_S = np.log(S + eps)
    # create tensor
    seg_dur = 43 # segment duration eq to 1 second
    spec_list = list()
    for idx in range(0, ln_S.shape[1] - seg_dur + 1, seg_dur):
        spec_list.append(ln_S[:, idx:(idx+seg_dur)])
    # print('Number of spectrograms:', len(spec_list))
    X = np.expand_dims(np.array(spec_list), axis=1)
    return ln_S, X

def plot_everything(list_inst, full_audio, ln_S, org_pred, agg_pred):
    time_sec = np.linspace(0, 60, full_audio.shape[0])
    fontsize = 10
    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,4), sharex=True)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,6), sharex=True)
    ax[0].imshow(ln_S, origin='lower', aspect='auto', extent=[0,60,0,128], interpolation='nearest')
    ax[0].set_title('Melspectrogram', fontsize=fontsize+1)
    ax[0].set_ylabel('Mel-bands', fontsize=fontsize)
    ax[1].imshow(org_pred, aspect='auto', interpolation='nearest')
    # plot ground truth
    color = '#ffffff'
    for j in range(0, 6):
        ax[1].add_patch(patches.Rectangle(((j*10)-0.5, j-0.5), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[1].set_title('Segment Predictions', fontsize=fontsize+1)
    ax[1].set_ylabel('Labels' , fontsize=fontsize)
    ax[1].set_yticks(np.arange(len(list_inst)))
    ax[1].set_yticklabels(list_inst, fontsize=fontsize)
    ax[2].imshow(agg_pred, aspect='auto', interpolation='nearest')
    # plot ground truth
    for j in range(0, 6):
        ax[2].add_patch(patches.Rectangle(((j*10)-0.5, j-0.5), 10, 1, fill=False, edgecolor=color, linewidth=2))
    ax[2].set_title('Aggregated Predictions', fontsize=fontsize+1)
    ax[2].set_ylabel('Labels', fontsize=fontsize)
    ax[2].set_yticks(np.arange(len(list_inst)))
    ax[2].set_yticklabels(list_inst, fontsize=fontsize)
    ax[2].set_xlabel('Seconds', fontsize=fontsize)
    plt.tight_layout()
    fig.savefig('jazz_show_case.png', bbox_inches='tight')
    plt.show()

def load_model(filename, path_to_model):
    if filename.find('solo') > 0:
        json_filename = os.path.join(path_to_model, 'model_transfer_solo.json')
        weights_filename = os.path.join(path_to_model, 'model_transfer_solo.hdf5')
    elif filename.find('solo') < 0:
        json_filename = os.path.join(path_to_model, 'model_transfer_mix.json')
        weights_filename = os.path.join(path_to_model, 'model_transfer_mix.hdf5')
    with open(json_filename, 'r') as json_file:
        loaded_model = json_file.read()
    model = model_from_json(loaded_model)
    model.load_weights(weights_filename)
    print('Model', json_filename, ' loaded!')
    return model

def organize_predictions(list_inst, labels_inst, norm_pred):
    org_pred = np.zeros(norm_pred.shape)
    for num, inst in enumerate(list_inst):
        for key in labels_inst.keys():
            if inst == labels_inst[key]:
                org_pred[num, :] = norm_pred[key, :]
    return org_pred

def aggregate_predictions(pred):
    agg_pred = np.zeros(pred.shape)
    chunk = 10
    for i in range(pred.shape[0]):
        agg_pred[:, i*chunk : (i+1)*chunk] = np.tile(np.sum(pred[:, i*chunk : (i+1)*chunk], axis=1) / chunk, (chunk, 1)).T
        agg_pred[:, i*chunk : (i+1)*chunk] /= np.max(agg_pred[:, i*chunk : (i+1)*chunk], axis=0)
    return agg_pred
    

if __name__ == '__main__':
    path_to_audio = './audio/'
    path_to_models = './models/'
    mode = 1

    list_inst = ['as','ts','ss','tb','tp','cl']
    labels_inst = {0: 'as', 1: 'cl', 2: 'ss', 3: 'tb', 4: 'tp', 5: 'ts'}

    if mode == 0:
        # concatenate mix files
        full_audio = concatenate_files(os.path.join(path_to_audio, 'original/'), list_inst)
        librosa.output.write_wav(os.path.join(path_to_audio, 'all_audio.wav'), full_audio, sr=22050)
        # concatenate solo files
        full_audio_solo = concatenate_files(os.path.join(path_to_audio, 'solo/'), list_inst)
        librosa.output.write_wav(os.path.join(path_to_audio, 'all_audio_solo.wav'), full_audio_solo, sr=22050)

    else:
        # predict files
        from keras.models import model_from_json
        sel = int(input('Select audio: [1] mix, [2] solo\n'))
        if sel == 1:
            filename = 'all_audio.wav'
        else:
            filename = 'all_audio_solo.wav'
        
        # load full_audio
        full_audio, sr = librosa.load(os.path.join(path_to_audio, filename))
        #extract spectrograms
        ln_S, X = create_spectrogram(full_audio)
        # load prediction model
        model = load_model(filename, path_to_models)
        # make predictions
        pred = model.predict(X)
        # organize predictions
        org_pred = organize_predictions(list_inst, labels_inst, pred.T)
        # aggregate
        agg_pred = aggregate_predictions(org_pred)
        # plot!
        plot_everything(list_inst, full_audio, ln_S, org_pred, agg_pred)
