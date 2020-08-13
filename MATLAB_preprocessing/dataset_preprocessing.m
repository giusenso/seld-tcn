%% Neural Network - Project
% Author:   Giuseppe Sensolini Arra'
% date:     August 2020
% brief:    create Mel-spectrograms and MFCCs from the given dataset

close all
clear all
clc

% audio paths
audio_folder_path = 'ov1_split1/wav_ov1_split1_30db/';
label_folder_path = 'ov1_split1/desc_ov1_split1/';
mel_folder_path = 'output_spectrogram';

% window specs
window_length = 512; 
win = hamming(window_length);
overlap_length = 256;

%% create TRAIN SET
fprintf("\nTRAIN SET pre-processing:\n\n");
train_set_size = 3; % CHANGE THIS PARAMETER
file_name = string(zeros(train_set_size,1));
track_path = string(zeros(train_set_size,1));
S = cell(1, train_set_size);
for i = 1:train_set_size
    file_name(i) = "train_" + string(i-1) + "_desc_30_100";
    track_path(i) = audio_folder_path + file_name(i) + '.wav';
    track_info(i) = audioinfo(track_path(i));
    fprintf("--  "+file_name(i)+" analysis... ");
    [audio_in, fs] = audioread(track_path(i));
    TRAIN_SET{i} = mfcc_analysis(track_path(i));
                                
	save_mel(audio_in, fs, i-1, window_length, overlap_length, mel_folder_path)
    fprintf("Done.\n");
end
TRAIN_SET = reshape_input(TRAIN_SET)
mfcc_plot(TRAIN_SET{1});

aFE = audioFeatureExtractor("SampleRate", fs, ...
    "Window", hamming(window_length), ...
    "OverlapLength", overlap_length, ...
    "FFTLength", 1024, ...
    "SpectralDescriptorInput", "melSpectrum", ...
    "spectralCentroid", true, ...
    "spectralSlope", true)

features = extract(aFE, TRAIN_SET{1}(:,:,1));
[numHopsPerSequence,numFeatures,numSignals] = size(features)


%% Funtions

% MFCC
function coeffs = mfcc_analysis(track_path)

[audio_in, freq] = audioread(track_path);

window_size = 512; 
win = hamming(window_size);
overlap_perc = 50;	% overlap percentage
overlap_samples = window_size*(overlap_perc/100);   % number of overlapping samples
    
S = stft(audio_in,"Window", win,"OverlapLength",overlap_samples,"Centered",false);
coeffs = mfcc(S, freq, "LogEnergy", "Ignore");
end

% make all input of the same (minimum) length
function NEW_SET = reshape_input(SET)
    num_tracks = size(SET,2)                % number of tracks
    num_samples = zeros(1,num_tracks);
    for i = 1:size(SET,2)
        num_samples(i) = size(SET{i},1);	% number of samples per track
    end
    max_samples = max(num_samples);
    
    NEW_SET = cell(1,num_tracks);
    for i = 1:num_tracks
        num_samples = size(SET{i},1);
        new_cols = max_samples - num_samples;
        for ch = 1:4
            NEW_SET{i}(:,:,ch) = [SET{i}(:,:,ch); zeros(new_cols,13)];
        end
    end
end

% save mel image
function save_mel(audio_in, fs, track_num, window_length, overlap_length, mel_folder_path)       
	for ch = 1:4       
        figure()
        melSpectrogram(  audio_in(:,ch), fs, ...
                                        'WindowLength', window_length,...
                                        'OverlapLength', overlap_length, ...
                                        'FFTLength',1024, ...
                                        'NumBands',13, ...
                                        'FrequencyRange',[62.5,20e3]);
        saveas(gcf, mel_folder_path + "train_" + track_num + "_desc_30_100_ch"+ch+".png");
    end
    close all
end

function mfcc_plot(coeffs)
    figure('Position', [30 300 1000 600], 'PaperPositionMode', 'auto', ... 
                 'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
	imagesc( [0,size(coeffs,1)], [0,size(coeffs,2)-1], coeffs(:,:,1).' ); 
	axis('xy');
    colormap(jet)
    colorbar()
	xlabel('Frame index');
	ylabel('Cepstrum index');
	title('Mel frequency cepstrum');
end


function NEW_SET = get_channel(SET, ch)
    num_tracks = size(SET,2);
    NEW_SET = cell(1,num_tracks);
    for i = 1:num_tracks
        NEW_SET{i} = SET{i}(:,:,ch);
    end
end







%
