function matlab_data_processor()
% MATLAB script to process EEG data and save for Python viewer
% This script applies the exact same processing as openephysreaderjkk.m
% and saves the data in a simple format for the Python interface

clear all;
clc;

fprintf('=== MATLAB EEG Data Processor ===\n');
fprintf('Processing data for Python EEG viewer...\n\n');

% Get current directory and set up paths
currentDir = pwd;
fprintf('Current directory: %s\n', currentDir);

% Add paths for MATLAB functions
addpath(fullfile(currentDir, '..', 'Matlab_EEG_Analysis', 'matlabscripts'));
addpath(fullfile(currentDir, '..', 'Matlab_EEG_Analysis', 'matlabscripts', 'OpenEphys'));

% Define data path
dataPath = fullfile(currentDir, '..', 'mouse_spikewave-matlab-style', 'data', 'SW1245_P30SS', 'SW123_1245.57.61_2025-03-17_09-43-42');

% Check if data exists
if ~exist(dataPath, 'dir')
    error('Data directory not found: %s', dataPath);
end

fprintf('Data path: %s\n', dataPath);

% Create output directory
outputDir = fullfile(currentDir, 'processed_data');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
fprintf('Output directory: %s\n', outputDir);

% Processing parameters (exact same as openephysreaderjkk.m)
fs = 250; % Target sample rate
lowpassfilter = 25; % 25 Hz lowpass
highpassfilter = 1; % 1 Hz highpass
filtering = 1;

fprintf('\nProcessing parameters:\n');
fprintf('Target sample rate: %d Hz\n', fs);
fprintf('Lowpass filter: %d Hz\n', lowpassfilter);
fprintf('Highpass filter: %d Hz\n', highpassfilter);

% Load data using OpenEphysLoader
fprintf('\nLoading data...\n');
try
    [fileData, fileHeader] = OpenEphysLoader(dataPath, 'desiredfs', fs);
    fprintf('Data loaded successfully!\n');
    fprintf('Data shape: %s\n', mat2str(size(fileData)));
    fprintf('Sample rate: %d Hz\n', fileHeader.fs);
    fprintf('Channels: %s\n', mat2str(fileHeader.chans));
catch ME
    error('Error loading data: %s', ME.message);
end

% Apply MATLAB processing pipeline
fprintf('\nApplying processing pipeline...\n');

% Step 1: Convert to mV
sst = fileData / 1000;
fprintf('Step 1 - After /1000 conversion:\n');
fprintf('  Range: %.6f to %.6f mV\n', min(sst(:)), max(sst(:)));

% Step 2: Apply filtering
if filtering > 0
    si = 1000/fs; % ms
    
    if lowpassfilter > 0
        fprintf('Step 2a - Applying lowpass filter at %d Hz...\n', lowpassfilter);
        [blpf, alpf] = butter(2, lowpassfilter*2*si/1000);
        sst = filter(blpf, alpf, sst);
    end
    
    if highpassfilter > 0
        fprintf('Step 2b - Applying highpass filter at %d Hz...\n', highpassfilter);
        [bhpf, ahpf] = butter(2, highpassfilter*2*si/1000, 'high');
        sst = filter(bhpf, ahpf, sst);
    end
    
    fprintf('Step 2 - After filtering:\n');
    fprintf('  Range: %.6f to %.6f mV\n', min(sst(:)), max(sst(:)));
end

% Step 3: Create time array
time = 0:length(sst)-1;
time = time * si; % ms
times = time / 1000.0; % seconds

fprintf('\nTime array created:\n');
fprintf('  Duration: %.2f seconds (%.2f minutes)\n', times(end), times(end)/60);

% Save processed data
fprintf('\nSaving processed data...\n');

% Save as simple CSV for Python
csv_file = fullfile(outputDir, 'eeg_data.csv');
csv_data = [times', sst];
csv_headers = {'Time_s', 'Channel_57_mV', 'Channel_61_mV'};
csv_table = array2table(csv_data, 'VariableNames', csv_headers);
writetable(csv_table, csv_file);
fprintf('✓ Saved as CSV: %s\n', csv_file);

% Save metadata
metadata_file = fullfile(outputDir, 'metadata.txt');
fid = fopen(metadata_file, 'w');
fprintf(fid, 'EEG Data Metadata\n');
fprintf(fid, '================\n\n');
fprintf(fid, 'Original Data Path: %s\n', dataPath);
fprintf(fid, 'Processing Date: %s\n', datestr(now));
fprintf(fid, 'Sample Rate: %d Hz\n', fs);
fprintf(fid, 'Duration: %.2f seconds (%.2f minutes)\n', times(end), times(end)/60);
fprintf(fid, 'Channels: %s\n', mat2str(fileHeader.chans));
fprintf(fid, 'Data Shape: %s\n', mat2str(size(sst)));
fprintf(fid, '\nProcessing Parameters:\n');
fprintf(fid, 'Lowpass Filter: %d Hz\n', lowpassfilter);
fprintf(fid, 'Highpass Filter: %d Hz\n', highpassfilter);
fprintf(fid, '\nData Statistics:\n');
fprintf(fid, 'Channel 57 - Min: %.6f, Max: %.6f, Mean: %.6f, Std: %.6f mV\n', ...
    min(sst(:,1)), max(sst(:,1)), mean(sst(:,1)), std(sst(:,1)));
fprintf(fid, 'Channel 61 - Min: %.6f, Max: %.6f, Mean: %.6f, Std: %.6f mV\n', ...
    min(sst(:,2)), max(sst(:,2)), mean(sst(:,2)), std(sst(:,2)));
fclose(fid);
fprintf('✓ Saved metadata: %s\n', metadata_file);

fprintf('\n=== Processing Complete ===\n');
fprintf('Data saved in: %s\n', outputDir);
fprintf('Files created:\n');
fprintf('  - eeg_data.csv (main data file)\n');
fprintf('  - metadata.txt (processing info)\n');
fprintf('\nYou can now run the Python EEG viewer!\n');

end 