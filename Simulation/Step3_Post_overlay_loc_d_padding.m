%% processBlurredImageAndLabels.m
% Process PSF movies: crop into patches, generate label maps, and save
% outputs.

%% Setup
close all 
clear all

path = '';  % Set the path to where the movies are. 
cd(path)

savepath_img = [path, 'imgpadding\'];
% Create the directory if it doesn't exist
if ~exist(savepath_img, 'dir')
    mkdir(savepath_img);
end

savepath_labelD = [path, 'labelpading\'];
savepath_pairD = [path, 'pairpading\'];
% Create the directory if it doesn't exist
if ~exist("savepath_labelD", 'dir')
    mkdir(savepath_labelD);
end

if ~exist("savepath_pairD", 'dir')
     mkdir(savepath_pairD)
end


%%  Parameters
numLoops = 5000;
frame_time = 0.030;                 % <s> Expousure time for each frame
frame_time_ori = 0.001;             % <s> The original expousure time for each frame. 

psf_sigma = 1;                      % Spatial blur sigma
d_sigma = 0.5;                      % Diffusion label sigma
D_slices = 10;                      % Number of D slices

pad_psf = 0;                        % Padding for x and y
pad_d = 2;                          % Padding for D

D_range = [0.01 2];                 % Diffusion coefficient range


%% Main Loop 
for loopIdx = 1:numLoops
    load(['movie_matrix_' num2str(loopIdx) '.mat'])
    load(['Traj_movie_' num2str(loopIdx) '.mat'])
    
    [height, width, num_frames] = size(movie_matrix); % Get image dimensions
    
        
    % Initiate the variables 
    label_locD = zeros(size(movie_matrix, 1) + pad_psf*2,size(movie_matrix, 2) + pad_psf*2,D_slices + pad_d*2);  % The extra layer for edge purposes 
    loc_pair = zeros(size(traj_movie,3), 3);
    image_size = size(movie_matrix, 1);
    overlay = round(frame_time / frame_time_ori) ;
    stacksize = num_frames /overlay;
    
    % Process the image for blurry effect 
    img_blur = zeros([height, width,stacksize]);

    for f = 1:stacksize
        img = sum(movie_matrix(:,:,(f-1)*overlay+1:f*overlay),3);
        img_blur(:,:,f) = img;  
    end
    % Generate labels and pairs
    for emitter = 1:size(traj_movie,3)
        loc = traj_movie(:, 1:2, emitter);
        loc_center = mean(loc,1);

        D = traj_movie(1,3,emitter);
        D_slice = ((D - D_range(1)) ./ (D_range(2) - D_range(1)) ) * (D_slices - 1) + 1; 
        % Matlab starts counting from 1 not 0. Adding the additional 1 to
        % fix the issue with offset. 
        x_pixel = loc_center(1);
        y_pixel = loc_center(2);
        loc_pair(emitter, :) = [x_pixel , y_pixel, D];
        % switch the x and y label since there's a wild issue 
        prob_map = gaussian_probability_3D(y_pixel,x_pixel,D_slice, ...
                     height, width, D_slices, psf_sigma, d_sigma, pad_psf, pad_d);
        label_locD = label_locD + prob_map;    % Add this emitter to the label
       
    end

    % Normalize the new image
    min_blur = (img_blur - min(min(img_blur)));
    img_blur = min_blur/max(max(max(min_blur)));
    
    % Save the results
    filename_img = fullfile(savepath_img, ['img_' num2str(loopIdx) '.mat']);
    filename_loc = fullfile(savepath_labelD,['img_' num2str(loopIdx) '_loc' '.mat']);
    filename_loc_pair = fullfile(savepath_pairD, ['img_' num2str(loopIdx) '_pair' '.mat']);

    save(filename_img, 'img_blur')
    save(filename_loc, 'label_locD')
    save(filename_loc_pair, 'loc_pair')
end


%% Function for probability calculation

function prob_map = gaussian_probability_3D(x,y,D,size_x,size_y, size_D, psf_sigma, d_sigma, pad_psf, pad_d)
    % Generate a probability map for x, y location and their D. 
    % 
    [i, j, k] = ndgrid(1-pad_psf: size_x+pad_psf , 1-pad_psf:size_y+pad_psf, 1-pad_d:size_D+pad_d);
    prob_map = exp(-(((i - x).^2) / (2 * psf_sigma^2) + ...
                     ((j - y).^2) / (2 * psf_sigma^2) + ...
                     ((k - D).^2) / (2 * d_sigma^2)));

    % Now normalize this: 
    prob_map = prob_map ./max(prob_map(:));
end
