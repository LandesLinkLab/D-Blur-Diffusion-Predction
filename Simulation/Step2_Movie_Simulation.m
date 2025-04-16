% Generate PSF movies from Brownian trajectories with random diffusion coefficients

%% Setup
clearvars; close all;
clear all;

path = '';  % Set the path to where the trajectory files are.
cd(path)


numLoops = 1;   % Set that same as the number of trajectory files you'd like to use.


%% Start simulation 

for loopIdx = 1: numLoops
    tic
    trajFilename = ['Trajs_loop_' num2str(loopIdx) '.mat'];
    labelFilename = ['Labels_loop_' num2str(loopIdx) '.mat'];
    load(trajFilename);
    load(labelFilename);
       
    % Know the trajectories
    N_trj = size(Traj,3);
    traj_movie = zeros(size(Traj)); 

    % Assign the parameters
    num_emitters = size(Traj,3);
    num_frames = size(Traj,1);
    
    %__________________________________________________________________________
    %Define Optical Parameters: User input
    lambda = 585e-9;                                                             % wavelength (m) 585e-9
    NA = 1.46;                                                                   % Objective Numerical Aperture 1.46, 0.2,0.46
    n_sub_0 = 1;                                                                 % Refractive index 
    M = 160;                                                                     % Total magnification
    Z_0 = 377;                                                                   % impedance of free space (Ohms).
    Z = Z_0/n_sub_0;                                                             % characteristic impedance of a dielectric with index of refraction n
    
    %4f System
    f_focal_length = 110;                                                        % 4f system lens focal length (mm)
    
    signal = 50;                                                                % Photons 219   
    power = 1e-13;                                                               % Watts.  No need to change. Was used to Check Perseval's theorem
    Phase_mask_type= 0;                                                          % 0 == no phase mask. 1 == DHPSF. 2 == RICE. 3 == Rod. 4==Experimental.
    pix_size = 16*1e-6;                                                          % (m)Camera pixel size 11*1e-6
    num_pixels= 512;                                                              % Number of camera pixels in array (even number) (500)
    image_size = 64;                                                            % output image size containing psf dynamics (250) (Previously was 300)
    
    exp_time = 0.001;                                                            % Exposure time (s)
    add_noise = 2;                                                               % 0 == no noise, 1 == Just shot noise, 2 == shot noise + dark noise
    random_start_pos = 1;                                                        % 1 == YES, random starting positions for all emitters. 0 == NO, emitters start at origin
    a_z = -2;                                                                    % Minimum axial position (um)
    b_z = 2;                                                                     % Maximum axial position (um)\

    %_________________ _________________________________________________________
    
    % Initiate a label matrix: 
    
    label_matrix = zeros([2,image_size,image_size,num_frames]);
    
    % define image coordinates
    
    dx_true = (pix_size./M);                                                                                % image plane sampling
    scaled_dx = n_sub_0*dx_true;                                                                            % due to Abbe sine condition, scale by imaging medium r.i. (see appendix of my journal club)
    x_image_plane_coord = linspace(-scaled_dx.*num_pixels/2,scaled_dx.*num_pixels/2,num_pixels);            % Step size inferred
    
    %Sampling for the cropped image later on
    cropped_sampling= linspace(-scaled_dx.*image_size/2,scaled_dx.*image_size/2,image_size);                % Step size inferred
    
    %__________________________________________________________________________
    %Randomize emitter trajectories starting position (add a lateral shift to all datapoints in a trajectory)
    % Add the preselected shift value from the randomizing selection to trajetory data
    
    adjusted_trajectories = zeros(size(Traj,1),size(Traj,2),size(Traj,3));
    
    for emitter_index = 1:num_emitters
    
        %rng(0,'twister');                                                      % Initialize the random number generator to make the results in this example repeatable.
        a_x = cropped_sampling(1).*1e6;                                        % this is currently the size of the resulting movie frame
        b_x = cropped_sampling(end).*1e6;
        r_x = (b_x-a_x).*rand(image_size,1) + a_x;
        random_index_x = randi(numel(cropped_sampling), 1);
        mu_x = r_x(random_index_x);                                             % (um)
    
        if random_start_pos == 0
            mu_x = 0;
        end
    
        a_y = cropped_sampling(1).*1e6;
        b_y = cropped_sampling(end).*1e6;
        r_y = (b_y-a_y).*rand(image_size,1) + a_y;
        random_index_y = randi(numel(cropped_sampling), 1);
        mu_y = r_y(random_index_y); %(um)
    
        if random_start_pos == 0
            mu_y = 0;
        end
    
        mu_z = 0;   % Change z to the same plane
      
        adjusted_trajectories (:,1,emitter_index)= -1*Traj (:,1,emitter_index) + mu_x; %have to make the initial trajectory value negative for correct representation with imagesc later
        adjusted_trajectories (:,2,emitter_index)= Traj (:,2,emitter_index) + mu_y;
        adjusted_trajectories (:,3,emitter_index)= Traj (:,3,emitter_index) + mu_z;
        adjusted_trajectories (:,4,emitter_index)= Traj (:,4,emitter_index);
    end
    %__________________________________________________________________________
    % define pupil coordinates
    dx = x_image_plane_coord(2)-x_image_plane_coord(1);                         % Sampling period (m)
    fS = 1/dx;                                                                  % Spatial sampling frequency (m^-1 or inverse meters)
    fS = fS * 1.518;                                                            % Scale it by the refractive index. Not sure why/if there're scientific fundations.
    df = fS/num_pixels;                                                         % Spacing b/w discrete frequency coords (m^-1)
    fx = linspace(-fS/2,fS/2,size(x_image_plane_coord,2));                      % Spatial frequency(m^-1)
    %__________________________________________________________________________
    %Explicit pupil generation
    
    fNA = NA/ lambda;                                                           % pupil radius (m^-1)
    pupil_radius_pixels = fNA/df;                                               % pupil radius pixels
    pupil_diameter_pixels = pupil_radius_pixels*2;
    pupil_center = num_pixels/2;                                                % Assume even number when finding the center
    u = (1:num_pixels);
    v=u;
    [U,V] = meshgrid(u,v);
    
    cir=zeros(size(u,2),size(u,2));
    for U=1:1:size(u,2)
        for V=1:1:size(u,2)
            radius_checker=sqrt((U-pupil_center)^2+(V-pupil_center)^2);
            if radius_checker<pupil_radius_pixels
                cir(U,V)=1;
            else
                cir(U,V)=0;
            end
        end
    end
    
    Norm_integral_precalculation = (df.*simps(df.*simps(((abs(cir)).^2))))/Z;   % 2D integration of aperture using Simpson's rule (Obtain Normalization constant)
    Norm_integral = Norm_integral_precalculation;
    cir = cir.* (1 + 1i*0);
    normalized_cir = cir.*(power/Norm_integral).^(0.5);
    new_power = (df.*simps(df.*simps(((abs(normalized_cir)).^2))))/Z;           % Check if power input = power output. (Perseval's theorem)
    
    %__________________________________________________________________________
    %Create clear Aperture if there is no phase mask
    
    if Phase_mask_type ==0
        pmask = ones(size(normalized_cir,1));                                   % phase would be exp(1i*pmask), and pmask would be zero.
        axial_phase_diameter = rndeven(pupil_diameter_pixels);                  % Round to the nearest even integer. If not, then there will be bugs resizing the phase masks later.
    end
    
    %__________________________________________________________________________
    % This section is for cropping, scaling and padding the DH phase pattern. The phase mask from
    % Wenxio's code is 180x180 for some reason. Phase mask size must match
    % the pupil
    
    if Phase_mask_type == 1
        load('DHPhaseValues.mat');                                              % Loading the phase mask (200x200)
        raw_mask = tempphasedatafile;
        pmaskX  = tempphasedatafile(10:189,10:189);                             % Removing a bunch of zeros that I found on the outside of the mask
        pmaskX = flip(pmaskX,2);                                                % Flipping it horizontally to reverse rotation (clockwise)
        pmaskX = imresize(pmaskX, pupil_diameter_pixels/size(pmaskX,1));        % Resizing the phase data in the mask to match the diameter of the %pupil
        axial_phase_diameter = size(pmaskX,1);                                  % Saving the size of the phase data for resizing the "phase delay" later
    
        %padding the mask data with zeros at the edges to match the zeros of the pupil
        while size(pmaskX,1)<size(normalized_cir,1)
            pmaskX= padarray(pmaskX,[1 1],0,'both');
        end
        pmask = exp(1i*pmaskX);
    end
    
    %__________________________________________________________________________
    % This section is for cropping, scaling and padding the RICE phase pattern. The phase mask from
    % Wenxio's code is 60 x 60 for some reason. Phase mask size must match
    % the pupil
    
    if Phase_mask_type == 2
        load('RICE_phase_mask.mat'); %Loading the phase mask (60x60)
    
        pmaskX = flip(retrieved_RICE_phasemask,1);                              % Flipping it vertically, otherwise the letters are upside down
        pmaskX = flip(pmaskX,2);                                                % Flipping it horizontally, otherwise the letters are right-to-left
        pmaskX = imresize(pmaskX, pupil_diameter_pixels/size(pmaskX,1));        % Resizing the phase data in the mask to match the diameter of the %pupil
        axial_phase_diameter = size(pmaskX,1);                                  % Saving the size of the phase data for resizing the "phase delay" later
    
        %padding the mask data with zeros at the edges to match the zeros of the pupil
        while size(pmaskX,1)<size(normalized_cir,1)
            pmaskX= padarray(pmaskX,[1 1],0,'both');
        end
        pmask = exp(1i*pmaskX);
    end
    
    % %________________________________________________________________________
    % This section is for cropping and scaling the Corkscrew phase pattern. The phase mask from
    % Wenxio's code is 199x199 for some reason.Phase mask size must match
    % the pupil
    
    if Phase_mask_type == 3                                                     % Loading the phase mask (200x200)
        load('rod_phase_mask.mat')
        pmaskX = rod_phase_values(1:199,1:199);                                 % Removing a bunch of zeros that I found on the outside of the mask
        figure
        imagesc(pmaskX)
        colormap bone
        colorbar
        pmaskX = imresize(pmaskX, pupil_diameter_pixels/size(pmaskX,1));        % Resizing the phase data in the mask to match the diameter of the %pupil
        axial_phase_diameter = size(pmaskX,1);                                  % Saving the size of the phase data for resizing the "phase delay" later
    
        %padding the mask data with zeros at the edges to match the zeros of the pupil
        while size(pmaskX,1)<size(normalized_cir,1)
            pmaskX= padarray(pmaskX,[1 1],0,'both');
        end
    
        pmask = exp(1i*pmaskX);
    end
    
    % %________________________________________________________________________
    % This section is for testing new masks.
    
    if Phase_mask_type == 4                                                     % Loading the phase mask (200x200)
        load('maskRec.mat')
        pmaskX = maskRec;                                 % Removing a bunch of zeros that I found on the outside of the mask
        figure
        imagesc(pmaskX)
        colormap bone
        colorbar
    
        pmaskX = maskRec(1178:2492,1178:2492);
        figure
        imagesc(pmaskX)
        colormap bone
        colorbar
        modulus = mod(pmaskX,2*pi);
        wrapped = modulus.*(modulus<=pi) + (modulus-2*pi).*(modulus>pi);
        pmaskX = wrapped;
    
        figure
        imagesc(pmaskX)
        colormap gray
        colorbar
    
        pmaskX = imresize(pmaskX, pupil_diameter_pixels/size(pmaskX,1));        % Resizing the phase data in the mask to match the diameter of the %pupil
        axial_phase_diameter = size(pmaskX,1);                                  % Saving the size of the phase data for resizing the "phase delay" later
    
    
    
        %padding the mask data with zeros at the edges to match the zeros of the pupil
        while size(pmaskX,1)<size(normalized_cir,1)
            pmaskX= padarray(pmaskX,[1 1],0,'both');
        end
        figure
        imagesc(pmaskX)
        colormap hot
        colorbar
    
        pmask = exp(1i*pmaskX);
    
    end
    
    %__________________________________________________________________________
    % Initialize an array to store frames
    
    frames = cell(1,num_frames);
    movie_matrix = zeros(image_size,image_size,num_frames);
    
    % wait_bar = waitbar(0, 'Starting');
    %__________________________________________________________________________
    % Initialize Phase and PSF intensity calculations
    
    for frame_index = 1:num_frames
        Itot = zeros(image_size);
    
        for emitter_index = 1:num_emitters
    
    
            x_SMs_phy = adjusted_trajectories(frame_index,1,emitter_index);                   % Physical Position of Emitter (m), these are given in microns in the file
            y_SMs_phy = adjusted_trajectories(frame_index,2,emitter_index);
            z_SMs_phy = adjusted_trajectories(frame_index,3,emitter_index).*10^-6;            % (m)
    
            x_SMs_phy = x_SMs_phy/(dx*1e6);                                                   % These are scaled to allow appropriate translation later.
            y_SMs_phy = y_SMs_phy/(dx*1e6);
                    
            
            x_loc = image_size - (image_size / 2 + x_SMs_phy);
            y_loc = image_size - (image_size / 2 + y_SMs_phy);
            % Convert displacements to pixel indices
            x_pixel = round(image_size - (image_size / 2 + x_SMs_phy)); % Round to nearest integer for pixel index
            y_pixel = round(image_size -(image_size / 2 + y_SMs_phy));
            
            % Save the pixel location
            traj_movie(frame_index,1:2,emitter_index) = [x_loc, y_loc]; 
            
            
            
            % Set the labels: 
            D_label = Lbls(emitter_index);
            PSF_range = 1;
            traj_movie(frame_index,3,emitter_index) = D_label;  % Save D_label for future access. 
    
            %__________________________________________________________________
            %phase delay resulting from z-position (from Wenxiao's code)
    
            f = f_focal_length;
            % making 'r' bigger reduces rotation rate, implies reduction in phase shift
            r = f.*NA/(sqrt(M^(2)-NA^(2)));                                                 % radius of E-field at BFP
    
            ratio=f/r;
            xx=linspace(-axial_phase_diameter/2,axial_phase_diameter/2,axial_phase_diameter);
            yy=xx;
    
            [x,y]=meshgrid(xx,yy);      % meshgrid of x and y cordinates from xx and yy
            h=sqrt(x.^2+y.^2);          % Distance from the origin to each point in the grid.
            f=ratio*max(xx);            % Focal Length (m)
            n=n_sub_0;                  % Refractive ingex of medium
    
    
            z0 = z_SMs_phy;
            zp=z0*(2*n-2*sqrt(n^2-NA^2))*ratio^2;                               % This line computes an intermediate distance zp based on z0 (BREAKS if NA>n)
    
            f=f/2;
            h=h/2;
    
            phase=sqrt((f+zp).^2+h.^2)-sqrt(f.^2+h.^2);                         % This formula computes the difference in optical path lengths between two points. It takes into account the focal length (f), intermediate distance (zp), and distance from the optical axis (h). The difference in optical path lengths is then converted to phase.
            phase=2*pi/lambda*phase;                                            % Finally, the phase values are scaled by (2*pi)/lambda, which is (2*pi) divided by the wavelength of light (lambda), to obtain the phase in radians.
            phase = exp(1i*phase);
    
            if size(phase,1)<size(normalized_cir,1)
                adlen = round((size(normalized_cir,1)-size(phase,1))./2,TieBreaker="fromzero"); 
                phase= padarray(phase,[adlen adlen],0,'both');                    % padding the phase data with zeros at the edges to match the zeros of the pupil, just like the phase mask
            end
    
            %_________________________________________________________________________
            %Create point-spread function
    
            % Convolve aperture with phase mask, and axial phase
             psf_a = fftshift(fft2(normalized_cir)).*df^2;
            img = (((abs(psf_a)).^2)/Z);                                        % The intensity is the square of the absolute value of the E-field
            img = rescale(img)*signal;
    
            I_nothing = zeros(image_size);
            x_cropping_correction = -(dx.*1e6)/2;                               % Applying x and y translation CORRECTION to the PSF within the cropped image. Necessary since assuming all matrices have even #rows and columns
            y_cropping_correction = -(dx.*1e6)/2;
    
            % Cropping the image to the given image_size
            output_image = img(size(img,1)/2-image_size/2+1:size(img,1)/2+image_size/2,size(img,1)/2-image_size/2+1:size(img,1)/2+image_size/2);
    
            % Applying x and y translation from trajectory data to the PSF within the cropped image
            I = I_nothing+imtranslate(output_image(:,:),[x_SMs_phy-x_cropping_correction, y_SMs_phy-y_cropping_correction],'bicubic');
            Itot = Itot + I;                                                    % Sum intensity value frame by frame (Tried to normalize?)

        end
    
    
    
        if add_noise == 0
            %Compute # photons, photoelectrons, and noise for the image
            [image_photoelectrons_output,image_photoelectrons, photoelectron_noise] = Noise_model_v5_EG(Itot,signal);
            Itot = image_photoelectrons_output;
        end
        if add_noise == 1
            %Compute # photons, photoelectrons, and noise for the image
            [image_photoelectrons_output,image_photoelectrons, photoelectron_noise] = Noise_model_v5_EG(Itot,signal);
            Itot = image_photoelectrons;
        end
    
        if add_noise == 2
            [image_photoelectrons_output,image_photoelectrons, photoelectron_noise] = Noise_model_v5_EG(Itot,signal);
            Itot = photoelectron_noise;
        end
        movie_matrix(:,:,frame_index) = fliplr(flipud(photoelectron_noise));
        clear Itot
    
    end
    

    
    %% -- save file as a random image
    
     save(['movie_matrix_' num2str(loopIdx) '.mat'],'movie_matrix')
     save(['Traj_movie_' num2str(loopIdx) '.mat'], 'traj_movie')
    toc
end


%% Functions 

delete(gcp('nocreate'));


function y = rndeven(x)
x = floor(x);
x(x <= 1) = 2;
y = mod(x,2)+x;
end






