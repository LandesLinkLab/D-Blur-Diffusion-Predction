% Generates `numLoops` sets of P Brownian‐motion trajectories,
% each with a random diffusion coefficient in D_range.
% Saves each trajectory set and its labels to a timestamped subfolder.
%% --- GLOBAL DECLARATIONS --- %%
close all
clear all

%% -- Set Parameter and Path -- %%
path = ''; % Add datapath here.

%% Create a folder with the current date and time 

% Get the current date and time
currentDateTime = datetime('now');

% Set the format to 'yyyy-MM-dd_HH-mm'
formattedDateTime = datestr(currentDateTime, 'yyyy-mm-dd');

% Set the format for the subfolder to 'HH-mm'
formattedTime = datestr(currentDateTime, 'HH-MM');

savepath = [path , formattedDateTime , '\'];

% Create the directory if it doesn't exist
if ~exist(savepath, 'dir')
    mkdir(savepath);
end

% Define the subfolder path
subfolderPath = [savepath , formattedTime , '\'];

% Create the subfolder directory if it doesn't exist
if ~exist(subfolderPath, 'dir')
    mkdir(subfolderPath);
end


%% Set the parameters. 


% Set the seed for reproducibility
seed = 54321;
rng(seed); 

% Set a loop number 
numLoops = 5000;

% Simulation Parameters %
T = 90;	% <#> Number of frames to simulate particles for	%
P = 10;	% <#> Number of particles to simulate				%
dt = 0.001;  % <s> Exposure time per frame. %
D_range = [0.01 1];   % <um^2/s> diffusion coefficient range
dim = 2;  % The dimemsion for simulation 

%% --- INITIALIZE --- %%
Time = linspace(0, (T-1) * dt, T)'; % The time range. 

% Generate the list of random values within the specified range, total
% number of P. 
D_values_all = D_range(1) + (D_range(2) - D_range(1)) * rand(numLoops, P);

%% --- SIMULATE MOTION --- %%
for loopIdx = 1:numLoops
    %% --- INITIALIZE --- %%
    Traj = zeros([T,4,P]); % [x,y,z,t] Position vector through time
    Lbls = zeros([P,1]); % Initialize Lbls for this iteration
    %% --- SIMULATE MOTION --- %%
    % Access the diffusion coefficients for this loop
    D_values = D_values_all(loopIdx, :);
    
    for i = 1:P
        trj = Brownian_simulation(Time, D_values(i), dim);
        Traj(:,:,i) = trj;
        Lbls(i) = D_values(i);
    end
    
    %% -- Save the Simulated Data -- %
    save(fullfile(subfolderPath, ['Trajs_loop_' num2str(loopIdx) '.mat']), 'Traj');
    save(fullfile(subfolderPath, ['Labels_loop_' num2str(loopIdx) '.mat']), 'Lbls');

end