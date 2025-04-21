% Generate a map with different Ds at different regions. For Dmap
% illustration.

%% Parameters 

num_particles = 100;
num_frames = 630;
dt = 0.001;
num_loops = 500;

%% Save path 

path = '';  % Set this path to where you wish to save the output. 

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

%% 
xmin = -12.5; % <um>
xmax = 12.5; % <um>
ymin = -12.5; % <um>
ymax = 12.5; % <um>

seed = 12345;
rng(seed)

map_size = [xmax-xmin, ymax-ymin]; % <um> 
%pix_size = 0.05;
pix_size = 0.10;
map_size_pix = map_size ./ pix_size;

x_pix = 1:map_size_pix(1);
y_pix = 1:map_size_pix(2);
xy_pix = combvec(x_pix, y_pix)';



D_bckg = 0.8; % <um^2/s>
map = D_bckg*ones(map_size_pix);

D_cir1 = 0.1; % <um^2/s>
dim_cir1 = 10; % <um>
xy0_cir1 = [0, 0]; % <um>
xy0_cir1_pix = round((xy0_cir1 - [xmin,ymin])./pix_size);
xy0_cir1_pix(xy0_cir1_pix == 0) = 1;
rad_cir1_pix = round(dim_cir1./pix_size./2);
coord_cir1 = xy_pix((xy_pix(:,1) - xy0_cir1_pix(1)).^2 + (xy_pix(:,2) - xy0_cir1_pix(2)).^2 <= rad_cir1_pix.^2,:);
map(sub2ind(size(map), coord_cir1(:,2), coord_cir1(:,1))) = D_cir1;


D_cir2 = 1; % <um^2/s>
dim_cir2 = 5; % <um>
xy0_cir2 = [1, 10]; % <um>
xy0_cir2_pix = round((xy0_cir2 - [xmin,ymin])./pix_size);
xy0_cir2_pix(xy0_cir2_pix == 0) = 1;
rad_cir2_pix = round(dim_cir2./pix_size./2);
coord_cir2 = xy_pix((xy_pix(:,1) - xy0_cir2_pix(1)).^2 + (xy_pix(:,2) - xy0_cir2_pix(2)).^2 <= rad_cir2_pix.^2,:);
map(sub2ind(size(map), coord_cir2(:,2), coord_cir2(:,1))) = D_cir2;


D_sq1 = 0.4; % <um^2/s>
len_sq1 = 4; % <um>
xy0_sq1 = [-5, -8]; % <um>
xy0_sq1_pix = round((xy0_sq1 - [xmin,ymin])./pix_size);
xy0_sq1_pix(xy0_sq1_pix == 0) = 1;
len_half_sq1_pix = round(len_sq1./pix_size./2);
coord_sq1 = xy_pix(xy_pix(:,1) >= xy0_sq1_pix(1) - len_half_sq1_pix & xy_pix(:,1) <= xy0_sq1_pix(1) + len_half_sq1_pix ...
    & xy_pix(:,2) >= xy0_sq1_pix(2) - len_half_sq1_pix & xy_pix(:,2) <= xy0_sq1_pix(2) + len_half_sq1_pix,:);
map(sub2ind(size(map), coord_sq1(:,2), coord_sq1(:,1))) = D_sq1;


D_sq2 = 0.2; % <um^2/s> 
len_sq2 = 6; % <um>
xy0_sq2 = [8, -5]; % <um>
xy0_sq2_pix = round((xy0_sq2 - [xmin,ymin])./pix_size);
xy0_sq2_pix(xy0_sq2_pix == 0) = 1;
len_half_sq2_pix = round(len_sq2./pix_size./2);
coord_sq2 = xy_pix(xy_pix(:,1) >= xy0_sq2_pix(1) - len_half_sq2_pix & xy_pix(:,1) <= xy0_sq2_pix(1) + len_half_sq2_pix ...
    & xy_pix(:,2) >= xy0_sq2_pix(2) - len_half_sq2_pix & xy_pix(:,2) <= xy0_sq2_pix(2) + len_half_sq2_pix,:);
map(sub2ind(size(map), coord_sq2(:,2), coord_sq2(:,1))) = D_sq2;


% Create a heatmap
figure();
h = heatmap(map);  % Store the HeatmapChart object

% Customize appearance
h.GridVisible = 'off';        % Turn off gridlines
h.XDisplayLabels = repmat({''}, 1, size(map, 2));  % Empty labels for x-axis
h.YDisplayLabels = repmat({''}, size(map, 1), 1);  % Empty labels for y-axis
h.FontSize = 16;                                % Set font size to 12

% Set the colorbar label (built-in with heatmap)
h.ColorbarVisible = 'on';                       % Ensure the colorbar is visible
h.Colormap = jet;                               % Set the colormap
h.ColorLimits = [min(map(:)), max(map(:))];     % Optional: Set color scale limits
%h.Title = 'Diffusion Coefficient (\mum^2/s)';              % Add a label to the colorbar
%% Generate particles 

x0 = rand(num_particles,num_loops).*(xmax - xmin) + xmin; %<um>
y0 = rand(num_particles,num_loops).*(ymax - ymin) + ymin; %<um>
z0 = rand(num_particles,num_loops).*0.5; %<um>


%% Get a loop out of this 
for loop_Idx = (1:num_loops)
XYZ0 = [x0(:,loop_Idx), y0(:,loop_Idx), z0(:,loop_Idx)];

trj = NaN(num_frames, 5, num_particles); 
time  = 0;

trj(1,:,:) = [XYZ0, time*ones(size(XYZ0,1),1), zeros(size(XYZ0,1),1)]';

% Assign the D manually: 
XY_pix = round((XYZ0(:,1:2) - [xmin,ymin])./pix_size);
XY_pix_out = XY_pix(:,1) < 1 | XY_pix(:,1) > size(map,2) | XY_pix(:,2) < 1 | XY_pix(:,2) > size(map,1);
XY_pix(XY_pix_out, 1) = 1;
XY_pix(XY_pix_out, 2) = 1;
XY_pix(XY_pix == 0) = 1;
Dv0 = map(sub2ind(size(map), XY_pix(:,2), XY_pix(:,1)));
trj(1,5,:) = Dv0';
XYZ = XYZ0;
for i = 2:num_frames
    XY_pix = round((XYZ(:,1:2) - [xmin,ymin])./pix_size);
    XY_pix_out = XY_pix(:,1) < 1 | XY_pix(:,1) > size(map,2) | XY_pix(:,2) < 1 | XY_pix(:,2) > size(map,1);
    XY_pix(XY_pix_out, 1) = 1;
    XY_pix(XY_pix_out, 2) = 1;
    XY_pix(XY_pix == 0) = 1;
    Dv = map(sub2ind(size(map), XY_pix(:,2), XY_pix(:,1)));
    time = time + dt;
    sigma = (2*Dv.*dt).^0.5;
    du_x = sigma .* randn(num_particles, 1);
    du_y = sigma .* randn(num_particles, 1);
    du_z = sigma .* randn(num_particles, 1);
    XYZ = XYZ + [du_x du_y du_z];
    XYZ_trj = XYZ;
    % XYZ_trj(XYZ_trj(:,1) < xmin | XYZ_trj(:,1) > xmax | XYZ_trj(:,2) < ymin | XYZ_trj(:,2) > ymax,:) = [0 0 0];

    out_of_bounds = XYZ_trj(:,1) < xmin | XYZ_trj(:,1) > xmax | ...
                XYZ_trj(:,2) < ymin | XYZ_trj(:,2) > ymax;
    if any(out_of_bounds)
        XYZ_trj(out_of_bounds, :) = repmat([nan, nan, nan], sum(out_of_bounds), 1);
    end
    trj(i,:,:) = [XYZ_trj, time*ones(size(XYZ,1),1), Dv]';
    %trj(i,:,:) = [XYZ_trj, time*ones(size(XYZ,1),1)]';  % Take only 4 dimensions 

end
%% Save
save(fullfile(subfolderPath, ['Trajs_loop_' num2str(loop_Idx) '.mat']), 'trj');


end




function [traj] = NormalDiffusion_3D(t, D)
N = length(t);	% Number of positions to produce				%
dt = diff(t);	% Time differential between positions			%
	
traj = zeros([N, 3+1]);	% Initialize the trajectory %
traj(:,end) = t;

% Normal distribution
mu = 0;
sigma = (2*D*mean(dt)).^0.5;
du_x = mu + sigma * randn(N, 1);
du_y = mu + sigma * randn(N, 1);
du_z = mu + sigma * randn(N, 1);
du_3d = [du_x du_y du_z];
traj(:,1:3) = cumsum(du_3d);
end