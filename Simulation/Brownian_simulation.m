% This function simulate Brownian motion with given time and diffusion
% coefficient. 
% This is modified from the Jagriti/Nikita's code.

function [trj] = Brownian_simulation(t, D, dim)

N = length(t);	% N - Number of positions, N-1 - number of displacements 				%
dt = diff(t);	% Time differential between positions			%
	
trj = zeros([N, 3+1]);	% Initialize the trajectory %
trj(:,end) = t;

% Normal distribution
mu = 0;
sigma = (2*D*dt).^0.5;
du_x = mu + sigma .* randn(N-1, 1);
du_y = mu + sigma .* randn(N-1, 1);
if dim == 3
du_z = mu + sigma .* randn(N-1, 1);
elseif dim == 2
du_z = zeros([N-1],1);
end
du_3d = [du_x du_y du_z];
trj(2:end,1:3) = cumsum(du_3d);

end


