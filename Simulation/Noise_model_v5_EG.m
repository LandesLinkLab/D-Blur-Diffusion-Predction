% # Program Name: Noise model for 3D SMT Simulations
% # Coders : Emil Gillett
% # Last Modified: May 4th, 2024; Time: 4:43 pm.
function [image_photoelectrons_output,image_photoelectrons, photoelectron_noise] = Noise_model_v5_EG(Itot,signal)
%_____________________________________________________
%Camera specs (Photometrics Prime 95b) (https://www.photometrics.com/products/prime-family/prime95b#resource-downloads)

QE = 0.95;                                                                  % Quantum Efficiency
sensitivity = 1.2;                                                          % e-/ADU (electrons per analog-to-digital unit)
sensitivity = (1/sensitivity);                                                % ADU/e-
read_noise = 1.6;                                                           % e- (median electrons)
bit_depth = 16;                                                             % "Combining the two measurements into a single 16-bit value..." 
bias_offset = 100;                                                          % Built-in baseline ADU, prevents #ADUs from being negative at low signal
%______________________________________________________
%Photon shot noise.(random incident photons) Variance of noise =  #photons for Poisson process (non-deterministic, random)
%Dark noise = readout noise (value given in spec sheet) and dark current (not too important for microscopy)

signal_photons = signal;
image_photons = rescale(Itot).*signal_photons;  
image_photoelectrons_output = rescale(Itot).*signal_photons.*QE; 
% Intensity scaled image
shot_noise = poissrnd(image_photons);                                         % shot noise
electrons = round(QE*shot_noise);                                           % photoelectrons for noise
image_photoelectrons = QE.*shot_noise;
%______________________________________________________
% Generate read noise (normal distribution), where read noise is the std. dev... Approach taken in SMLM challenge
read_noise_array = normrnd(image_photoelectrons,read_noise);
electrons_out = round(read_noise_array + electrons);                        % Add read noise to photoelectrons
max_adu = round(2.^(bit_depth)-1);                                          % Ensure discrete ADU count and set maximum value
adu = round(electrons_out.*sensitivity);                                    % e- to ADU conversion
adu(adu>max_adu) = max_adu;                                                 % Limit ADU values to the max
adu = (adu + bias_offset);                                                  % add the baseline that the spec sheet says
photoelectron_noise = round(adu/sensitivity);


end
