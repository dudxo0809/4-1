function x = Homework2()    % Find X

t = 0.01:0.01:0.2;          % time
freq = 11:30;               % frequency
freq = freq.';              % Transpose
A = cos(2*pi*freq*t);       % 20 sinusoidal signals

b = importdata("output.mat");   % import data 'b'

disp((inv(A) * b).');           % Display data X
disp((slv(A,b).'));

x = slv(A,b);               % return data X
%x = inv(A) * b;