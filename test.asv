%% clear everything
close all
clear
clc

%% read input data
% images = {'images/lena.bmp', 'images/peppers.bmp', 'images/boat.bmp'};
images = {'images/lena.bmp'};
cI = readImages(images);
cI = cI./255;

% sparsity level
k0 = 4;

% number of transformstions
m = 32;
[Ub, Xb, positionsb, valuesb, tusb, errb] = b_dla(cI, k0, m, 8);

% number of stages
stages = 8;
[Um, Xm, Sm, positionsm, valuesm, tusm, errm] = m_dla(cI, k0, stages);
[Umg, Xmg, Smg, positionsmg, valuesmg, tusmg, errmg] = m_dla_greedy(cI, k0, stages);

% precision
bits = 1;
[Us, D, Xs, positionss, valuess, tuss, errs] = s_dla(cI, k0, m, bits);

% this will take a while, uncomment, run and then be patient
% [U, X, positions, values, tus, err] = bg4x4_dla(cI, k0, m);
