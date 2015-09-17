clear all;
close all;

% This script needs the MCRoomSim Matlab package to generate room 
% impulse response data

addpath '../'

% number of rooms
NR = 100;
 
fs = 96000
base_name = 'set_N20_'
%base_name = strcat(strcat(base_name, int2str(fs)), '_');
c = 343;

M = 5;
N = 20;

for nr = 50:NR
    data = {}
    % room dimensions
    w = 8; %x
    l = 6; %y
    h = 5; %z

    L = [w l h];

    sources = [];
    receivers = [];

    % define microphones
    margin = 1;
    r(:,1) = margin + (w-1-margin).*rand(M,1);
    r(:,2) = margin + (l-1-margin).*rand(M,1);
    r(:,3) = margin + (h-1-margin).*rand(M,1);
    receivers = [receivers; r];
    R = []
    for i = 1:M
        R = [R AddReceiver('Location',r(i,:), 'Fs', fs)]
    end

    % define sources
    margin = 1;
    s(:,1) = margin + (w-1-margin).*rand(N,1);
    s(:,2) = margin + (l-1-margin).*rand(N,1);
    s(:,3) = margin + (h-1-margin).*rand(N,1);
    sources = [sources; s];
    S = []
    for i = 1:N
        S = [S AddSource('Location',s(i,:), 'Fs', fs)];
    end

    Room = SetupRoom('Dim', [w,l,h]);
    Options = MCRoomSimOptions('Fs', fs, 'SoundSpeed', c, 'MinImgSrc', -200, 'AutoCrop', false, 'SimDiff',false, 'SimDirect', true);
    data = RunMCRoomSim(S,R,Room, Options);

    filename = strcat(strcat(base_name, int2str(nr)), '.mat');
    save(filename,'fs', 'sources', 'receivers', 'data', 'w','l','h','N','M','c')
    %copyfile(filename, '~')
end
