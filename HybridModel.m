%% Simulation of a hybrid oscillatory interference (OI) - continuous 
%  attractor network (CAN) model of grid cell firing 
%
%  Daniel Bush, UCL Institute of Cognitive Neuroscience (2016)
%  www.danbush.co.uk
%
%  For details and description of the model, see:
%  
%  Bush D, Burgess N (2014) Journal of Neuroscience 34: 5065-5079
%
%  Bush D, Schmidt-Hieber C (in press) Hippocampal Microcircuits: A 
%  Computational Modeller's Resource Book. Springer, NY
%
%  Note that 2D simulations of 20 minutes random foraging take around 20
%  minutes to run on a standard desktop PC

%% Provide some settings for the simulation
Environment     = '1D';         % Environment type (1D or 2D)
gridScale       = 30;           % Grid scale (cm)

%% Load tracking data, assign VCO orientations and initial phase offsets
switch Environment
    case '1D'
        load('TrackingData.mat','OneD');                
        track           = OneD;
        nGC             = size(OneD.vcoInput,2);                        % Number of grid cells
        vcoOrientations = [2*pi*ones(nGC,1) ; 5*pi/3*ones(nGC,1) ; pi/3*ones(nGC,1)];
        vcoPhases       = [linspace(4*pi,2*pi/nGC,nGC)' ; linspace(2*pi,2*pi/nGC,nGC)' ; linspace(2*pi,2*pi/nGC,nGC)'];
        vcoInput        = OneD.vcoInput;                                % VCO to GC connectivity template
        clear OneD
    case '2D'
        load('TrackingData.mat','TwoD');                
        track           = TwoD;
        nGC             = size(TwoD.vcoInput,2);                        % Number of grid cells
        vcoOrientations = repmat(pi/3:pi/3:2*pi,sqrt(nGC),1);           % VCO orientations
        vcoOrientations = vcoOrientations(:);
        vcoPhases       = repmat(linspace(2*pi,2*pi/sqrt(nGC),sqrt(nGC))',1,6);
        vcoPhases       = vcoPhases(:);                                 % VCO phases
        vcoInput        = TwoD.vcoInput;                                % VCO to GC connectivity template
        clear TwoD
end

%% Generate the VCO rate functions
fBaseline       = 8;                                                    % Baseline oscillation frequency
beta            = 1/gridScale;                                          % Slope of the running speed - VCO bursting firing frequency relationship
vcoSignal       = nan(length(track.t_log),length(vcoPhases));           % Assign some memory
for vco         = 1 : length(vcoPhases)
    vector              = [cos(vcoOrientations(vco,1)) sin(vcoOrientations(vco,1))];
    disp                = dot(repmat(vector,length(track.t_log),1),[track.x_log' track.y_log'],2); clear vector
    velocity            = diff(disp)./track.dt; clear disp
    frequency           = fBaseline + beta .* velocity; clear velocity
    phase               = cumsum([vcoPhases(vco) ; frequency.*2*pi.*track.dt]); clear frequency
    circ_dist           = abs(angle(exp(1i*track.head_dir)./exp(1i*vcoOrientations(vco,1))))';    
    vcoSignal(:,vco)    = (1+cos(phase)).*(circ_dist<pi/2); clear phase circ_dist
end
clear vco beta vcoOrientations vcoPhases fBaseline

%% Generate the grid cell input rate functions
gcSignal        = nan(length(track.t_log),nGC);                         % Assign some memory
for gc          = 1 : nGC
    gcSignal(:,gc)      = sum(vcoSignal(:,vcoInput(:,gc)==1),2);        % Integration VCO input rate functions to each grid cell
end
clear gc vcoSignal vcoInput

%% Initialise the recurrent inhibitory connectivity
gcReps          = 48;                                                   % Number of grid cells that share a spatial offset (i.e. have the same VCO inputs)
intReps         = 12;                                                   % Number of interneurons that share a spatial offset (i.e. receive input from the same grid cells)
gcInhW          = 0.2;                                                  % Gain (i.e. relative strength) of grid cell to interneuron weights
gcInhSig        = 0.2;                                                  % Standard deviation of Gaussian distributed grid cell to interneuron weights
gcInhConn       = 0.5;                                                  % Mean grid cell to interneuron connection probability
inhGCW          = 0.04;                                                 % Gain (i.e. relative strength) of inhibitory to grid cell weights
inhGCSig        = 0.1;                                                  % Standard deviation of Gaussian distributed inhibitory to grid cell weights
inhGCConn       = 0.7;                                                  % Mean interneuron to grid cell connection probability
inhGcWeights    = toeplitz((cos(linspace(pi,3*pi-2*pi/nGC,nGC))+1)/2);  % Interneuron to grid cell connectivity profile
excW            = zeros(nGC*gcReps+nGC*intReps,nGC*gcReps+nGC*intReps); % Excitatory weight matrix
inhW            = zeros(nGC*gcReps+nGC*intReps,nGC*gcReps+nGC*intReps); % Inhibitory weight matrix
for c           = 1 : nGC
    exc         = (c-1)*gcReps+1:c*gcReps;
    inh         = nGC*gcReps+1+(c-1)*intReps:nGC*gcReps+c*intReps;
    excW(exc,inh)       = (1+gcInhSig*randn(length(exc),length(inh))).*gcInhW; clear exc
    for c2      = 1 : nGC
        exc     = (c2-1)*gcReps+1:c2*gcReps;
        inhW(inh,exc)   = (1+inhGCSig*randn(length(inh),length(exc))).*inhGcWeights(c,c2).*inhGCW; clear exc
    end
    clear c2 inh
end
excW(excW<0) 	= 0;                                                    % Ensure all weights are positive
excW            = excW.*double(rand(size(excW))<=gcInhConn);            % Sparsify excitatory connectivity
inhW(inhW<0) 	= 0;                                                    % Ensure all weights are positive
inhW            = inhW.*double(rand(size(inhW))<=inhGCConn);            % Sparsify inhibitory connectivity
clear c gcInhConn gcInhSig gcInhW inhGCConn inhGCSig inhGCW inhGcWeights

%% Initialise the neural dynamics
vcoReps         = 30;                                                   % Number of VCO cells that share a phase offset, for each orientation / ring attractor
vcoRate         = 50;                                                   % Mean firing rate of each VCO input
vcoGCw          = 4.5e-3;                                               % VCO to grid cell weights
excPersI        = 8.5e-4;                                               % Persistent current to grid cells
inhPersI        = 1.25e-4;                                              % Persistent current to interneurons
excSig          = 1.25e-4;                                              % Standard deviation of noise input to grid cells
inhSig          = 2.5e-4;                                               % Standard deviation of noise input to interneurons

Cm              = 0.5e-3;                                               % Membrane conductance (mF)
gm              = 25e-6;                                                % Leak conductance (mS)
Vl              = -70;                                                  % Leak reversal potential (mV)
Vt              = -50;                                                  % Firing threshold (mV)
Vr              = -65;                                                  % Reset potential (mV)

tauAMPA         = 5.26;                                                 % AMPA current decay constant (ms)
E_AMPA          = 0;                                                    % AMPA reversal potential (mV)
g_AMPA          = 0.215e-4;                                             % Maximum AMPA conductance (mS)

tauGABA_r       = 3;                                                    % GABA rise time constant (ms)
tauGABA_1       = 50;                                                   % GABA decay constant (ms)
tauGABA_2       = (tauGABA_r*tauGABA_1)/(tauGABA_r+tauGABA_1);          % GABA time constant (ms)
GABA_B          = ((tauGABA_2/tauGABA_1)^(tauGABA_r/tauGABA_1)-(tauGABA_2/tauGABA_1)^(tauGABA_r/tauGABA_2))^-1;
E_GABA          = -80;                                                  % GABA reversal potential (mV)
g_GABA          = 0.14e-4;                                              % Maximum GABA conductance (mS)

v               = Vl*ones(size(excW,1),1);                              % Membrane voltage of all cells (mV)
AMPAExp         = zeros(size(excW,1),1);                                % AMPA open channel probability for all cells
GABAExp1        = 0.8*ones(size(excW,1),1);                             % GABA open channel probability for all cells
GABAExp2        = 0.04*ones(size(excW,1),1);                            % GABA close channel probability for all cells
if strncmp(Environment,'1D',2)
    logged      = ceil(rand*nGC*gcReps);                                % Choose a random grid cell to log
    v_log       = nan(1,max(track.t_log)*1000);                         % Assign memory for the membrane voltage log
    spikeTimes  = nan(max(track.t_log)*nGC*(gcReps+intReps),2);         % Assign memory to log spike times for each cell (estimating mean firing rate of 1Hz)
elseif strncmp(Environment,'2D',2)
    offset      = floor(rand*nGC);                                      % Select a random grid cell offset
    logged  	= offset*gcReps + (1:gcReps);                           % Log the spike times for all grid cells with that spatial offset...
    logged      = [logged nGC*gcReps+offset*intReps+(1:intReps)];       % ...and the corresponding population of interneurons
    spikeTimes  = nan(max(track.t_log)*(gcReps+intReps),2);           	% Assign memory to log spike times for those cells (estimating mean firing rate of 1Hz)    
    clear offset
end
sCount          = 1;                                                    % Index for logging spike times
tic                                                                     % Start the clock

%% Run the neural dynamics
for time        = track.t_log(1)*1000 : max(track.t_log)*1000                             % For each 1ms time step
    
    % Compute the total number of Poisson input spikes to each grid cell 
    % from VCOs and add those to the GABA open channel probabilities
    if mod(time,track.dt*1000)  == 0
        index                   = round(time/1000/track.dt - (track.t_log(1)-track.dt)/track.dt);
        vcoInputs               = reshape(poissrnd(repmat(vcoReps*vcoRate*track.dt*gcSignal(index,:),gcReps,1)),nGC*gcReps,1);
        GABAExp1(1:nGC*gcReps)	= GABAExp1(1:nGC*gcReps) + vcoInputs*vcoGCw; 
        GABAExp2(1:nGC*gcReps)	= GABAExp2(1:nGC*gcReps) + vcoInputs*vcoGCw; clear index vcoInputs
    end
    
    % Compute the AMPA, GABA and persistent currents
    AMPA_I      = -g_AMPA .* AMPAExp .* (v-E_AMPA);
    GABA_I      = -g_GABA .* GABA_B .* (GABAExp1-GABAExp2) .* (v-E_GABA);    
    pers_I      = [excPersI*ones(nGC*gcReps,1)+randn(nGC*gcReps,1)*excSig ; inhPersI*ones(nGC*intReps,1)+randn(nGC*intReps,1)*inhSig];
    
    % Update the AMPA and GABA open channel probabilities
    AMPAExp     = AMPAExp .* exp(-1/tauAMPA);
    GABAExp1    = GABAExp1.* exp(-1/tauGABA_1);
    GABAExp2    = GABAExp2.* exp(-1/tauGABA_2);
    
    % Update the membrane voltages (in two steps for numerical stability)
    v           = v + 0.5 * (1/Cm) * (AMPA_I + GABA_I + pers_I - (v - Vl) .* gm);
    v           = v + 0.5 * (1/Cm) * (AMPA_I + GABA_I + pers_I - (v - Vl) .* gm); clear AMPA_I GABA_I pers_I        
                 
    % Find the neurons that fired, update the AMPA and GABA open channel probabilities
    fired       = v>=Vt;
    AMPAExp     = AMPAExp  + sum(excW(fired,:),1)';
    GABAExp1    = GABAExp1 + sum(inhW(fired,:),1)';
    GABAExp2    = GABAExp2 + sum(inhW(fired,:),1)';
    
    % Reset the membrane potential
    v(fired)    = Vr;
    
    % Log the output (membrane voltage etc)
    if strncmp(Environment,'1D',2)
        v_log(time)             = v(logged);
    elseif strncmp(Environment,'2D',2)
        fired                   = fired(logged);
    end
    if sum(fired)>0    
        spikeTimes(sCount:sCount+sum(fired)-1,:)	= [nonzeros(fired.*(1:size(fired,1))') time*ones(sum(fired),1)/1000];
        sCount 	= sCount + sum(fired);
    end
    clear fired
    
end
spikeTimes      = spikeTimes(1:sCount-1,:);                             % Truncate the spike time log
spikeInds    	= round((spikeTimes(:,2)-track.t_log(1))./track.dt)+1;  % Compute the tracking index at each spike time
spikeTimes(:,3) = track.x_log(spikeInds)';                              % Compute the x location of each spike
spikeTimes(:,4) = track.y_log(spikeInds)'; clear spikeInds              % Compute the y location of each spike
simTime         = toc;                                                  % Compute the total simulation time
clear time sCount AMPAExp Cm E_AMPA E_GABA GABAExp1 GABAExp2 GABA_B Vl Vt
clear excPersI excSig excW fired g_AMPA g_GABA gcSignal gm gridScale 
clear inhPersI inhSig inhW tauAMPA tauGABA_1 tauGABA_2 tauGABA_r v tic 
clear vcoGCw vcoRate vcoReps

%% Analyse and plot the output
if strncmp(Environment,'1D',2)
    
    % Linearly interpolate the voltage signal to remove spikes
    interpWin   = [-1 20];                                              % Interpolation window around each spike (ms)
    spikeInds   = find(v_log==Vr)-1;                                    % Find the index of all spikes fired by the logged cell
    for spike   = 1 : length(spikeInds)                                 % Linearly interpolate the voltage trace around each spike
        v_log(spikeInds(spike)+interpWin(1):spikeInds(spike)+interpWin(2))	= linspace(v_log(spikeInds(spike)+interpWin(1)),v_log(spikeInds(spike)+interpWin(2)),diff(interpWin)+1);
    end
    clear interpWin spikeInds spike Vr
    
    % Mean normalise the voltage signal
    vMean       = nanmean(v_log);                                       % Mean normalise the voltage trace (for filtering)
    v_log       = v_log - vMean;
    v_log(isnan(v_log)) = 0;                                            % Eliminate any nan entries (for filtering)
    
    % Filter in the <3Hz and 5-11Hz range with a 400th order FIR filter    
    a         	= fir1(400, [5 11]*2/1000, 'band');                     % Set up a theta band pass filter
    b          	= fir1(400, 3*2/1000, 'low');                           % Set up a low pass ramp filter
    Theta      	= filtfilt(a,1,v_log);                                  % Filter in the theta band
    Theta       = abs(hilbert(Theta));                                  % Extract theta amplitude
    Ramp        = filtfilt(b,1,v_log); clear a b                        % Filter in the ramp band
        
    % Compute the mean firing rate of all grid cells with that offset
    binSize     = 0.2;                                                  % Temporal bin size (s)
    gridOffset  = ceil(logged/gcReps);                                  % Identify all cells that shared a spatial offset with the logged cell
    spikeInds   = ismember(spikeTimes(:,1),(gridOffset-1)*gcReps : gridOffset*gcReps);    
    spikeTs     = spikeTimes(spikeInds,2); clear gridOffset spikeInds   % Extract the spike times of all those cells
    spikeRate   = histc(spikeTs,0:binSize:max(track.t_log)); clear spikeTs
    spikeRate   = spikeRate(1:end-1)/gcReps/binSize;                    % Compute the firing rate for that sub-population of cells
    fields      = spikeRate>=(0.1*max(spikeRate));                      % Identify candidate firing fields as rate > 10% of the peak rate
    fields      = regionprops(fields,'PixelIdxList');                   % Narrows those down to fields with at least three consecutive bins
    inField     = zeros(size(v_log));                                   % Generate a firing field mask to examine changes in theta and ramp depolariation
    for f       = 1 : length(fields)
        if length(fields(f,1).PixelIdxList)>=3
            inds(1)     = round(find(track.t_log>=(fields(f,1).PixelIdxList(1)*binSize),1,'first')*track.dt*1000);
            inds(2)     = round(find(track.t_log<=(fields(f,1).PixelIdxList(end)*binSize),1,'last')*track.dt*1000);
            inField(inds(1):inds(2)) = 1; clear inds
        end
    end
    clear f fields
    
    % Plot the output
    h1          = subplot(4,1,1);                                       % Plot the mean firing rate along the track
    plot(h1,binSize/2:binSize:max(track.t_log)-binSize/2,spikeRate,'k','LineWidth',2)    
    ylabel(h1,{'Firing','Rate (Hz)'},'FontSize',16)    
    ylim([0 10])
    clear spikeRate
    
    h2          = subplot(4,1,2);                                       % Plot the membrane voltage trace along the track
    plot(h2,linspace(track.t_log(1),track.t_log(end),length(v_log)),v_log+vMean,'k','LineWidth',2)    
    hold(h2,'on')
    APs         = round(spikeTimes(spikeTimes(:,1)==logged,2)*1000);    % Add the action potentials back in
    for spike   = 1 : length(APs)
        plot(h2,[APs(spike) APs(spike)]./1000,[v_log(APs(spike))+vMean 0],'k','LineWidth',2)
    end
    clear spike APs
    hold(h2,'off')
    ylabel(h2,{'Membrane','Potential (mV)'},'FontSize',16)
    ylim([-80 10])
    
    h3          = subplot(4,1,3);                                       % Plot theta amplitude in the membrane potential along the track
    plot(h3,linspace(track.t_log(1),track.t_log(end),length(v_log)),Theta,'k','LineWidth',2)   
    hold(h3,'on')                                                       % Add mean theta amplitude inside and outside the firing fields
    plot(h3,linspace(track.t_log(1),track.t_log(end),length(v_log)),mean(Theta(inField==1))*ones(size(v_log)),'r--','LineWidth',2)
    plot(h3,linspace(track.t_log(1),track.t_log(end),length(v_log)),mean(Theta(inField==0))*ones(size(v_log)),'b--','LineWidth',2)
    hold(h3,'off')
    ylabel(h3,{'Theta';'Amplitude (mV)'},'FontSize',16)
    ylim([-5 5])
    
    h4          = subplot(4,1,4);                                       % Plot ramp depolarisation in the membrane potential along the track
    plot(h4,linspace(track.t_log(1),track.t_log(end),length(v_log)),Ramp,'k','LineWidth',2)    
    hold(h4,'on')                                                       % Add mean ramp depolarisation inside and outside the firing fields
    plot(h4,linspace(track.t_log(1),track.t_log(end),length(v_log)),mean(Ramp(inField==1))*ones(size(v_log)),'r--','LineWidth',2)
    plot(h4,linspace(track.t_log(1),track.t_log(end),length(v_log)),mean(Ramp(inField==0))*ones(size(v_log)),'b--','LineWidth',2)
    hold(h4,'off')
    xlabel(h4,'Time (s)','FontSize',16)
    ylabel(h4,{'Ramp','Depolarisation (mV)'},'FontSize',16)
    ylim([-5 5])
    clear fields
    
    linkaxes([h1 h2 h3 h4],'x'); clear h1 h2 h3 h4
    xlim([1 max(track.t_log)])                                          % Set the x axis limits to movement periods only
    v_log       = v_log + vMean; clear vMean                            % Return the voltage trace to its original offset
    
elseif strncmp(Environment,'2D',2)
    
    % Generate smoothed rate maps
    binSize     = 2;                                                    % Spatial bin size (cm)
    smthKern    = 5;                                                    % Size of boxcar smoothing kernel (bins)
    mapSize     = ceil(max(spikeTimes(spikeTimes(:,1)<=gcReps,3:4))./binSize);
    gridInds    = ceil(spikeTimes(spikeTimes(:,1)<=gcReps,3:4)./binSize);
    gridInds    = accumarray(gridInds,1,mapSize)/gcReps;                % Generate raw grid cell rate map
    intInds     = ceil(spikeTimes(spikeTimes(:,1)>gcReps,3:4)./binSize);
    intInds     = accumarray(intInds,1,mapSize)/gcReps;                 % Generate raw interneuron rate map
    locInds     = ceil([track.x_log' track.y_log']./binSize);
    locInds     = accumarray(locInds,1,mapSize)*track.dt;               % Generate raw occupancy map
    denom       = filter2(ones(smthKern),double(locInds>0));
    denom(denom==0) = nan;
    locInds     = filter2(ones(smthKern),locInds)./denom;               % Smooth the occupancy map
    gridInds    = filter2(ones(smthKern),gridInds)./denom;              % Smooth the grid cell and interneuron rate maps
    intInds     = filter2(ones(smthKern),intInds)./denom; clear denom smthKern
    gridMap     = (gridInds./locInds)'; 
    intMap      = (intInds./locInds)'; clear gridInds intInds locInds
    
    % Plot the grid cell spike locations
    subplot(2,2,1);                                                     % Plot animal trajectory and grid cell spike locations
    plot(track.x_log,track.y_log,'Color',[0.8 0.8 0.8])
    hold on
    scatter(spikeTimes(spikeTimes(:,1)<=gcReps,3),spikeTimes(spikeTimes(:,1)<=gcReps,4),'r.')
    hold off
    title('Grid Cells','FontSize',24)
    axis square
    xlabel('Position (cm)','FontSize',16)
    ylabel('Position (cm)','FontSize',16)
    
    % Plot the grid cell rate map
    subplot(2,2,2);                                                     % Plot the smoothed grid cell rate map
    imagesc(binSize/2:binSize:mapSize(1)*binSize-binSize/2,binSize/2:binSize:mapSize(1)*binSize-binSize/2,gridMap)
    set(gca,'YDir','normal')
    title(['Peak rate = ' num2str(max(gridMap(:)),3), 'Hz'],'FontSize',16)
    axis square
    xlabel('Position (cm)','FontSize',16)
    ylabel('Position (cm)','FontSize',16)
    
    % Plot the interneuron spike locations
    subplot(2,2,3);                                                     % Plot animal trajectory and interneuron spike locations
    plot(track.x_log,track.y_log,'Color',[0.8 0.8 0.8])
    hold on
    scatter(spikeTimes(spikeTimes(:,1)>gcReps,3),spikeTimes(spikeTimes(:,1)>gcReps,4),'b.')
    hold off
    title('Interneurons','FontSize',24)
    axis square
    xlabel('Position (cm)','FontSize',16)
    ylabel('Position (cm)','FontSize',16)
    
    % Plot the interneurons rate map
    subplot(2,2,4);                                                     % Plot the smoothed interneuron rate map
    imagesc(binSize/2:binSize:mapSize(1)*binSize-binSize/2,binSize/2:binSize:mapSize(1)*binSize-binSize/2,intMap)
    set(gca,'YDir','normal')
    title(['Peak rate = ' num2str(max(intMap(:)),3), 'Hz'],'FontSize',16)
    axis square
    xlabel('Position (cm)','FontSize',16)
    ylabel('Position (cm)','FontSize',16)
    linkaxes    
    xlim([0 100])
    ylim([0 100])
    clear mapSize
    
end
clear Vr binSize gcReps logged intReps nGC
