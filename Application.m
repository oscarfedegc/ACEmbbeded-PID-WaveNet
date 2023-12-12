%{
	===========================================================================                               
           Filename: Application.m
               Date: Sep 29th, 2022
        Last update: Dec 12th, 2023   

        CC BY-NC-SA 3.0 (http://creativecommons.org/licenses/by-nc-sa/3.0/)
	===========================================================================
%}

classdef Application < handle
    
    % Class properties
    properties (Access = public)
        % Execution parameters
        isRandom, isTraining, isFixed, behavior
        
        % Time simulation and trayectory
        period, fTime, fixed, posPitch, posYaw, states, signals
        samples, desired, instants
        
        % WaveNet-IIR Parameters
        inputs, outputs, neurons, coeffsM, coeffsN, pSignal, activac,
        learningRates, weigths, scales, shifts, feedbacks, feedforwards
        eMemory, zMemory, yMemory, fMemory, dMemory, gamma, wave
        
        mGamma, tDisturbance
        
        % Fileparameters
        prefix, fWeights, fShifts, fScales, fBacks, fForwards, reqTime
        fWavelets, fNormErrs
        
        % PID gains and update rates
        gains, updateRates
    end
    
    % Main methods
    methods (Access = public)
        %{
            Setting the parameters proposed by the designer
        %}
        function this = Application()
            clc, close all
            
            this.isRandom   = true;
            this.isTraining = true;
            this.isFixed    = false;
            this.activac    = ActivationFunctions();
            this.wave       = 'rasp1';

            % Time simulation and trayectory
            this.period = 0.005;                        % Sampling period, seconds
            this.fTime  = 15;                           % Simulation time, seconds
            this.fixed  = [deg2rad(5), deg2rad(10)];    % Values for a fixed trajectory
            this.posPitch = deg2rad([-30 0 0 0]);
            this.posYaw   = deg2rad([0 0 20 20 20 20]);
            this.tDisturbance = 6;                      % seconds
            
            % WaveNet-IIR parameters
            this.inputs  = 2;       % System's input
            this.outputs = 2;       % System's output
            this.neurons = 3;       % Number of neurons in hidden layer
            this.coeffsM = 5;       % Number of feedback coefficients
            this.coeffsN = 4;       % Number of feed-forward coefficients
            this.pSignal = 3.36e-2; % Persistent signal for IIR filters
            this.learningRates = [3e-8 3e-8 3e-8 5e-8 5e-5];
            
            % Setting PID Controller parameters and its update rates
            this.prefix  = 'AC-PID-WNet-MassDisturbance';
            this.inputs  = this.inputs + 1;
            this.outputs = this.outputs + 1;
            
            this.gains = [100 0.5 10; 100 0.5 10];
            this.gains = 0.8 * this.gains;
            this.updateRates = [1 0.001 1; 1 0.001 1];
            
            this.prefix = [this.prefix '-' this.wave];
            
            this.activac.choose(this.wave); % Activation function for WaveNet
            
            this.gamma = 0.01;              % Discount factor for Reinforcement Learning
            this.initialize();
            this.run();
        end
        
        %{
            Initializes the parameters values
        %}
        function initialize(this)
            clc, format long;
            disp('Limpiando la consola de comandos...')
            fprintf('Tipo de controlador: %s.\n', this.prefix)
            
            % Initialize parameters
            this.getTrajectory();
            
            % Getting the initial data for WaveNet-IIR
            [this.weigths, this.scales, this.shifts,this.feedbacks, this.feedforwards, ...
                this.eMemory, this.zMemory, this.yMemory, this.fMemory, this.dMemory] = ...
                Application.getInitParams(this.isRandom, this.inputs, this.outputs, ...
                this.neurons, this.coeffsM, this.coeffsN, this.prefix);
            
            % Initialize the matrices to savings
            [this.fWeights, this.fShifts, this.fScales, this.fBacks, this.fForwards, ...
                this.fWavelets, this.fNormErrs] = ...
                Application.getCSV(this.samples, this.neurons, ...
                this.coeffsM, this.coeffsN, this.outputs);
            
            % Initializing a matrix for saving the general behavior of the system
            J = 1 + this.outputs*6 +  2*length(this.gains(1,:)) - 3;
            
            this.behavior = zeros(this.samples,J);
            this.states = zeros(this.samples,4);
            this.states(1,1) = -0.70;    
            this.signals = [0,0];
            
            this.mGamma = zeros(this.samples,4);
        end
        
        %{
            Excutes the algorithm
        %}
        function run(this)
            disp('Generando simulación, espere...')
            isOK = this.RLPID();
            
            if isOK
                this.saveCSV();
                this.plotResults();
            end
        end
   
        %{
            Generates a polinomial trajectory desired from simulation time.
        %}
        function getTrajectory(this)
            if this.isFixed
                refPitch = this.fixed(1);
                refYaw = this.fixed(2);
            else
                refPitch = this.posPitch;
                refYaw = this.posYaw;
            end
            
            [this.samples, dPitch, this.instants] = ...
                Application.createTrajectory(this.period, this.fTime, refPitch);
            [~, dYaw, ~] = ...
                Application.createTrajectory(this.period, this.fTime, refYaw);
            
            this.desired = [dPitch dYaw];
        end
        
        %{
            Implements the control algorithm using WaveNet-IIR and RL
            technique for the regulation of positions.
        
            @params object  $this   Instance of the main class
            @return boolean $isOk   Indicates the simulation status
        %}
        function isOK = RLPID(this)
            this.feedbacks = this.feedbacks(1:2,:);
            this.feedforwards = this.feedforwards(1:2,:);
            isOK = true;
            
            for idx = 1:this.samples
                kT  = this.instants(idx);
                yRef = this.desired(idx,:);
                
                % Control signals for each axis
                up = Application.getCtrlSignal(this.signals(1), this.gains(1,:), ...
                    this.eMemory(:,1));
                uy = Application.getCtrlSignal(this.signals(2), this.gains(2,:), ...
                    this.eMemory(:,2));
                
                this.signals = [up uy];
                
                % Position "mesaured" from linear or nonlinear models
                [this.states, modelRho, modelGamma] = Application.NonlinearModel(...
                    this.states, this.signals, idx, this.period, this.tDisturbance);
                yMes = [this.states(idx,1), this.states(idx,3)];                
                this.mGamma(idx,1:2) = modelGamma;
                
                % Calculating of Wavelets and WaveNet
                [tau, func, dfunc] = this.activac.funcOutputs(kT, this.shifts, this.scales);
                
                % Actor part
                actorWeigths = this.weigths(1:2,:);
                wNetOutputs  = Application.getOutputWaveNet(this.signals, actorWeigths, func);
                
                % IIR Filters outputs
                tempZ = [wNetOutputs; this.zMemory(1:this.coeffsM-1,1:2)];
                [actor, Rho, Gamma] = Application.getOutputFilters(this.feedbacks, ...
                    this.feedforwards, tempZ, this.yMemory(:,1:2), this.pSignal);
                
                this.mGamma(idx,3:4) = Gamma;
                
                error   = yMes - actor;                         % Estimation error
                epsilon = yRef - yMes;                          % Tracking error
                reward  = 0.5 * (error(1) + error(2))^2;        % Reinforcement signal
                
                % Critic part
                criticWeigths = this.weigths(3,:);
                sigma = [this.signals reward];
                critic = Application.getOutputWaveNet(sigma, criticWeigths, func);
                
                TD = reward + this.gamma*critic - this.zMemory(2,3);
                
                % Update memories                
                this.zMemory = [wNetOutputs critic; this.zMemory(1:this.coeffsM-1,:)];
                this.yMemory = [      actor critic; this.yMemory(1:this.coeffsN-1,:)];
                this.fMemory = [              func; this.fMemory(1:this.coeffsM-1,:)];
                this.dMemory = [             dfunc; this.dMemory(1:this.coeffsM-1,:)];
                this.eMemory = [    epsilon   0-TD; this.eMemory(1:2,:)];
                
                % Actor-Critic - Gradient method
                [dW, dC, dD] = this.gradientActorRL(this.signals, error, this.feedbacks, ...
                    this.fMemory, this.zMemory(:,1:2), this.yMemory(:,1:2), this.pSignal);
                
                [dO, db, da] = this.gradientCriticRL([this.signals reward], TD, this.gamma, ...
                    criticWeigths, func, dfunc, tau);
                
                % Update rules for WaveNet-IIR parameters
                actorWeigths  =  actorWeigths - this.learningRates(1)*dW;
                criticWeigths = criticWeigths - this.learningRates(1)*dO;
                
                this.weigths      = [actorWeigths; criticWeigths];
                this.shifts       = this.shifts - this.learningRates(2)*db;
                this.scales       = this.scales - this.learningRates(3)*da;
                this.feedbacks    = this.feedbacks - this.learningRates(4)*dC;
                this.feedforwards = this.feedforwards - this.learningRates(5)*dD;
                
                % Update rules for controllers
                if this.isTraining
                    npPID = Application.updateGains(this.gains(1,:), ...
                            this.updateRates(1,:), this.eMemory(:,1), error(1), Gamma(1));
                    nyPID = Application.updateGains(this.gains(2,:), ...
                            this.updateRates(2,:), this.eMemory(:,2), error(2), Gamma(2));
                        
                    this.gains = [npPID; nyPID];
                else
                    npPID = this.gains(1,:);
                    nyPID = this.gains(2,:);
                end
                
                % Setting the current parameter values
                [this.fWeights, this.fShifts, this.fScales, this.fBacks, this.fForwards] = ...
                    Application.setCSV(idx, this.fWeights, this.fShifts, this.fScales, ...
                    this.fBacks, this.fForwards, this.weigths, this.shifts, this.scales, ...
                    this.feedbacks, this.feedforwards);
                
                this.fWavelets(idx,:) = func;
                this.fNormErrs(idx,:) = [norm(modelRho - Rho), norm(modelGamma - Gamma)];
                
                % Update matrix to save
                this.behavior(idx,:) = [kT yRef yMes actor epsilon error up uy npPID nyPID ...
                    reward TD critic];
                
                this.logger(idx)
                
                if isnan(actor(1))
                    isOK = false;
                    return
                end
            end
        end
        
        %{
            Shows the simulation information every timestep.
        
            @param integer $idx     Stands of the current iteration.
        %}
        function logger(self, idx)
            clc
            data = self.behavior(idx,:);
            
            fprintf('=======================================================\n')
            fprintf(' time = %.3f\n', data(1));
            fprintf('ref01 = %8.3f\tref02 = %8.3f\n', data(2), data(3));
            fprintf('mes01 = %8.3f\tmes02 = %8.3f\n', data(4), data(5));
            fprintf('act01 = %8.3f\tact02 = %8.3f\n', data(6), data(7));
            fprintf('epp01 = %8.3f\tepy02 = %8.3f\n', data(8), data(9));
            fprintf('err01 = %8.3f\terr02 = %8.3f\n', data(10), data(11));
            fprintf('  kpp = %8.3f\t  kip = %8.3f\t kdp = %8.3f\n', ...
                data(14), data(15), data(16));
            fprintf('  kpp = %8.3f\t   ip = %8.3f\t kdp = %8.3f\n', ...
                data(17), data(18), data(19));
            fprintf('   up = %8.3f\t   uy = %8.3f\n', data(12), data(13));
            fprintf(' rwrd = %8.3f\t   TD = %8.3f\tcrit = %8.3f\n', ...
                data(20), data(21), data(22));
            fprintf('=======================================================\n')
        end
        
        %{
            Calls the functions to create the graphics.
        %}
        function plotResults(this)
            this.showNeuronsParams()
            this.showWeights()
            this.showGains()
            this.showNormError()
            this.showSignalsRL()
            this.showTrackingResults()
        end
        
        %{
            Print the simulation results for desired positions.
        %}
        function showTrackingResults(this)
            kT = this.behavior(:,1);
            
            figure('Name','Tracking results','NumberTitle','off','units',...
                'normalized','outerposition',[0 0 1 1]);
            
            subplot(3,2,1)
                hold on
                plot(kT,this.behavior(:,2),'k--','LineWidth',1)
                plot(kT,this.behavior(:,4),'r','LineWidth',1)
                title('Pitch')
                ylabel('Position, y [grad]')
                legend('Reference','Real')
                xline(this.tDisturbance,'-.','DisplayName','Disturbance 1.5 mass')
                
            subplot(3,2,3)
                plot(kT,this.behavior(:,8),'r','LineWidth',1)
                ylabel('Error [grad]')
            subplot(3,2,5)
                hold on
                plot(kT,this.behavior(:,12),'r','LineWidth',1)
                ylabel('Input signal [Nm]')
                xlabel('Time, kT [sec]')
                legend('\tau')
                
            subplot(3,2,2)
                hold on
                plot(kT,this.behavior(:,3),'k--','LineWidth',1)
                plot(kT,this.behavior(:,5),'r','LineWidth',1)
                title('Yaw')
                ylabel('Position, y [grad]')
                legend('Reference','Real')
                xline(this.tDisturbance,'-.','DisplayName','Coupling pertubation')
                
            subplot(3,2,4)
                plot(kT,this.behavior(:,9),'r','LineWidth',1)
                ylabel('Error [grad]')
            subplot(3,2,6)
                hold on
                plot(kT,this.behavior(:,13),'r','LineWidth',1)
                ylabel('Input signal [Nm]')
                xlabel('Time, kT [sec]')
                legend('\tau')
        end
        
        %{
            Shows the behavior of neuron parameters and their outputs.
        %}
        function showNeuronsParams(this)
            kT = this.behavior(:,1);
            
            fig = length(this.fScales(1,:)) - 1;
            cols = 3;
            rows = fig;
            
            figure('Name','Scaling and shifting parameters',...
                'NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
            for item = 1:fig
                subplot(rows, cols, 1 + cols*(item-1))
                    plot(kT,this.fScales(:,item + 1),'r','LineWidth',1)
                    ylabel(sprintf('a_{%i}', item))
            end
            
            for item = 1:fig
                subplot(rows, cols, 2 + cols*(item-1))
                    plot(kT,this.fShifts(:,item + 1),'r','LineWidth',1)
                    ylabel(sprintf('b_{%i}', item))
            end
            
            for item = 1:fig
                subplot(rows, cols, 3 + cols*(item-1))
                    plot(kT,this.fWavelets(:,item + 1),'r','LineWidth',1)
                    ylabel(sprintf('\\psi(\\tau_{%i})', item))
            end
        end
        
        %{
            Shows the gains behavior from controllers.
        %}
        function showGains(this)
            kT = this.behavior(:,1);
            
            figure('Name','Controllers gains','NumberTitle','off',...
                'units','normalized','outerposition',[0 0 1 1]);
            
            subplot(3,2,1)
                plot(kT, this.behavior(:,14),'r','LineWidth',1)
                ylabel('Proportional, k_{p_\theta}')
                title('Pitch')
                
            subplot(3,2,3)
                plot(kT, this.behavior(:,15),'r','LineWidth',1)
                ylabel('Integral, k_{i_\theta}')
                
            subplot(3,2,5)
                plot(kT, this.behavior(:,16),'r','LineWidth',1)
                ylabel('Derivative, k_{d_\theta}')
                
            subplot(3,2,2)
                plot(kT, this.behavior(:,17),'r','LineWidth',1)
                ylabel('Proportional, k_{p_\phi}')
                title('Yaw')
                
            subplot(3,2,4)
                plot(kT, this.behavior(:,18),'r','LineWidth',1)
                ylabel('Integral, k_{i_\phi}')
                
            subplot(3,2,6)
                plot(kT, this.behavior(:,19),'r','LineWidth',1)
                ylabel('Derivative, k_{d_\phi}')
        end
        
        function showNormError(this)
            kT = this.behavior(:,1);
            
            figure('Name','Identification error normalized','NumberTitle','off',...
                'units','normalized','outerposition',[0 0 1 1]);
            subplot(2,1,1)
                plot(kT,this.fNormErrs(:,2),'LineWidth',1)
                ylabel('||\Phi - \Phi_e||')
                xlabel('Time, kT [s]')
                
            subplot(2,1,2)
                plot(kT,this.fNormErrs(:,3),'LineWidth',1)
                ylabel('||\Gamma - \Gamma_e||')
                xlabel('Time, kT [s]')
        end
        
        %{
            Shows the synaptic weights behavior of the WaveNet-IIR.
        %}
        function showWeights(this)
            kT = this.behavior(:,1);
            
            fig = length(this.fWeights(1,:)) - 1;
            cols = 3;
            rows = round(fig/cols);
            
            figure('Name','Synaptic weights','NumberTitle','off',...
                'units','normalized','outerposition',[0 0 1 1]);
            for item = 1:rows
                subplot(rows, cols, 1 + cols*(item-1))
                    plot(kT,this.fWeights(:,item + 1),'r','LineWidth',1)
                    ylabel(sprintf('w_{\\theta_%i}', item))
                    
                subplot(rows, cols, 2 + cols*(item-1))
                    plot(kT,this.fWeights(:,item + 1 + rows),'r','LineWidth',1)
                    ylabel(sprintf('w_{\\phi_%i}', item))
                    
                subplot(rows, cols, 3 + cols*(item-1))
                    plot(kT,this.fWeights(:,item + 1 + 2*rows),'r','LineWidth',1)
                    ylabel(sprintf('\\omega_{%i}', item))
            end
        end
        
        %{
            Plots the critic, reward and temporal difference signals.
        %}
        function showSignalsRL(this)
            kT = this.behavior(:,1);
            
            idx = 20:1:22;
            figure('Name','Reinforcement and TD Signals','NumberTitle','off',...
                'units','normalized','outerposition',[0 0 1 1]);
            subplot(3,1,1)
            plot(kT,this.behavior(:,idx(1)),'r','LineWidth',1)
            ylabel('Reward signal')
            subplot(3,1,2)
            plot(kT,this.behavior(:,idx(2)),'b','LineWidth',1)
            ylabel('TD signal')
            subplot(3,1,3)
            plot(kT,this.behavior(:,idx(3)),'g','LineWidth',1)
            ylabel('Output critic')
            xlabel('Time, kT [s]')
        end
        
        %{
            Saves the similation results into CSV files.
        %}
        function saveCSV(this)
            writematrix(this.weigths,       ['csv-files/' this.prefix ' weights.csv'])
            writematrix(this.weigths,       ['csv-files/' this.prefix ' weights.csv'])
            writematrix(this.shifts,        ['csv-files/' this.prefix ' shifts.csv'])
            writematrix(this.scales,        ['csv-files/' this.prefix ' scales.csv'])
            writematrix(this.feedbacks,     ['csv-files/' this.prefix ' feedbacks.csv'])
            writematrix(this.feedforwards,  ['csv-files/' this.prefix ' feedforwards.csv'])
            
            time = this.behavior(:,1);
            items = 1:3:length(time) - 1;
            
            this.behavior = this.behavior(items,:);
            this.behavior(:,2:11) = rad2deg(this.behavior(:,2:11));
            
            T = array2table(this.behavior, 'VariableNames',{'idx','yp','yy','mp','my', ...
                'np','ny','epp','epy','ep','ey','up','uy','kpp','kpi','kpd','kyp',...
                'kyi','kyd','r','td','x'});
            
            writetable(T, ['csv-files/' this.prefix ' behavior.csv'])
            
            this.fWeights = [time this.fWeights];
            this.fShifts = [time this.fShifts];
            this.fScales = [time this.fScales];
            this.fBacks = [time this.fBacks];
            this.fForwards = [time this.fForwards];
            this.fWavelets = [time this.fWavelets];
            this.fNormErrs = [time this.fNormErrs];
            
            this.fWeights = this.fWeights(items,:);
            this.fShifts = this.fShifts(items,:);
            this.fScales = this.fScales(items,:);
            this.fBacks = this.fBacks(items,:);
            this.fForwards = this.fForwards(items,:);
            this.fWavelets = this.fWavelets(items,:);
            this.fNormErrs = this.fNormErrs(items,:);
            
            w = array2table(this.fWeights);
            q = array2table(this.fShifts);
            s = array2table(this.fScales);
            c = array2table(this.fBacks);
            d = array2table(this.fForwards);
            l = array2table(this.fWavelets);
            e = array2table(this.fNormErrs);

            writetable(w,  ['csv-files/' this.prefix ' behavior weigths.csv'])
            writetable(q,  ['csv-files/' this.prefix ' behavior shifts.csv'])
            writetable(s,  ['csv-files/' this.prefix ' behavior scales.csv'])
            writetable(c,  ['csv-files/' this.prefix ' behavior feedbacks.csv'])
            writetable(d,  ['csv-files/' this.prefix ' behavior forwards.csv'])
            writetable(l,  ['csv-files/' this.prefix ' behavior wavelets.csv'])
            writetable(e,  ['csv-files/' this.prefix ' behavior norms.csv'])
        end
	end % of public methods
    
    methods (Static)        
        %{
            Evaluate a polynomial function over a given range.

            @params array   $tSegment   Segment time
                    double  $pInit      Initial value
                    double  $pFinal     Final value
            @return array   $rst        Trajectory calculated
        %}
        function rst = trayectorySegment(tSegment,pInit,pFinal)

            t = (tSegment-min(tSegment))/(max(tSegment)-min(tSegment));
            rst = pInit + 3*(pFinal-pInit)*t.^2 - 2*(pFinal-pInit)*t.^3;
        end
        
        %{
            Generates a polinomial trajectory desired from simulation time.

            @params double  $period         Sampling period
                    double  $fTime          Simulation time
                    array   $references     Values desired for the trajectories
            @return integer $samples        Number of samplings
                    matrix  $desired        Trajectory desired for each angle
                    array   $instants       
        %}
        function [samples, desired, instants] = createTrajectory(period, fTime, references)
            samples  = round(fTime/period) + 1;
            desired  = zeros(samples,1);
            instants = (0:1:samples-1)*period;

            if isempty(references)
                references = 0;
            end
            
            if length(references) == 1
                desired(:,1) = references;
                return
            else
                count = length(references);
            end

            % Calculating intervals for the equation's cases
            dPos = round(linspace(1,samples,count));

            % Calculating the angle position for each sample about the axis
            for i = 1:length(dPos)-1
                t = instants(dPos(i):dPos(i+1));
                desired(dPos(i):dPos(i+1),1) = ...
                    Application.trayectorySegment(t,references(i),references(i+1));
            end
        end
        
        %{
            Initalize the WaveNet-IIR parameters. If the variable isRandom is true, data is
            randomly generated else data is loaded from a file.

            @params double  $isRandom       Sampling period
                    double  $inputs         Number of inputs for neural network
                    double  $outputs        Number of outputs for neural network
                    double  $neurons        Number of neurons in the internal layer
                    double  $coeffsM        Number of feedbacks coefficients
                    double  $coeffsN        Number of feedforwards coefficients
                    double  $filename       Filename to load initial data
            @return matrix  $weigths        Synaptics weights matrix 
                    array   $scales         Scale array for the wavelet
                    array   $shifts         Shift array for the wavelet
                    matrix  $feedbacks      Feedback coefficients matrix
                    matrix  $feedforwards   Feedforward coefficients matrix
                    matrix  $eMemory        Memory array for tracking error
                    matrix  $zMemory        Memory array for outputs the neural network 
                    matrix  $yMemory        Memory array for position estimated
                    matrix  $fMemory        Memory array for wavelet functions
                    matrix  $dMemory        Memory array for derivatives of wavelet
        %}
        function [weigths, scales, shifts, feedbacks, feedforwards, eMemory, ...
                zMemory, yMemory, fMemory, dMemory] = getInitParams(isRandom, ...
                inputs, outputs, neurons, coeffsM, coeffsN, prefix)

            randd = @(a,b,f,c) a + (b-a)*rand(f,c);

            eMemory   = zeros(3,outputs);
            zMemory   = zeros(coeffsM,outputs);
            yMemory   = zeros(coeffsN,outputs);
            fMemory   = zeros(coeffsM,neurons);
            dMemory   = zeros(coeffsM,neurons);

            if isRandom
                weigths      = randd(-0.5,0.5,inputs,neurons);
                scales       = randd(1,1,1,neurons);
                shifts       = randd(-1,1,1,neurons);
                feedbacks    = randd(-0.5,0.5,outputs,coeffsM);
                feedforwards = randd(-0.5,0.5,outputs,coeffsN);
            else
                weigths      = load(['csv-files/' prefix ' weights.csv']);
                scales       = load(['csv-files/' prefix ' shifts.csv']);
                shifts       = load(['csv-files/' prefix ' scales.csv']);
                feedbacks    = load(['csv-files/' prefix ' feedbacks.csv']);
                feedforwards = load(['csv-files/' prefix ' feedforwards.csv']);
            end
        end
        
        %{
            Estimate the angular accelerations of the helicopter from its nonlinear mode.

            @params double $theta       Angular position for pitch axis
                    array  $dots        Angular speed for each axis
                    array  $signals     Control signals
                    double $idx         Iteration value
                    double $period      Sampling time
            @return array  $newState    Angular accelarations estimated about each axis
        %}
        function [newState, rho, gamma] = NonlinearModel(states, signals, idx, period,...
                tDisturbance)
            % Set constants
            m   = 1.3872;
            g   = 9.8100;
            l   = 0.1860;
            Bp  = 0.8000;
            By  = 0.3180;
            Jp  = 0.0384;
            Jy  = 0.0432;
            Kpp = 0.2040;
            Kyy = 0.0720;
            Kyp = 0.0219;
            Kpy = 0.0068;
            
            % Indices a parametric disturbance at tDisturbance seconds
            if period*idx >= tDisturbance
                m = 1.5*m;
                Jp = 1.5*Jp;
            end

            % States
            x1 = states(idx,1);
            x2 = states(idx,2);
            x4 = states(idx,4);

            % Differential Equations (Nonlinear model)
            % Representation in state variables
            f1 = -(Bp*x2 + m*(x2*l)^2*sin(x1)*cos(x1) + m*g*l*cos(x1))/(Jp + m*l^2);
            f2 = -(By*x4 + 2*m*sin(x1)*cos(x1)*x2*x4*l^2)/(Jy + m*(l*cos(x1))^2);

            g11 = Kpp/(Jp + m*l^2);
            g12 = Kpy/(Jp + m*l^2);
            g21 = Kyp/(Jy + m*(l*cos(x1))^2);
            g22 = Kyy/(Jy + m*(l*cos(x1))^2);

            f = [x2; f1; x4; f2];
            g = [0,0; g11,g12; 0,0; g21,g22];
            u = [signals(1); signals(2)];

            xdot = f + g*u;

            % States -- Euler aproximation of integration
            states(idx+1,:) = states(idx,:) + period*xdot';

            % Return
            newState = states;
            rho = states(idx,:) + f'*period;
            rho = rho([2 4]);
            gamma = (g*period*u)';
            gamma = gamma([2 4]);
        end
        
        %{
            Calculate the output of the WaveNet.

            @params array   $signals     Control signals for each axis
                    array   $weigths     Synaptics weights matrix 
                    array   $wavelets    Arrangement of evaluated functions
            @return array   $z           Outputs of the neural network
        %}
        function z = getOutputWaveNet(signals, weights, wavelets)
            z = sum(signals)*(weights * wavelets')';
        end
        
        %{
            Calculate the output of the IIR filters.

            @params matrix  $backs          Feedback coefficients matrix
                    matrix  $forwards       Feedforward coefficients matrix
                    matrix  $zMemory        Memory array for outputs the neural network 
                    matrix  $yMemory        Memory array for position estimated
                    double  $pSignal        Persistent signal for the IIR filters
            @return array   $y              Position estimaed by IIR Filters
                    array   $gamma          Wavenet parameters to auto-tunning
        %}
        function [y, rho, gamma] = getOutputFilters(backs, forwards, zMemory, yMemory, pSignal)
            gamma = sum(backs * zMemory);
            rho = sum(forwards*yMemory);
            y = gamma + rho*pSignal;
        end
        
        %{
            Calculating the control signal using the controller gains and the
            memory array for tracking error.

            @params double  $signal     Sum of the control signals on each axis
                    array   $gains      Controller gains array for an angle
                    array   $epsilon    Tracking error array for an angle
                    double  $idx        Refers to the desired axis
            @return double  $rst        New control signal generated
        %}
        function rst = getCtrlSignal(signal, gains, epsilon)
            kp = gains(1);
            ki = gains(2);
            kd = gains(3);

            rst = signal + kp*(epsilon(1)-epsilon(2)) + ki*epsilon(1) - ...
                kd*(epsilon(1) - 2*epsilon(2) + epsilon(3));
        end
        
        %{
            Apply update rule for controller gains.

            @params array   $gains          Control signals for the axis
                    matrix  $rates          Values for wavelets
                    array   $epsilon        Tracking error array
                    double  $error          Estimation error
                    array   $gamma          Parameter from the IIR Filters
                    double  $idx            Refers to the desired axis
            @return matrix  $rst            New gain values
        %}
        function rst = updateGains(gains, rates, epsilon, error, gamma)
            kp = gains(1);      ki = gains(2);      kd = gains(3);
            mp = rates(1);      mi = rates(2);      md = rates(3);

            kp = kp + mp*error*gamma*(epsilon(1) - epsilon(2));
            ki = ki + mi*error*gamma*(epsilon(1));
            kd = kd + md*error*gamma*(epsilon(1) - 2*epsilon(2) + epsilon(3));

            rst = abs([kp, ki, kd]);
        end
        
        %{
            Initializes the variables to store the behavior of each parameter.

            @params matrix  $fWeights       Matrix for the synaptic weights of each axis
                    array   $fShifts        Array for wavelet function shifts
                    array   $fScales        Array for wavelet function scales
                    matrix  $fBacks         Matrix for the feedbacks coefficients
                    matrix  $fForwards      Matrix for the feedforwards coefficients
            @return double  $samples        Number of samplings
                    double  $neurons        Number of neurons
                    double  $coeffsM        Number of feedbacks coefficients
                    double  $coeffsN        Number of feedforwards coefficients
        %}
        function [fWeights, fShifts, fScales, fBacks, fForwards, fWavelets, fNormErrs]=...
                getCSV(samples, neurons, coeffsM, coeffsN, outputs)

            fWeights  = zeros(samples, outputs*neurons);
            fShifts   = zeros(samples, neurons);
            fScales   = zeros(samples, neurons);
            fBacks    = zeros(samples, 2*coeffsM);
            fForwards = zeros(samples, 2*coeffsN);
            fWavelets = zeros(samples, neurons);
            fNormErrs   = zeros(samples, 2);
        end
        
        %{
            Stores the value of each parameter in its corresponding variable.
        %}
        function [fWeights, fShifts, fScales, fBacks, fForwards] = setCSV(idx, ...
                fWeights, fShifts, fScales, fBacks, fForwards, weigths, shifts, ...
                scales, feedbacks, feedforwards)

            fWeights(idx,:) = [weigths(1,:) weigths(2,:) weigths(3,:)];
             fShifts(idx,:) = shifts;
             fScales(idx,:) = scales;
              fBacks(idx,:) = [feedbacks(1,:) feedbacks(2,:)];
           fForwards(idx,:) = [feedforwards(1,:) feedforwards(2,:)];
        end
        
        %{
            Calculates the output of the IIR filters.

            @params array   $signals        Control signals for each axis
                    array   $tau            Values for wavelets
                    array   $error          Estimatio error array
                    matrix  $feedbacks      Feedbacks coefficients matrix
                    matrix  $feedforwards   Feedforward coefficients matrix
                    matrix  $yMemory        Memory array for position estimated
                    matrix  $fMemory        Memory array for wavelet functions
                    matrix  $zMemory        Memory array for outputs the neural network 
                    matrix  $yMemory        Memory array for position estimated
                    double  $pSignal        Persistent signal for the IIR filters
            @return matrix  $d{W,q,s,C,D}   Variation in each parameter
        %}
        function [dW, dq, ds, dC, dD] = gradientPID(signals, tau, error, feedbacks, ...
                fMemory, dMemory, zMemory, yMemory, pSignal)

            U = sum(signals);
            p = length(signals);
            Ie = error .* diag(ones(p,1),0);
            
            dW = U*Ie*feedbacks*fMemory;
            dq = U*error*(feedbacks*dMemory);
            ds = dq;
            dC = U*Ie*zMemory';
            dD = pSignal*Ie*yMemory';

            for i = 1:length(ds)
                ds(i) = ds(i) * tau(i);
            end
        end
        
        %{
            Calculates the gradients for Actor stage.

            @params array   $signals        Control signals for each axis
                    array   $error          Estimatio error array
                    matrix  $feedbacks      Feedbacks coefficients matrix
                    matrix  $fMemory        Memory array for wavelet functions
                    matrix  $zMemory        Memory array for outputs the neural network 
                    matrix  $yMemory        Memory array for position estimated
                    double  $pSignal        Persistent signal for the IIR filters
            @return matrix  $d{W,C,D}       Variation in each parameter
        %}
        function [dW, dC, dD] = gradientActorRL(signals, error, feedbacks, ...
                fMemory, zMemory, yMemory, pSignal)
            dW = error'.*(feedbacks*(sum(signals)*fMemory));
            dC = error'.*zMemory';
            dD = error'.*(pSignal*yMemory)';
        end
        
        %{
            Calculates the gradients for Critic stage.

            @params array   $signals        Control signals for each axis
                    double  $TD             Temporal difference signal
                    array   $omega          Synaptic weights
                    array   $tau            Values for wavelets
            @return matrix  $d{O,b,a}       Variation in each parameter
        %}
        function [dO, db, da] = gradientCriticRL(signals, TD, gamma, omega, func, dfunc, tau)
            aux = TD*gamma*sum(signals);
            dO = aux.*func;
            db = -aux.*(omega.*dfunc);
            da = tau.*db;
        end
    end % of static methods
end