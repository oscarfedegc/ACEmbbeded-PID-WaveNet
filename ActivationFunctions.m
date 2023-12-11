%{
	===========================================================================
                Universidad Politécnica Metropolitana de Hidalgo
                      Master in Aerospace Engineering
                               
        Filename:   ActivationFuncs.m
            Date:   Sep 29th, 2022

        CC BY-NC-SA 3.0 (http://creativecommons.org/licenses/by-nc-sa/3.0/)
	===========================================================================
%}

classdef ActivationFunctions < handle
    
    %% Class properties
    properties (Constant)
        x_Up     = load('coefficients/x_Up.csv');
        Phi_Up   = load('coefficients/Phi_Up.csv');
        dPhi_Up  = load('coefficients/dPhi_Up.csv');

        FlatTop  = load('coefficients/FlatTop.csv');
        dFlatTop = load('coefficients/dFlatTop.csv');
    end
    
    properties (Access = protected)
        funcTag, tau, shifts, scales, Up, FT
    end
    
    %% Main methods
    methods (Access = public)
        %{
            Class constructor.
        
            params  string  $funcTag    Activation function name, example:
                                        "Rasp1", "Morlet", "Atomic"...
            return  object  $this       Instanced object
        %}
        function this = ActivationFunctions()
            this.Up = [this.x_Up; this.Phi_Up; this.dPhi_Up];
            this.FT = [this.FlatTop; this.dFlatTop];
        end
        
        function choose (this, funcTag)
            this.funcTag = funcTag;
        end
        
        %{
            Return the argument vector of activation functions.
        
            @return array   $rst    Vector of functions' arguments
        %}
        function rst = getTau(this)
            rst = this.tau;
        end
        
        %{
            Calculate the tau vector.
        
            @params  object  $this       Instanced class.
                     integer $instant    Time value
                     array   $shifts     Translation values
                     array   $scales     Scale values
        %}
        function setTau(this, inputs, shifts, scales)
            this.tau = (inputs - shifts) ./ scales;
            this.shifts = shifts;
            this.scales = scales;
        end
        
        function [tau, func, dfunc] = funcOutputs(this, inputs, shifts, scales)
            this.setTau(inputs, shifts, scales);
            tau = this.getTau();
            
            [func, dfunc] = this.evaluate();
        end
        
        %{
            Redefines the tau vector
        %}
        function setTauSum(this, inputs, scales, shifts)
            this.tau = (sum(inputs) - shifts) ./ scales;
            this.scales = scales;
            this.shifts = shifts;
        end
        
        %{
        %}
        function [tau, func, dfunc] = funcOutputsSum(this, inputs, shifts, scales)
            this.setTauSum(inputs, scales, shifts)
            tau = this.tau;
            [func, dfunc] = this.evaluate();
            
            func = 1./sqrt(scales) .* func;
        end
        
        %{
            Evaluate the function from tau values.
        
            @return  array   $rst    Activation functions outputs
        %}
        function [func, dfunc] = evaluate(this)
            t = this.tau;
            a = this.scales;
            b = this.shifts;
            j = length(t);
            act = this.funcTag;
            K = 3/16;
            
            % Implements the window functions
            if strcmp(act,'FlatTop1') || strcmp(act,'FlatTop2') || ...
               strcmp(act,'FlatTop3') || strcmp(act,'FlatTop4') || ...
               strcmp(act,'FlatTop5') || strcmp(act,'FlatTop6')
                num = str2double(extractAfter(act,7));
                [func, dfunc] = GeneralCosines(num,t,K);
            elseif strcmp(act,'Hanning')    
                [func, dfunc] = GeneralCosines(7,t,K);
            elseif strcmp(act,'Hamming')
                [func, dfunc] = GeneralCosines(8,t,K);
            elseif strcmp(act,'Blackman')
                [func, dfunc] = GeneralCosines(9,t,K);
            elseif strcmp(act,'Blackman-Harris')
                [func, dfunc] = GeneralCosines(10,t,K);
            elseif strcmp(act,'Atomic')
                x    = this.Up(1,:);
                xUp  = this.Up(2,:);
                dxUp = this.Up(3,:);

                xi = x(1,1);
                xf = x(1,length(x));
                yi = 1;
                yf = length(x);

                idxs = abs(round((t-xi)*(yf-yi)/(xf-xi)));

                for i = 1:j
                    if idxs(i) > length(x)
                        idxs(i) = length(x);
                    end
                end

                func  = xUp(idxs);
                dfunc = dxUp(idxs);
            else
                % Implements tha wavelet functions
                switch act
                    case 'Morlet'
                        w0 = 0.5;
                        func  = cos(w0*t).*exp(-0.5*t.^2);
                        dfunc = (w0.*sin(w0*t).*exp(-0.5*t.^2) + t.*func)./(b);
                    case 'Rasp1'
                        func  = t./(t.^2 + 1).^2;
                        dfunc = (3.*t - 1)./(b.*(t.^2 + 1).^3);
                    case 'Rasp2'
                        func  = t.*cos(t)./(t.^2 + 1);
                        dfunc = ((t.^3 + t).*sin(t) + ...
                           (t.^2 - 1).*cos(t))./(b.*(t.^2 + 1).^2);
                    case 'Rasp3'
                        func  = sin(pi.*t)./(t.^2 - 1);
                        dfunc = (2.*t.*sin(pi.*t) + ...
                           pi.*(t.^2 - 1).*cos(pi.*t))./(b.*(t.^2 - 1).^2);
                    case 'Polywog5'
                        func  = sin(pi.*t)./(t.^2 - 1);
                        dfunc = (2.*t.*sin(pi.*t) + ...
                           pi.*(t.^2 - 1).*cos(pi.*t))./(b.*(t.^2 - 1).^2);
                    otherwise
                        func  = sin(pi.*t)./(t.^2 - 1);
                        dfunc = (2.*t.*sin(pi.*t) + ...
                           pi.*(t.^2 - 1).*cos(pi.*t))./(b.*(t.^2 - 1).^2);
                end
            end
            
            % Applies the normalization for the daugther-wavelets
            func = 1./sqrt(a) .* func;
        end % evaluateFunct()
    end % public methods
end % class 