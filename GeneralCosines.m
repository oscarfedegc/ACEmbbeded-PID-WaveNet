function [Val,DerVal] = GeneralCosines(k,x)
        % Ventanas Flat Top & Clasicas by zecmol Oct/2022
        %   k=1-6 Flat Top Window
        %   k=7   Hanning
        %   k=8   Hamming
        %   k=9   Blackman
        %   k=10  Blackman-Harris
        %   x     Valor donde se desea calcular la ventana

        % Resultado en Variables Val y DerVal

        K=3/16;  % Este valor de K puede ir variando

        %Ventana Coeficientes de F(x)
        A=[ 0.139969361944199, 0.279646558214300, 0.267153027459332, 0.202120943545989, 0.092889443059007, 0.018244323707956, 0; ...
            0.188101508860125, 0.369231207086926, 0.287018792909551, 0.130768792913074, 0.024879698230324, 0,                 0; ...
            0.201424880000000, 0.39291808, 0.28504554, 0.10708192, 0.01352957, 0, 0; ...
            0.209785458600496, 0.407530071019158, 0.2811792263005, 0.092475737791143, 0.009041123909306, 0, 0; ...
            0.21375736, 0.41424355, 0.27860627, 0.08592806, 0.00746476, 0, 0; ...
            0.2710514, 0.43329794, 0.218123, 0.06592546, 0.0108117, 0.0007766, 0.0000139;...
            0.5, 0.5, 0, 0, 0, 0, 0;
            0.54, 0.46, 0, 0, 0, 0, 0;
            0.42, 0.5, 0.08, 0, 0, 0, 0;
            0.35875, 0.48829, 0.14128, 0.01168, 0, 0, 0];
        
        %Derivada Coeficientes F'(x) dF(x)
        B=[0.2796465582143, 0.267153027459332, 0.202120943545989, 0.092889443059007, 0.018244323707956, 0, 0; ...
             0.369231207086926, 0.287018792909551, 0.130768792913074, 0.024879698230324, 0, 0, 0; ...
             0.39291808, 0.28504554, 0.10708192, 0.01352957, 0, 0, 0; ...
             0.407530071019158, 0.2811792263005, 0.092475737791143, 0.009041123909306, 0, 0, 0; ...
             0.41424355, 0.27860627, 0.08592806, 0.00746476, 0, 0,0;
             0.43329794, 0.218123, 0.06592546, 0.0108117, 0.0007766, 0.0000139,0; ...
             0.5, 0, 0, 0, 0, 0, 0; 0.46, 0, 0, 0, 0, 0, 0; 0.5, 0.08, 0, 0, 0, 0, 0;  0.48829, 0.14128, 0.01168, 0, 0, 0, 0];
        
        [~,b]=size(A);
        Val=0;  DerVal=0;
        for l=1:b
            Val=A(k,l)*cos((K*(l-1)).*x)+Val;
            DerVal=-(K*l)* B(k,l)*sin((K*l).*x)+DerVal;
        end
end
