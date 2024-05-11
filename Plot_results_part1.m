%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part 1
clear
clc
close all force
format long e

base_path = "D:\Semester5\1_PINNs_paper_materials\";
data = dlmread(base_path+"EOPIERS1900.txt","",1,0);
interannual = dlmread(base_path+"interannual_with_uncertainty.txt");
interannual = interannual(interannual(:,1)<=2018, :);
data = data(data(:,1)<=2018, :);

dt_sampling = mean(diff(data(:,1)))*365.25;
n1 = round(365.25/dt_sampling)*1;
n6 = round(365.25/dt_sampling)*6;
% time year
t_year = data(:,1);
x_pole = data(:,2)*1e3; % mas
y_pole = data(:,4)*1e3; % mas

%% plot with respect to mean of 2002-2018?
use_mean_2002_2018 = true;

if use_mean_2002_2018
    x_pole = x_pole - mean(x_pole([t_year>=2002 & t_year<2019]));
    y_pole = y_pole - mean(y_pole([t_year>=2002 & t_year<2019]));
end

%% 1: plot raw data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h=figure('Position', get(0, 'Screensize'));
set(h, 'PaperPositionMode', 'auto')
h.WindowState = 'maximized';
set(gcf,'renderer','painters')
hold on
set(gca, 'FontName', 'SansSerif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(t_year, x_pole, "Color","#0072BD", "LineWidth",3);hold on;
plot(t_year, y_pole, "Color","#D95319", "LineWidth",3);
set(gca,'xticklabel',[])
xlim([data(1,1), data(end,1)])
NumTicks = 7;
set(gca,'XTick',[1900, 1920, 1940, 1960, 1980, 2000, 2018])
xticklabels({"1900", "1920", "1940", "1960", "1980", "2000", "2019"})
%xlabel("time [year]")
ylabel("[mas]")
xL=xlim;
yL=ylim;
text(xL(1)+1,0.999*yL(2),'(a)','HorizontalAlignment',...
    'left','VerticalAlignment','top','FontSize',65)

box on
grid on
H = gca;
H.LineWidth=3;
H.FontSize=40;

[hiii, icons] = legend("$x_p$", "$y_p$", "Location","southeast", "interpreter","latex", "fontsize", 60);
icons1 = findobj(icons, 'type', 'line');
icons2 = findobj(icons, 'type', 'text');
hiii.ItemTokenSize=[120,0];
set(icons1, 'linewidth', 30);
set(icons1,'XData',[0.5 0.05])
% get(icons2, 'position')
for i=1:size(icons2,1)
% icons1(i).XData = [0.2 0];
icons2(i).Position = icons2(i).Position + [0.25 0 0];
end
set(gca, 'FontName', 'SansSerif')
set(gcf, 'Position', get(0, 'Screensize'));
exportgraphics(gcf, "Figure1_data.pdf", "Resolution",200)
close all force

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2
clear
clc
close all force
format long e

base_path = "D:\Semester5\1_PINNs_paper_materials\";
data = dlmread(base_path+"EOPIERS1900.txt","",1,0);
interannual = dlmread(base_path+"interannual_with_uncertainty.txt");
interannual = interannual(interannual(:,1)<2019, :);

%% 2: plot interannual
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% with respect to mean 1900-1905
u = interannual(interannual(:,1) <= 1900 & interannual(:, 1) >=1900, :);
mean_x_1900_1905 = mean(u(:,2));
mean_y_1900_1905 = mean(u(:,3));
interannual(:, 2) = interannual(:, 2) - mean_x_1900_1905;
interannual(:, 3) = interannual(:, 3) - mean_y_1900_1905;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h=figure('Position', get(0, 'Screensize'));
set(h, 'PaperPositionMode', 'auto')
h.WindowState = 'maximized';
set(gcf,'renderer','painters')
axes1 = polaraxes('Layer','top');%,'Position',[0.15 0.15 0.8 0.8]
hold(axes1,'all');
hold on
set(gca, 'FontName', 'SansSerif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rlim_max=12.01; % meters
mas2cm=3.0887;

num_point=1000;
polarplot([0:2*pi/(num_point-1):2*pi],rlim_max*ones(1,num_point),'-k','linewidth',1.5,'parent',axes1);

polarplot([3*pi/2 pi/2],[rlim_max rlim_max],'-k','linewidth',1,'parent',axes1);
polarplot([0 pi],[0 rlim_max],'-k','linewidth',1,'parent',axes1);

for i=1:max(size(interannual))
    %(i,1)
    if interannual(i,2)>0 && interannual(i,3)>0
        ang_a(i,1) = atan(interannual(i,3)/interannual(i,2));
    elseif interannual(i,2)>0 && interannual(i,3)<0
        ang_a(i,1) = 2*pi-atan(abs(interannual(i,3)/interannual(i,2)));
    elseif interannual(i,2)<0 && interannual(i,3)>0
        ang_a(i,1) = pi-atan(abs(interannual(i,3)/interannual(i,2)));
    elseif interannual(i,2)<0 && interannual(i,3)<0
        ang_a(i,1) = pi+atan(abs(interannual(i,3)/interannual(i,2)));
    end
end

mag_a = sqrt(interannual(:,2).^2 + interannual(:,3).^2) * mas2cm/100;

polarplot(ang_a, mag_a,'b','linewidth', 5 ,'markerfacecolor','b','parent',axes1,"linestyle", "-");
axes1.ThetaZeroLocation='bottom';
axes1.ThetaDir = 'clockwise';
axes1.ThetaLim = [0 360];
axes1.FontSize = 30;
axes1.RAxisLocation = 180;
axes1.RLim = [0 rlim_max];
axes1.RTick = [0:3:rlim_max];
axes1.RTickLabel={'';'';'';'';''};
axes1.LineWidth=3.00;
%axes1.GridColor = 'k';
thetaticks(0:45:315)
% }}}

% plot scale {{{
polarplot([3*pi/4 3*pi/4],[6 9],'-k','linewidth',15,'parent',axes1);
% Create textbox
% % annotation(h,'textbox',...
% %     [0.515 0.68 0.18 0.08],...
% %     'String',{'3 m'},...
% %     'FitBoxToText','off','EdgeColor','none', "FontSize",50, 'rotation',53);
% % % }}}

annotation(h,'textbox',...
    [0.48 0.68 0.18 0.045],...
    'String',{'3 m'},...
    'FitBoxToText','off','EdgeColor','none', "FontSize",50, 'rotation',53);


thetaticklabels({"0^\circ","45^\circ W","90^\circ W",...
    "135^\circ W", "180^\circ", "135^\circ E",...
    "90\circ E", "45^\circ E"})

% set(gca,'Color',[1 1 0.6])
set(gcf, 'Position', get(0, 'Screensize'));

rlim([0, 12])
thetalim([0, 135])

xL=rlim;
yL=thetalim;
xL(1) = xL(1) + 15;
yL(1) = -130;

annotation(h,'textbox',...
    [0.6 0.900 0.15 0.08],...
    'String',{'(c)'},...
    'FitBoxToText','off','EdgeColor','none', "FontSize",50);

set(gca, 'FontName', 'SansSerif')

exportgraphics(gcf, "Figure1_inset.pdf", "Resolution",200)
close all force

% view(30, 40)
