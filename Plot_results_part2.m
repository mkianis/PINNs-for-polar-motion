clear
clc
close all force
format long e
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_path_all = "D:\Semester5\Adhikari_Long_Range_Polar_Motion\" + ...
    "13_final_codes_PINNs\results_PINNs_New_GM\";

base_path_no_physics = "D:\Semester5\Adhikari_Long_Range_Polar_Motion\" + ...
    "13_final_codes_PINNs\results_PINNs_New_GM_no_physics\";

base_data = "D:\Semester5\Adhikari_Long_Range_Polar_Motion\" + ...
    "13_final_codes_PINNs\";
path_uncertainty = "D:\Semester5\Adhikari_Long_Range_Polar_Motion\13_final_codes_PINNs\ensembles\results_PINNs_no_physics\";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pred_x_with_gm = dlmread(base_path_all+"errors_with_gm_xp.txt");
pred_y_with_gm = dlmread(base_path_all+"errors_with_gm_yp.txt");

pred_x_without_gm = dlmread(base_path_all+"errors_without_gm_xp.txt");
pred_y_without_gm = dlmread(base_path_all+"errors_without_gm_yp.txt");

pred_x_no_physics = dlmread(base_path_no_physics+"errors_xp.txt");
pred_y_no_physics = dlmread(base_path_no_physics+"errors_yp.txt");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
interannual = dlmread("D:\Semester5\1_PINNs_paper_materials\interannual_with_uncertainty.txt");
u = interannual(interannual(:,1) < 2019 & interannual(:, 1) >=2002, :);
mean_x_2002_2018 = mean(u(:,2));
mean_y_2002_2018 = mean(u(:,3));
interannual(:, 2) = interannual(:, 2) - mean_x_2002_2018;
interannual(:, 3) = interannual(:, 3) - mean_y_2002_2018;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread("D:\Semester5\1_PINNs_paper_materials\EOPIERS1900.txt","",1,0);
dt_sampling = mean(diff(data(:,1)))*365.25;
n1 = round(365.25/dt_sampling)*1;
n6 = round(365.25/dt_sampling)*6;

t = data(:,1);

[r,c] = find(interannual(:,1)<1976);

[r2,c] = find(interannual(:,1)<2019);

time = interannual(r,1);
n = size(time,1);

h=figure('Position', get(0, 'Screensize'));
set(h, 'PaperPositionMode', 'auto')
h.WindowState = 'maximized';
set(gcf,'renderer','painters')
%tiledlayout(1,1,'TileSpacing','tight','Padding','tight')
hold on
set(gca, 'FontName', 'SansSerif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot what
plot_what = "xp";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch plot_what
    case "xp"

        Qx = data(:,3)*1e3;
        Qx = sqrt(movmean(movmean(Qx.^2, n1), n6));

        y1 = interannual(:,2)+Qx;
        y1 = y1(r2, 1);
        y2 = interannual(:,2)-Qx;
        y2 = y2(r2, 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ax1 = gca;
        plot(ax1,t(r2, 1), interannual(r2,2),"color","#0072BD", "LineWidth",8);hold on
        patch(ax1,[t(r2, 1)' fliplr(t(r2, 1)')], [y1' fliplr(y2')], [0 0.4470 0.7410], 'facealpha',0.15, 'edgecolor', [0 0.4470 0.7410], 'edgealpha', 0.15)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        plot(ax1, time, 1000*pred_x_no_physics+interannual(1:n,2), "color",	"#77AC30", "LineWidth",8)

        counter =1;
        for kkk=401:600
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 500;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 500;

        y1 = 1000*pred_x_no_physics+interannual(1:n,2) + tmp_x;
        y2 = 1000*pred_x_no_physics+interannual(1:n,2) - tmp_x;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [0.4660 0.6740 0.1880],...
            'facealpha',0.1, 'edgecolor', [0.4660 0.6740 0.1880], 'edgealpha', 0.5)

        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        plot(ax1, time, 1000*pred_x_without_gm+interannual(1:n,2), "color","#FF00FF", "LineWidth",8)
        counter =1;
        for kkk=201:400
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 250;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 250;

        y1 = 1000*pred_x_without_gm+interannual(1:n,2) + tmp_x;
        y2 = 1000*pred_x_without_gm+interannual(1:n,2) - tmp_x;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [1 0 1],...
            'facealpha',0.1, 'edgecolor', [1 0 1], 'edgealpha', 0.5)

        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        plot(ax1, time, 1000*pred_x_with_gm+interannual(1:n,2), "color","#000000", "LineWidth",10)
        counter =1;
        for kkk=0:200
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 250;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 250;

        y1 = 1000*pred_x_with_gm+interannual(1:n,2) + tmp_x;
        y2 = 1000*pred_x_with_gm+interannual(1:n,2) - tmp_x;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [0 0 0],...
            'facealpha',0.1, 'edgecolor', [0 0 0], 'edgealpha', 0.4)

        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t = xlim;
        t = [t(1):0.1:t(end)]';
        t_pred_past = t(t>=1900 & t<1976, 1);
        t_pred_future = t(t>=2019);
        t_train = t(t>=1976 & t<2019, 1);
        ylim([-160,50])
        y = ylim;

        y1 = repmat(y(1),size(t_train));
        y2 = repmat(y(end),size(t_train));
        patch(ax1, [t_train' fliplr(t_train')], [y1' fliplr(y2')], [0 1 1], 'facealpha',0.1, 'edgecolor', [0 1 1], 'edgealpha', 0.1)


        [h,icons]=legend("$x_p$","", "\textsf{no processes} $\hat{x}_p$",...
            "",...
            "\textsf{no core dynamics} $\hat{x}_p$",...
            "",...
            "\textsf{all processes} $\hat{x}_p$",...
            "", "", ...
            "Location","southeast", "interpreter", "latex", "fontsize", 35);

        icons1 = findobj(icons, 'type', 'line');
        icons2 = findobj(icons, 'type', 'text');
        h.ItemTokenSize=[120,0];
        set(icons1, 'linewidth', 20);
        set(icons1,'XData',[0.23 0.03])
        % get(icons2, 'position')
        for i=1:size(icons2,1)
            % icons1(i).XData = [0.2 0];
            icons2(i).Position = icons2(i).Position + [0.17 0 0];
        end

    case "yp"

        Qx = data(:,3)*1e3;
        Qx = sqrt(movmean(movmean(Qx.^2, n1), n6));
        Qy = data(:,5)*1e3;
        Qy = sqrt(movmean(movmean(Qy.^2, n1), n6));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ax1 = gca;

        y1 = interannual(:,3)+Qy;
        y1 = y1(r2, 1);
        y2 = interannual(:,3)-Qy;
        y2 = y2(r2, 1);
        plot(ax1,t(r2, 1), interannual(r2,3),"color", "#D95319", "LineWidth",8);hold on
        patch(ax1,[t(r2, 1)' fliplr(t(r2, 1)')], [y1' fliplr(y2')], [0.8500 0.3250 0.0980], 'facealpha',0.15, 'edgecolor', [0.8500 0.3250 0.0980], 'edgealpha', 0.15)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        plot(ax1, time, 1000*pred_y_no_physics+interannual(1:n,3), "color",	"#77AC30", "LineWidth",8)

        counter =1;
        for kkk=401:600
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 500;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 500;


        y1 = 1000*pred_y_no_physics+interannual(1:n,3) + tmp_y;
        y2 = 1000*pred_y_no_physics+interannual(1:n,3) - tmp_y;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [0.4660 0.6740 0.1880],...
            'facealpha',0.1, 'edgecolor', [0.4660 0.6740 0.1880], 'edgealpha', 0.5)
        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        plot(ax1, time, 1000*pred_y_without_gm+interannual(1:n,3), "color","#FF00FF", "LineWidth",8)
        counter =1;
        for kkk=201:400
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 250;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 250;


        y1 = 1000*pred_y_without_gm+interannual(1:n,3) + tmp_y;
        y2 = 1000*pred_y_without_gm+interannual(1:n,3) - tmp_y;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [1 0 1],...
            'facealpha',0.1, 'edgecolor', [1 0 1], 'edgealpha', 0.5)
        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        plot(ax1, time, 1000*pred_y_with_gm+interannual(1:n,3), "color","#000000", "LineWidth",10)
        counter =1;
        for kkk=0:200
            s = readmatrix(path_uncertainty+"errors_"+string(kkk)+".txt");
            if mean(abs(s(:,2))) > 0 &&  mean(abs(s(:,2))) < 10 && mean(abs(s(:,3)))>0 && mean(abs(s(:,3)))<10
                tmp_x{1,counter} = s(:,2);
                tmp_y{1,counter} = s(:,2);
                counter = counter + 1;
            end
            if size(tmp_x,2)>=100
                break
            end
        end
        tmp_x = cell2mat(tmp_x);
        tmp_x = std(tmp_x, [], 2) * 250;
        tmp_y = cell2mat(tmp_y);
        tmp_y = std(tmp_y, [], 2) * 250;


        y1 = 1000*pred_y_with_gm+interannual(1:n,3) + tmp_y;
        y2 = 1000*pred_y_with_gm+interannual(1:n,3) - tmp_y;
        patch(ax1,[time' fliplr(time')], [y1' fliplr(y2')], [0 0 0],...
            'facealpha',0.1, 'edgecolor', [0 0 0], 'edgealpha', 0.4)
        clear s tmp_y tmp_x
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t = xlim;
        t = [t(1):0.1:t(end)]';
        t_pred_past = t(t>=1900 & t<1976, 1);
        t_pred_future = t(t>=2019);
        t_train = t(t>=1976 & t<2019, 1);
        ylim([-350,100])
        y = ylim;

        y1 = repmat(y(1),size(t_train));
        y2 = repmat(y(end),size(t_train));
        patch(ax1, [t_train' fliplr(t_train')], [y1' fliplr(y2')], [0 1 1], 'facealpha',0.1, 'edgecolor', [0 1 1], 'edgealpha', 0.1)


        [h,icons]=legend("$y_p$","", "\textsf{no processes} $\hat{y}_p$",...
            "",...
            "\textsf{no core dynamics} $\hat{y}_p$",...
            "",...
            "\textsf{all processes} $\hat{y}_p$",...
            "", "", ...
            "Location","southeast", "interpreter", "latex", "fontsize", 35);

        icons1 = findobj(icons, 'type', 'line');
        icons2 = findobj(icons, 'type', 'text');
        h.ItemTokenSize=[120,0];
        set(icons1, 'linewidth', 20);
        set(icons1,'XData',[0.23 0.03])
        % get(icons2, 'position')
        for i=1:size(icons2,1)
            % icons1(i).XData = [0.2 0];
            icons2(i).Position = icons2(i).Position + [0.17 0 0];
        end

end

% icons = findobj(icons,'Type','line');
% % icons = findobj(icons,'Marker','none','-xor');
% set(icons([1, 3 , 5, 9, 13]),'linewidth',30);


xlim([1900, 2019])
NumTicks = 7;
set(gca,'XTick',[1900, 1920, 1940, 1960, 1980, 2000, 2019])
xticklabels({"1900", "1920", "1940", "1960", "1980", "2000", "2019"})
xL=xlim;
yL=ylim;

switch plot_what
    case "xp"
        text(xL(1)+4,0.999*yL(2),'(a)','HorizontalAlignment',...
            'left','VerticalAlignment','top','FontSize',70)
    case "yp"
        text(xL(1)+4,0.999*yL(2),'(b)','HorizontalAlignment',...
            'left','VerticalAlignment','top','FontSize',70)
        xlabel("time [year]")
end

grid on
box on

ylabel("[mas]")

H=gca;
H.LineWidth=3;
H.FontSize=40;
set(gca, 'FontName', 'SansSerif')
exportgraphics(gcf, "Figure3_PINNs_prediction_"+plot_what+".pdf", "Resolution",200)

close all force
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part 2

clear
clc
close all force
format long e
warning off
base_path = "D:\Semester5\Adhikari_Long_Range_Polar_Motion\13_final_codes_PINNs\CircularPlots\";

MAE_x = readtable(base_path+"effect_x.csv");
MAE_y = readtable(base_path+"effect_y.csv");
col = MAE_x.Properties.VariableNames;
col = {'GIA-MC', 'B-SL', 'EQ', 'CD', 'B-SL & GIA-MC',...
    'GIA-MC & EQ', 'GIA-MC & CD', 'B-SL & EQ', 'B-SL & CD', ...
    'CD & EQ', 'B-SL & GIA-MC & EQ', 'B-SL & GIA-MC & CD',...
    'GIA-MC & CD & EQ','B-SL & CD & EQ',...
    'B-SL & GIA-MC & CD & EQ'};

MAE_x = table2array(MAE_x(1, :));
MAE_y = table2array(MAE_y(1, :));

% 1. gia
% 2. B-SL
% 3. EQ
% 4. CD
% 5. gia & B-SL
% 6. gia & EQ
% 7. gia & CD
% 8. B-SL & EQ
% 9. B-SL & CD
% 10. EQ & CD
% 11. gia & B-SL & EQ
% 12. gia & B-SL & CD
% 13. gia & EQ & CD
% 14. B-SL & EQ & CD
% 15. gia & B-SL & EQ & CD

ONE_x = [MAE_x(1:4)];
ONE_y = [MAE_y(1:4)];

TWO_x = [MAE_x(5:10)];
TWO_y = [MAE_y(5:10)];

THREE_x = [MAE_x(11:14)];
THREE_y = [MAE_y(11:14)];

FOUR_x = [MAE_x(15)];
FOUR_y = [MAE_y(15)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1: 1-comb
tmp = array2table([ONE_x;ONE_y;sqrt(ONE_x.^2+ONE_y.^2)]',...
    "VariableNames",{'xp', 'yp', 'xp2'}, "RowNames", col(1:4));
tmp = sortrows(tmp, 'xp2', 'ascend');
col_names_comb1 = tmp.Properties.RowNames;
tmp_comb1 = table2array(tmp)';
tmp_comb1 = tmp_comb1(3,:);
%% 2: 2-comb
tmp = array2table([TWO_x;TWO_y;sqrt(TWO_x.^2+TWO_y.^2)]',...
    "VariableNames",{'xp', 'yp', 'xp2'}, "RowNames", col(5:10));
tmp = sortrows(tmp, 'xp2', 'ascend');
col_names_comb2 = tmp.Properties.RowNames;
tmp_comb2 = table2array(tmp)';
tmp_comb2 = tmp_comb2(3,:);
%% 3: 3-comb
tmp = array2table([THREE_x;THREE_y;sqrt(THREE_x.^2+THREE_y.^2)]',...
    "VariableNames",{'xp', 'yp', 'xp2'}, "RowNames", col(11:14));
tmp = sortrows(tmp, 'xp2', 'ascend');
col_names_comb3 = tmp.Properties.RowNames;
tmp_comb3 = table2array(tmp)';
tmp_comb3 = tmp_comb3(3,:);
%% 4: 4-comb
tmp = array2table([FOUR_x;FOUR_y;sqrt(FOUR_x.^2+FOUR_y.^2)]',...
    "VariableNames",{'xp', 'yp', 'xp2'}, "RowNames", col(15));
tmp = sortrows(tmp, 'xp2', 'ascend');
col_names_comb4 = tmp.Properties.RowNames;
tmp_comb4 = table2array(tmp)';
tmp_comb4 = tmp_comb4(3,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h=figure('Position', get(0, 'Screensize'));
set(h, 'PaperPositionMode', 'auto')
h.WindowState = 'maximized';
set(gcf,'renderer','painters')
hold on
set(gca, 'FontName', 'SansSerif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ax1 = gca;
X=[0,  3,4,5,6,  9,10,11,12,13,14,  17,18,19,20];
tmp_X = [X(1)-3:0.1:X(end)+3]';
plot(tmp_X, repmat(9, size(tmp_X)), 'r--', "LineWidth", 4)

tmp = [tmp_comb4,tmp_comb3,tmp_comb2,tmp_comb1];
b=bar(ax1, X, tmp, "facecolor", "flat",'EdgeColor',[0 0 0],'LineWidth',2);
col_names = [col_names_comb4', col_names_comb3', col_names_comb2', col_names_comb1'];
for i=1:15
    b(1).CData(i,:) =[0.4660 0.6740 0.1880];
end


set(gca,'XTick',X)
set(gca,'YTick', [0, 20, 40, 60, 80, 100, 120, 140])

xticklabels({"all","", "      triplet", "", "", "", "", "      double", "", "",...
    "","","        individual"})
xtickangle(0)
 
for i=1:size(tmp,2)
text(X(i), tmp(i)+2, col_names(i), "rotation",90, "FontSize",35, "FontWeight","bold")
end
ylim([0, 150])
xL=xlim;
yL=ylim;
text(xL(1)+6.33,0.999*yL(2),'(c)','HorizontalAlignment',...
    'right','VerticalAlignment','top','FontSize',65)

ylabel("RMSE [mas]")
H = gca;
H.LineWidth=3;
H.FontSize=40;
box on
grid on
% grid minor
set(gca, 'YGrid', 'on', 'XGrid', 'off')
H.XAxis.TickLength = [0 0];
xlim([-1, 21])
set(gca, 'FontName', 'SansSerif')
exportgraphics(gcf, "Figure3_couplings.pdf", "Resolution",200)
close all force