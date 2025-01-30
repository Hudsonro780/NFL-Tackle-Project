%% Header
%{             Robert Hudson
%  
%   NFL ETP Program (Expected Tackle Probability)
% 
%         roberthudson2021@gmail.com   }


%% INTRODUCTION
%{ The following program is designed to take the position of the ball
% carrier, offensive, and defensive players on any given play in order to
% determine the percent chance of a player making the tackle.}


%% Maintenence Functions
close all
clear
clc

%% Identify User Inputs
rnginput = input("Please enter an integer value to identify the random number generation." + ...
    "\n(To allow repeatability in data, enter previously used integers.):\n");


%% Import Files

% These options are to import specific columns as specfic variable types
% into the table

opt_games = detectImportOptions("games.csv");
opt_games = setvartype(opt_games, {'gameId'}, 'string');

opt_plays = detectImportOptions("plays.csv");
opt_plays = setvartype(opt_plays, {'gameId','playId','ballCarrierId'}, 'string');

opt_players = detectImportOptions("players.csv");
opt_players = setvartype(opt_players, {'nflId'}, 'string');

opt_tackles = detectImportOptions("tackles.csv");
opt_tackles = setvartype(opt_tackles, {'gameId','playId','nflId'}, 'string');

opt_track1 = detectImportOptions("tracking_week_1.csv");
opt_track1 = setvartype(opt_track1, {'gameId','playId','nflId','frameId'}, 'string');

% Assigns the tables to the following variable names using the precoded
% options

games = readtable("games.csv",opt_games);
players = readtable("players.csv",opt_players);
plays = readtable("plays.csv", opt_plays);
Tackles = readtable("tackles.csv", opt_tackles);
TrackWeek1 = readtable("tracking_week_1.csv",opt_track1);
TrackWeek2 = readtable("tracking_week_2.csv",opt_track1);
TrackWeek3 = readtable("tracking_week_3.csv",opt_track1);
TrackWeek4 = readtable("tracking_week_4.csv",opt_track1);
TrackWeek5 = readtable("tracking_week_5.csv",opt_track1);
TrackWeek6 = readtable("tracking_week_6.csv",opt_track1);
TrackWeek7 = readtable("tracking_week_7.csv",opt_track1);
TrackWeek8 = readtable("tracking_week_8.csv",opt_track1);
TrackWeek9 = readtable("tracking_week_9.csv",opt_track1);


%% Initialize Functions for Later Use

% Creates an angle in degrees between two arbitrary points
function angle = find_angle(x1,y1,x2,y2)
    angle = rad2deg(atan2(y2-y1,x2-x1));
end

% Determines if an angle is within acceptable bounds to be close to
% tackling a ballcarrier
function is_between_result = is_between(angle_diff, bounds)
    if (angle_diff <= bounds) && (angle_diff >= -bounds)
        is_between_result = true;
    else
        is_between_result = false;
    end
end

% Determines if a blocker is within the sector between the defender and the
% ballcarrier
function is_within_sector_result = is_within_sector(defender_coords, ballcarrier_coords, blocker_coords, dist_to_bc)
    boundary_angle = find_angle(defender_coords(1), defender_coords(2), ballcarrier_coords(1), ballcarrier_coords(2));
    blocker_angle = find_angle(defender_coords(1), defender_coords(2), blocker_coords(1), blocker_coords(2));
    angle_diff = mod(boundary_angle - blocker_angle + 180 + 360,360) - 180;

    if is_between(angle_diff, 30) && (norm(defender_coords - blocker_coords) < dist_to_bc)
        is_within_sector_result = true;
    else
        is_within_sector_result = false;
    end
end

%identifies the defenders with a blocker in between them and the ballcarier
function is_blocked_result = is_blocked(defender_coords, bc_coords, dist_to_bc, x_blocker_list, y_blocker_list)
    for i = 1:length(x_blocker_list)
        if is_within_sector(defender_coords, bc_coords, [x_blocker_list(i), y_blocker_list(i)], min(2,dist_to_bc))
           is_blocked_result = true;
           return
        end
    end
    is_blocked_result = false;
end

% SMOTE - Oversampling function designed to increase usable minority data
% points for the program. The program will generate a confusion matrix to
% demonstrate the oversampling success and program accuracy

function [x_generated, y_generated, new_x, new_y] = smote(x, y, k, randominput)
    
    %identify unique values and split into two groups, Tackle (1) or 
    % Nontackle (0)

    rng(randominput);
    groups = unique(y);
    
    %identifies ranges in numbers that collect either a tackle(1) or
    %nontackle (0). 
    group_counts = histcounts(y, [groups' - .5, max(groups) + 1]);

    % identifies the lowest "bin" to be the minority data and the highest
    % count bin to be the majority data set.
    [minority_count, minority_group_indexs] = min(group_counts);
    majority_count = max(group_counts);
    
    % seperates the non tackle rows from tackle rows
    % enters the minority data into new array based on the indexes
    % collected from the min function above
    minority_data = groups(minority_group_indexs);

    % gives the x values of the original data that correspond to the
    % minority data into the array X_minority
    X_minority = x(y == minority_data,:);
        
    %number of datapoints to generate - to make an even number between
    %majority and minority
    numbertogenerate = majority_count - minority_count;

    generated_data = [];

    %For loop iterates as many times as nessecary to generate enough data
    %to overcome the data point discrepancy between the minority number of
    %data and the majority
    for j = 1:numbertogenerate
        i = randi(min(group_counts));
        distances = vecnorm(X_minority - X_minority(i,:) , 2 , 2);
        [~, neighbor_indexes] = sort(distances);
        k_neighbors = X_minority(neighbor_indexes(2:k+1),:);
    %finds a random neighbor close to the initalized point, and creates a 
    % random factor to multiply it by 
        random_neighbor = k_neighbors(randi(k),:);
        random_factor = rand;
        generated_data_point = X_minority(i,:) + random_factor * (random_neighbor - X_minority(i,:));
    % Generated point then saved under the array Generated Data
        generated_data = [generated_data; generated_data_point];
    end
    %identify the generated X values and create the Y values (the minority
    %data is always the Tackles, so we create the number of cells = 1 that
    %is equal to the generated X data. 
    x_generated = generated_data;
    y_generated = [repmat(minority_data, size(generated_data,1),1)];

    new_x = [x;generated_data];
    new_y = [y; repmat(minority_data, size(generated_data,1),1)];
end

% Function to calculate tackle probability

function tackle_probability = prediction(inputvariables,intercept,coef)
    linear_combo = intercept + inputvariables * coef;
    tackle_probability = 1./(1+exp(-linear_combo));
end



%% Select Needed Data

% Combine all weeks data into one - TrackWeeks
% Saves TrackWeeks with ALL rows and just the desired columns

TrackWeek1 = [TrackWeek1;TrackWeek2;TrackWeek3;TrackWeek4;TrackWeek5;TrackWeek6;TrackWeek7;TrackWeek8;TrackWeek9];
TrackWeeks = TrackWeek1(:,["gameId","playId", "nflId","displayName","frameId","club","x","y","s","event"]);
clear TrackWeek1
clear TrackWeek2
clear TrackWeek3
clear TrackWeek4
clear TrackWeek5
clear TrackWeek6
clear TrackWeek7
clear TrackWeek8
clear TrackWeek9

% Saves with all values when the display name DOES NOT equal football
TrackWeeks = TrackWeeks(TrackWeeks.displayName ~= "football", :);

%% Combine ID Numbers

% Add new Columns for the two new ID's
placeholder = zeros(height(TrackWeeks),1);
TrackWeeks.("gameplayId") = placeholder;
TrackWeeks.("gameplayerId") = placeholder;

placeholder = zeros(height(Tackles),1);
Tackles.('gameplayId') = placeholder;
Tackles.('gameplayerId') = placeholder;

placeholder = zeros(height(plays),1);
plays.('gameplayId') = placeholder;
plays.('gameplayerId') = placeholder;

clear placeholder;

% Concatonates all the strings into the newly made columns (strcat)
TrackWeeks.gameplayId = strcat(TrackWeeks.gameId, '-', TrackWeeks.playId);
TrackWeeks.gameplayerId = strcat(TrackWeeks.gameplayId, '-', TrackWeeks.nflId);

Tackles.gameplayId = strcat(Tackles.gameId, '-', Tackles.playId);
Tackles.gameplayerId = strcat(Tackles.gameplayId,'-',Tackles.nflId);

plays.gameplayId = strcat(plays.gameId,'-',plays.playId);
plays.gameplayerId = strcat(plays.gameplayId,'-', plays.ballCarrierId);


%% Sort and Prepare Data
% Separate into Tackles, Solo Tackles, and Assists

Tackles = Tackles(Tackles.tackle == 1 | Tackles.assist == 1, :);
assists = Tackles(Tackles.assist == 1, :);
solo_tackles = Tackles(Tackles.tackle == 1,:);

% Create Array with Tackle ID's
tackleId = [Tackles.gameplayerId];
assistId = [assists.gameplayerId];
solo_tackleId = [solo_tackles.gameplayerId];

% Add indicator to player on specific play that made tackle
TrackWeeks.("Tackle") = ismember(TrackWeeks.gameplayerId,tackleId);
TrackWeeks.("assist") = ismember(TrackWeeks.gameplayerId, assistId);
TrackWeeks.("solo_tackle") = ismember(TrackWeeks.gameplayerId, solo_tackleId);

% Identify Ballcarrier and add into Track Week table
ballcarrierId = [plays.gameplayerId];
TrackWeeks.("ballcarrier") = ismember(TrackWeeks.gameplayerId, ballcarrierId);

% Create Defense Column to determine if player is a defender
% Create new array to only hold defensive players
% NOTE - already imported long ID's as strings (see import opt functions)

defense_positions = ["CB","FS","SS","DB","MLB","ILB","OLB","DE","DT","NT"];
players.("defense") = ismember(players.position, defense_positions);
defense_players = players(players.defense == true,1:7);

% Array of ID's that identify defensive players - insert into TrackWeek1
defenseIds = [defense_players.nflId];
TrackWeeks.("defense") = ismember(TrackWeeks.nflId, defenseIds);

% Identify and Tag blockers as NON defensive or Ballcarrier personnel
TrackWeeks.("blocker") = (TrackWeeks.defense == 0 & TrackWeeks.ballcarrier == 0);

% Split ballcarriers, defenders, and blockers into seperate arrays
ballcarriers = TrackWeeks(TrackWeeks.ballcarrier == true,:);
defenders = TrackWeeks(TrackWeeks.defense == true, :);
blockers = TrackWeeks(TrackWeeks.blocker == true, :);

% Combine frameId into gameplayId - must be after identifying player roles
ballcarriers.frameId = strcat(ballcarriers.gameplayId, '-', ballcarriers.frameId);
defenders.frameId = strcat(defenders.gameplayId, '-', defenders.frameId);
blockers.frameId = strcat(blockers.gameplayId, '-', blockers.frameId);

% Group the X and Y positions for defenders at each given frame
[G, groupedFrameId] = findgroups(blockers.frameId);
grouped_x_values = splitapply(@(x){x}, blockers.x, G);
grouped_y_values = splitapply(@(x){x}, blockers.y, G);
blockers_x = table(groupedFrameId,grouped_x_values,'VariableNames', {'frameId','blockers_x'});
blockers_y = table(groupedFrameId, grouped_y_values, 'VariableNames', {'frameId', 'blockers_y'});

clear grouped_y_values
clear grouped_x_values
clear groupedFrameId

% rename the x,y, and speed column to identify the Ballcarrier and prepare
% to merge tables
ballcarriers = renamevars(ballcarriers, ["x","y","s"],["x_bc","y_bc","s_bc"]);

% Add blocker's X and Y, and add the Ball Carrier X, Y and Speed to the
% Defender table
defenders = join(defenders,blockers_x, 'Keys', 'frameId');
defenders = join(defenders, blockers_y, "Keys","frameId");
defenders = join(defenders, ballcarriers(:,["frameId","x_bc","y_bc","s_bc"]),"Keys","frameId");

clear ballcarriers
clear blockers_y
clear blockers_x

% filter out columns - keep only nessecary
defenders_filtered = defenders(:,["frameId", "nflId","x","y","s","x_bc","y_bc","s_bc",...
    "Tackle","assist","solo_tackle","event","blockers_x","blockers_y"]);

% use normal distance formula to calculate the "distance to BC" of all
% defenders
defenders_filtered.("dist_to_bc") = sqrt((defenders_filtered.x_bc - defenders_filtered.x).^2 + ...
    (defenders_filtered.y_bc - defenders_filtered.y).^2);


%finds the minimum distance to the bc in the disttobc column in each FRAME*

[G2, uniqueFrameId] = findgroups(defenders_filtered.frameId);
min_dist_values = splitapply(@min, defenders_filtered.dist_to_bc, G2);
min_dist = min_dist_values(G2);

% Saves Minimum distance as new table column and sorts frame ID's
defenders_filtered.("min_dist") = min_dist;
defenders_filtered = sortrows(defenders_filtered, "frameId", "ascend");

% Creates a logical identifyer of player closest to the ballcarier 
defenders_filtered.("is_closest") = double(defenders_filtered.dist_to_bc == defenders_filtered.min_dist);

%initializes variable is blocked
is_blocked_placeholder = false(height(defenders_filtered),1);

%calculates if defender is blocked by calling the IS_BLOCKED premade
%function

for i = 1:height(defenders_filtered)
    defender_coords = [defenders_filtered.x(i),defenders_filtered.y(i)];
    bc_coords = [defenders_filtered.x_bc(i), defenders_filtered.y_bc(i)];
    dist_to_bc = defenders_filtered.dist_to_bc(i);
    x_blocker_list = cell2mat(defenders_filtered.blockers_x(i));
    y_blocker_list = cell2mat(defenders_filtered.blockers_y(i));

    is_blocked_placeholder(i) = is_blocked(defender_coords, bc_coords, dist_to_bc, x_blocker_list, y_blocker_list);
end

defenders_filtered.("is_blocked") = double(is_blocked_placeholder);

% Final filter of data
TacklePercentData = defenders_filtered(:,["frameId","nflId","s","s_bc","Tackle","assist","solo_tackle",...
    "dist_to_bc","is_closest","is_blocked","event"]);

clear defenders_filtered

tackle_data = TacklePercentData(TacklePercentData.event == "tackle",:);
X = tackle_data(:, ["dist_to_bc","s","s_bc","is_blocked","is_closest"]);
Y = tackle_data(:,"Tackle");

% Prepare data for use in certain functions only operable with arrays
X = table2array(X);
Y = table2array(Y);

clear tackle_data

%% Oversample the data using created Program
% Implement SMOTE Oversampling in Matlab through alternative coding - See
% function written above for the oversampling.

[x_generated, y_generated, new_x, new_y] = smote(X,Y, 10, rnginput);

% A check to ensure that the oversampling worked properly by displaying the
% original minority and majority counts and the final counts.

disp('Original Data:');
disp(['Number of Minority Samples: ', num2str(sum(Y == 1))]);
disp(['Number of Majority Samples: ', num2str(sum(Y == 0))]);

disp('Resampled Data:');
disp(['Number of Minority Samples: ', num2str(sum(new_y == 1))]);
disp(['Number of Majority Samples: ', num2str(sum(new_y == 0))]);

%plot the oversampling by the computer
figure('name','SMOTE DATA');
hold on;
scatter(X(Y==1,1), X(Y==1,2), 'b','DisplayName' ,'filled');
scatter(x_generated(y_generated==1,1), x_generated(y_generated==1,2), 'r', 'DisplayName','filled');
legend('Original Minority', 'Generated Minority');
title('SMOTE Resampling');
hold off;

clear x_generated
clear y_generated
clear X
clear Y

%% Linear Regression Model

% Create and Train a linear regression model
rng(rnginput);

% splits the data into two size sets, in this case 80% and 20%
splitdata = cvpartition(size(new_x,1),'HoldOut',.2);
X_train = new_x(training(splitdata),:);
Y_train = new_y(training(splitdata), :);

X_test = new_x(test(splitdata),:);
Y_test = new_y(test(splitdata),:);


% Manual cross validation code
folds = 5;
indices = crossvalind('Kfold',Y_train,folds);

models = cell(folds,1);
accuracy = zeros(folds,1);

% tests 5 folds of the model and compares their individual accuracy
% then provides the average accuracy of all models at the end.
for i = 1:folds
    test_idx = (indices == i);
    train_idx = ~test_idx;
    X_train_crossV = X_train(train_idx,:);
    Y_train_crossV = Y_train(train_idx,:);

    X_validate = X_train(test_idx,:);
    Y_validate = Y_train(test_idx);

    models{i} = fitclinear(X_train_crossV, Y_train_crossV, 'Learner','logistic');

    y_pred_cv = predict(models{i}, X_validate);
    accuracy(i) = mean(y_pred_cv == Y_validate);
end

clear folds

fprintf('\n');
disp('Cross-Validation Accuracy: ')
disp(accuracy);
fprintf('\n');
fprintf(['Mean Accuracy:  ', num2str(mean(accuracy)*100),'%%\n']);

% Uses all the data to create a final linear regression model for our use
% in the tackle probability function

final_model = fitclinear(X_train, Y_train, 'Learner', 'logistic');
y_testing_prediction = predict(final_model, X_test);

clear X_test
clear X_train
clear Y_train

% Creates confustion matrix to show program's success at identifying a
% Tackle or Non Tackle
conf_mat = confusionmat(Y_test, y_testing_prediction);

clear Y_test
clear y_testing_prediction

figure('Name','Matrix of Prediction Success');
conf_chart = confusionchart(conf_mat, {'No Tackle', 'Tackle'});
conf_chart.Title = 'Matrix of Regression Model Success';
conf_chart.RowSummary = 'row-normalized';
conf_chart.ColumnSummary = 'column-normalized';

conf_chart.XLabel = 'Predicted Outcome';
conf_chart.YLabel = 'Actual Outcome';

disp(conf_chart);

% Identifies components (coefficients and the intercepts) of the final
% model to be used in the prediction function 

intercept = final_model.Bias;
coefficients = final_model.Beta;

% Presets the input variables for the tackle probability function (AT START)
input_variables = [TacklePercentData.dist_to_bc, TacklePercentData.s, TacklePercentData.s_bc, TacklePercentData.is_blocked, TacklePercentData.is_closest];

%Uses prediction function to create Tackle Probability column in the table
TacklePercentData.("TackleProbability") = prediction(input_variables, intercept, coefficients); % Full Final Data Set


%% Tackles Above Average For All Plays

% Filters for the instances of a Pass, Handoff, or Run
total_TAA_data = TacklePercentData(TacklePercentData.event == "pass_outcome_caught" | ...
    TacklePercentData.event == "handoff" | TacklePercentData.event == "run",:);

% Reclassifies column variable types in the table
total_TAA_data.Properties.VariableTypes = ["string","string","double","double"...
    ,"logical","double","logical", "double","double","double","cell","double"];

% Indentifies the assist tackle as the multiplecation factor of the tackle
% column, either an assist counts as half of the expected tackle or
% multiply the null.
total_TAA_data.assist(total_TAA_data.assist == 1) = 0.5;
total_TAA_data.assist(total_TAA_data.assist ~= 0.5) = 1;

%tackle probability over expected equation
total_TAA_data.("Tackle_Prob_Over_Expected") = total_TAA_data.assist.*(total_TAA_data.Tackle - total_TAA_data.TackleProbability);

% groups the tackle probability by NFLid (players) and finds their total
% over expected tackle probability across the data
player_sum = groupsummary(total_TAA_data,"nflId",'sum',"Tackle_Prob_Over_Expected");

%joins the new players Tackle over expected data to the Total TAA data
total_TAA_data = outerjoin(total_TAA_data, player_sum(:,{'nflId','sum_Tackle_Prob_Over_Expected'}), 'Keys', 'nflId','MergeKeys',true);
total_TAA_data = total_TAA_data(:,["nflId", "sum_Tackle_Prob_Over_Expected"]);
total_TAA_data.Properties.VariableNames{'sum_Tackle_Prob_Over_Expected'} = 'Tackles Over Expected'; %Renames

% Collects and combines the tackles above expected into one row per player
total_TAA_data = groupsummary(total_TAA_data,["nflId","Tackles Over Expected"],"none");
total_TAA_data = total_TAA_data(:,["nflId","Tackles Over Expected"]);
total_TAA_data = outerjoin(total_TAA_data, players(:,{'nflId','displayName','position'}), 'Keys', 'nflId','MergeKeys',true);

%finds the position mean of tackes over expected and then finds each
%player's tackles above average from that
position_mean = groupsummary(total_TAA_data,"position","mean","Tackles Over Expected");
total_TAA_data = outerjoin(total_TAA_data,position_mean(:,{'position','mean_Tackles Over Expected'}),'Keys','position','MergeKeys',true);
total_TAA_data.("Tackles Above Average") = total_TAA_data.("Tackles Over Expected") - total_TAA_data.("mean_Tackles Over Expected");


defense_position_TAA_data = [];

for i = 1:length(defense_positions)
    position = defense_positions(i);
    
    sorted_defense_data = total_TAA_data(strcmp(total_TAA_data.position,position),:);
    sorted_defense_data = sortrows(sorted_defense_data(:,{'position','displayName','Tackles Above Average'}),'Tackles Above Average', 'descend');

    defense_position_TAA_data = [defense_position_TAA_data; sorted_defense_data]; 
end

%clean up initialized space
clear sorted_defense_data
clear position
clear i

% removes players who didnot make a play or data is unusable
total_TAA_data = defense_position_TAA_data(~isnan(defense_position_TAA_data.("Tackles Above Average")),:);

clear defense_position_TAA_data

%% Tackles Above Average, Breakdown for Run and Pass Plays

%uses the same process as the Total data, just split into run and pass
%options by identifying Handoff and Run as Run plays

% **See notes above explaining each line - same process

run_play_data = TacklePercentData(TacklePercentData.event == "handoff" | TacklePercentData.event == "run",:);
pass_play_data = TacklePercentData(TacklePercentData.event == "pass_outcome_caught",:);

clear TacklePercentData

run_play_data.assist(run_play_data.assist == 1) = 0.5;
run_play_data.assist(run_play_data.assist ~= 0.5) = 1;

pass_play_data.assist(pass_play_data.assist == 1) = 0.5;
pass_play_data.assist(pass_play_data.assist ~= 0.5) = 1;

run_play_data.("Tackle_Prob_Over_Expected") = run_play_data.assist.*(run_play_data.Tackle - run_play_data.TackleProbability);
pass_play_data.("Tackle_Prob_Over_Expected") = pass_play_data.assist.*(pass_play_data.Tackle - pass_play_data.TackleProbability);

% Run Play Manipulation

player_sum = groupsummary(run_play_data,"nflId",'sum',"Tackle_Prob_Over_Expected");
run_play_data = outerjoin(run_play_data, player_sum(:,{'nflId','sum_Tackle_Prob_Over_Expected'}), 'Keys', 'nflId','MergeKeys',true);
run_play_data = run_play_data(:,["nflId", "sum_Tackle_Prob_Over_Expected"]);
run_play_data.Properties.VariableNames{'sum_Tackle_Prob_Over_Expected'} = 'Tackles Over Expected (RUN)';

run_play_data = groupsummary(run_play_data,["nflId","Tackles Over Expected (RUN)"],"none");
run_play_data = run_play_data(:,["nflId","Tackles Over Expected (RUN)"]);
run_play_data = outerjoin(run_play_data, players(:,{'nflId','displayName','position'}), 'Keys', 'nflId','MergeKeys',true);

position_mean_run = groupsummary(run_play_data,"position","mean","Tackles Over Expected (RUN)");
run_play_data = outerjoin(run_play_data,position_mean_run(:,{'position','mean_Tackles Over Expected (RUN)'}),'Keys','position','MergeKeys',true);
run_play_data.("Tackles Above Average (RUN)") = run_play_data.("Tackles Over Expected (RUN)") - run_play_data.("mean_Tackles Over Expected (RUN)");

run_TAA_data = [];

for i = 1:length(defense_positions)
    position = defense_positions(i);
    
    sorted_defense_data_run = run_play_data(strcmp(run_play_data.position,position),:);
    sorted_defense_data_run = sortrows(sorted_defense_data_run(:,{'position','displayName','nflId','Tackles Over Expected (RUN)','Tackles Above Average (RUN)'}),'Tackles Above Average (RUN)', 'descend');

    run_TAA_data = [run_TAA_data; sorted_defense_data_run]; 
end
clear sorted_defense_data_run
clear position_mean_run
clear position
clear i

run_TAA_data = run_TAA_data(~isnan(run_TAA_data.("Tackles Above Average (RUN)")),:);

% Pass Play Manipulation

player_sum = groupsummary(pass_play_data,"nflId",'sum',"Tackle_Prob_Over_Expected");
pass_play_data = outerjoin(pass_play_data, player_sum(:,{'nflId','sum_Tackle_Prob_Over_Expected'}), 'Keys', 'nflId','MergeKeys',true);
pass_play_data = pass_play_data(:,["nflId", "sum_Tackle_Prob_Over_Expected"]);
pass_play_data.Properties.VariableNames{'sum_Tackle_Prob_Over_Expected'} = 'Tackles Over Expected (PASS)';

pass_play_data = groupsummary(pass_play_data,["nflId","Tackles Over Expected (PASS)"],"none");
pass_play_data = pass_play_data(:,["nflId","Tackles Over Expected (PASS)"]);
pass_play_data = outerjoin(pass_play_data, players(:,{'nflId','displayName','position'}), 'Keys', 'nflId','MergeKeys',true);

position_mean_pass = groupsummary(pass_play_data,"position","mean","Tackles Over Expected (PASS)");
pass_play_data = outerjoin(pass_play_data,position_mean_pass(:,{'position','mean_Tackles Over Expected (PASS)'}),'Keys','position','MergeKeys',true);
pass_play_data.("Tackles Above Average (PASS)") = pass_play_data.("Tackles Over Expected (PASS)") - pass_play_data.("mean_Tackles Over Expected (PASS)");

pass_TAA_data = [];

for i = 1:length(defense_positions)
    position = defense_positions(i);
    
    sorted_defense_data_pass = pass_play_data(strcmp(pass_play_data.position,position),:);
    sorted_defense_data_pass = sortrows(sorted_defense_data_pass(:,{'position','displayName','nflId','Tackles Over Expected (PASS)','Tackles Above Average (PASS)'}),'Tackles Above Average (PASS)', 'descend');

    pass_TAA_data = [pass_TAA_data; sorted_defense_data_pass]; 
end
clear sorted_defense_data_pass
clear position_mean_run
clear position
clear i

pass_TAA_data = pass_TAA_data(~isnan(pass_TAA_data.("Tackles Above Average (PASS)")),:);

%% Merge Run and Pass Plays into RUN AND PASS PLAYS table

run_and_pass_plays = run_TAA_data;

% Join the pass plays matrix and select specific needed columns, join with
% the NFLiD as the key and combine.
run_and_pass_plays = outerjoin(run_and_pass_plays, ...
    pass_TAA_data(:, {'nflId', 'Tackles Over Expected (PASS)', 'Tackles Above Average (PASS)'}),...
    'Keys', 'nflId', 'MergeKeys', true);

%identify proper variable types for the manipulation
run_and_pass_plays.Properties.VariableTypes = ["string", "string", "double", "double", "double", "double", "double"];

%% Generate Positional Average Tackle Data
 
position_mean = position_mean(ismember(position_mean.position, defense_positions),:);
position_mean = position_mean(:,["position", "mean_Tackles Over Expected"]);
position_mean.Properties.VariableNames = ["Position","Average Tackles Over Expected"];

%% PLOTS

% line graph
figure('Name','Tackle Probability Over Expected (Pass vs Run)', 'Position', [100,100,1500,800]);
plot(run_and_pass_plays.nflId, run_and_pass_plays.("Tackles Over Expected (RUN)"), 'Color', 'r','LineStyle','-','DisplayName','RUN TOE');
hold on;
plot(run_and_pass_plays.nflId, run_and_pass_plays.("Tackles Over Expected (PASS)"), 'Color', 'b', 'LineStyle','-','DisplayName','PASS TOE');
hold off;

xlabel('nflId');
ylabel('Tackle Probability Over Expected');
title('Tackle Probability Over Expected (Run vs Pass');

legend show
grid on

% Scatter Plot
figure('Name', 'Tackle Probability Over Expected Scatter Plot','Position',[0,0,1000,1000]);
scatter(run_and_pass_plays.("Tackles Over Expected (RUN)"),run_and_pass_plays.("Tackles Over Expected (PASS)"),'b','filled');
xlabel('Tackle Probability Over Expected (RUN)');
ylabel('Tackle Probability Over Expected (PASS)');


% Position Tackle Bar Graph
figure('Name','BarofPosition Tackles Over Expected');
bar(position_mean.Position, position_mean.("Average Tackles Over Expected"));
xlabel('Position');
ylabel('Average Tackles Over Expected');
title('Positions Average Tackles Over Expected')

% display table with most successful tacklers by positions and overall?
defense_position_TAA_data = [];
defense_positions = cellstr(defense_positions);
for i = 1:length(defense_positions)
    defense_position_TAA_data = total_TAA_data(strcmp(total_TAA_data.position,defense_positions{i}),:);
    defense_position_TAA_data = sortrows(defense_position_TAA_data(:,{'displayName','Tackles Above Average'}),...
        "Tackles Above Average", "descend");
    top10 = defense_position_TAA_data(1:min(10,height(defense_position_TAA_data)),:);
    disp(['Top 10 for position:  ', defense_positions{i}]);
    disp(top10);
    clear top10
end




%% Scrapped SMOTE function, did not operate efficiently or as intended

%{

function [x_generated, y_generated] = smote(x, y, k, random_seeding)
    

    rng(random_seeding);
    classes = unique(y);
    class_counts = histcounts(y, [classes' - .5, max(classes) + 1]);

    [minority_class_count, minority_class_index] = min(class_counts);
    majority_class_count = max(class_counts);
    
    % seperates the non tackle rows from tackle rows
    minority_class = classes(minority_class_index);

    X_minority = x(y == minority_class,:);
        
    numbertogenerate = majority_class_count - minority_class_count;

    generated_data = [];
    for i = 1:minority_class_count
        distances = vecnorm(X_minority - X_minority(i,:), 2, 2);
        [~, neighbor_indices] = sort(distances);
        k_neighbors = X_minority(neighbor_indices(2:k+1),:);

        for j = 1:numbertogenerate
            random_factor = rand;
            generated_data_point = X_minority(i,:) + random_factor*(k_neighbors(randi(k),:) - X_minority(i,:));
            generated_data = [generated_data; generated_data_point];
        end
    end

    x_generated = [x; generated_data];
    y_generated = [y; repmat(minority_class, size(generated_data, 1),1)];
end
%}

%% Scrapped Tackle Probability Function: Too inefficient

%{
function tackle_probability = predict_prob(dist_to_bc,s,s_bc,is_blocked,is_closest,intercept,coef)
    tackle_probability = 1/(1+exp(-1*(intercept+(coef(1)*dist_to_bc)+ ...
        (coef(2)*s)+(coef(3)*s_bc)+(coef(4)*is_blocked)+(coef(5)*is_closest))));
end

for i = 1:height(final_data)
    final_data.Tackle_Probability(i) = predict_prob(final_data.dist_to_bc(i),final_data.s(i), ...
        final_data.s_bc(i),final_data.is_blocked(i),final_data.is_closest(i), intercept, coefficient);
end
%}