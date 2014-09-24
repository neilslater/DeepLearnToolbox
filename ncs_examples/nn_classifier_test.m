clear ; close all; clc; more off

addpath(genpath('DeepLearnToolbox'));

fprintf( "Loading data  . . .\n\n");
datas = csvread( 'YearPredictionMSD.txt' );

fprintf( "Building training, validation and test sets  . . .\n\n");

y = datas(:,1);
minYear = min(y); maxYear = max(y);
numLabels = maxYear - minYear + 1;
y = ( y - minYear + 10 ) ./ ( numLabels + 20 ); % better than mapping close to 1.0 or 0.0

train_y = y(1:463715,:); train_x = datas(1:463715,2:end);
test_y = y(463716:end,:); test_x = datas(463716:end,2:end);
rp = randperm(size(train_y,1));
train_y = train_y(rp,:);
train_x = train_x(rp,:);

cross_y = train_y(1:2500,:);
cross_x = train_x(1:2500,:);
train_y = train_y(2501:end,:);
train_x = train_x(2501:end,:);

% This drops 15 random training cases in aid of larger mini-batch size
train_x = train_x(1:461200,:);
train_y = train_y(1:461200,:);

mu = mean( train_x );
sigma = std( train_x );
for i=1:size(train_x, 2),
  train_x(:,i) = ( train_x(:,i) - mu(i) ) / sigma(i);
  cross_x(:,i) = ( cross_x(:,i) - mu(i) ) / sigma(i);
  test_x(:,i) = ( test_x(:,i) - mu(i) ) / sigma(i);
end;

fprintf( "Training neural network . . .");

opts.numepochs =   15;
opts.batchsize =   100;

% Setup 2: accuracy = 6.15 (was 90 360 120 and 0.1 multiplier on init weights)
nnr = nnsetup( [90 600 200 1] );
 % May need to adjust weights down for this problem . . .
for i = 2 : nnr.n,
    nnr.W{i-1} = nnr.W{i-1} * 1.0;
end;

nnr.activation_function    = 'tanh_opt';
nnr.output                 = 'sigm';

for i = 1 : 10,
    fprintf( "\n\nTraining stage %d. . .\n\n", i );
    nnr.dropoutFraction        = 0.5;
    nnr.momentum               = 0.9;
    nnr.learningRate           = 0.5;
    nnr.scaling_learningRate   = 0.8;

    % Re-randomise training set and re-train
    rp = randperm(size(train_y,1));
    train_y = train_y(rp,:);
    train_x = train_x(rp,:);

    nnr = nntrain( nnr, train_x, train_y, opts, cross_x, cross_y );

    fprintf( "\nMaking test predictions . . .\n\n");
    test_predictions = round( nnregression( nnr, test_x ) * ( numLabels + 20 ) - 10 + minYear );
    test_actual = round( test_y * ( numLabels + 20 ) - 10 + minYear );
    correct = test_predictions == test_actual;
    accuracy = mean( correct );
    fprintf( "Accuracy = %0.2f %%\n\n", accuracy * 100 );
end;

fprintf( "Saving network as nnr_01.dat\n\n");
save -binary nnr_01.dat nnr
