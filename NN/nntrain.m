function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number of input arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end
% Limit attempts to feed-forward giant data sets just to get "super-accurate" error values
train_loss_size = min( size(train_x,1), 2000 );

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end

    if i == 1
       report_stats = zeros( nn.report_numepochs, 4 ); % time, mb train, train, val
    end
    stat_id = mod(i,nn.report_numepochs);
    do_report = 0;
    if stat_id == 0
        stat_id = nn.report_numepochs;
        do_report = 1;
    end;
    report_stats( stat_id, 2 ) = mean(L((n-numbatches):(n-1)));

    if opts.validation == 1
        loss = nneval(nn, loss, train_x(1:train_loss_size,:), train_y(1:train_loss_size,:), val_x, val_y);
        report_stats( stat_id, 3 ) = loss.train.e(end);
        report_stats( stat_id, 4 ) = loss.val.e(end);
        t = toc;
        report_stats( stat_id, 1 ) = t;
        if do_report
            if stat_id == 1
                str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', report_stats(1,3), report_stats(1,4));
            else
                rs_mu = mean( report_stats );
                rs_sd = std( report_stats );
                str_perf = sprintf('; TrainE %f +-%f; ValE %f +-%f (min %f)', rs_mu(3), rs_sd(3), rs_mu(4), rs_sd(4), min( report_stats(:,4) ) );
            end
        end
    else
        loss = nneval(nn, loss, train_x(1:train_loss_size,:), train_y(1:train_loss_size,:));
        report_stats( stat_id, 3 ) = loss.train.e(end);
        t = toc;
        report_stats( stat_id, 1 ) = t;
        if do_report
            if stat_id == 1
                str_perf = sprintf('; Full-batch train err = %f', report_stats(1,3));
            else
                rs_mu = mean( report_stats );
                rs_sd = std( report_stats );
                str_perf = sprintf('; TrainE %f +-%f', rs_mu(3), rs_sd(3) );
            end
        end
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end

    if do_report
        if stat_id == 1
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch train mse ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
        else
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. ' sprintf('%0.2f',rs_mu(1)) ' s/epoch' '.  MBTrainE ' sprintf('%0.3f',rs_mu(2)) str_perf]);
        end
    end

    switch nn.epochFunction
        case 'none'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
        case 'scale_dropout_01'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
            nn.dropoutFraction = 0.4 - 0.2 * (i/numepochs);
        case 'scale_max_norm_01'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
            nn.dropoutFraction = 0.2 - 0.1 * (i/numepochs);
            nn.maxNorm = 1.0 - 0.35 * (i/numepochs);
        case 'scale_max_norm_02'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
            nn.maxNorm = 0.65 - 0.1 * (i/numepochs);
        case 'scale_max_norm_03'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
            nn.maxNorm = 0.55 - 0.05 * (i/numepochs);
        case 'v19_stage02'
            nn.dropoutFraction = 0.4 - 0.2 * (i/numepochs);
        case 'v19_stage03'
            nn.dropoutFraction = 0.2 - 0.1 * (i/numepochs);
            nn.maxNorm = 1.0 - 0.35 * (i/numepochs);
        case 'v19_stage04'
            nn.maxNorm = 0.65 - 0.1 * (i/numepochs);
        case 'v19_stage05'
            nn.maxNorm = 0.55 - 0.05 * (i/numepochs);
        case 'test'
            nn.learningRate = nn.learningRate * nn.scaling_learningRate;
            nn.dropoutFraction = 0.1 + 0.1 * (i/numepochs);
    end;
end
end
