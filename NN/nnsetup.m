function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.maxNorm                          = 0;            %  max-norm regularization (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.adaptive_learningRate            = 0;            %  if non-zero then use adaptive learning rate
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.epochFunction                    = 'none';       %  Choose a function and drop it in.
    nn.rmsprop                          = 0;            %  Rough implementation of RMSPROP
    nn.rmsprop_adaptRate                = 0.01;         %  Speed at which RMPSPROP uses most-recent gradients

    nn.report_numepochs                 = 1;            % Number of epochs between reports.

    for i = 2 : nn.n   
        wsigma = 0.5 * sqrt ( 6 / (nn.size(i) + nn.size(i - 1) ));
        nn.W{i - 1} = wsigma * randn( nn.size(i), nn.size(i - 1)+1 );
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % For RMSPROP
        nn.rmsdW{i - 1} = ones( size( nn.W{i - 1}) );

        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
