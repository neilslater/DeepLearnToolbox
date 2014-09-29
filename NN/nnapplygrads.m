function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end

    % max-norm regularisation of neuron input weights
    if (nn.maxNorm > 0)
        for i = 1 : (nn.n - 1)
            norms = sqrt( sum( (nn.W{i}(:,2:end) .* nn.W{i}(:,2:end) ),2) );
            adjust = norms > nn.maxNorm;
            nn.W{i}( adjust, 2:end ) = bsxfun( @rdivide, ( nn.W{i}( adjust, 2:end ) * nn.maxNorm ), norms( adjust ) );
        end
    end

    % Needs verification. Is this a valid max-norm regularisation?
    % This ref: http://www.cs.toronto.edu/~nitish/msc_thesis.pdf suggest per-neuron max-norm
    % whilst the setting here is for the whole network.
    if(nn.globalMaxNorm > 0),
        sum_sq = 0;
        for i = 1 : (nn.n - 1),
            sum_sq += sum( sum(  nn.W{i}(:,2:end) .* nn.W{i}(:,2:end)  ) );
        end;
        norm = sqrt( sum_sq );
        if norm > nn.globalMaxNorm,
            ratio = nn.globalMaxNorm / norm;
            for i = 1 : (nn.n - 1),
                nn.W{i}(:,2:end) = nn.W{i}(:,2:end) * ratio;
            end;
        end;
    end;
end
