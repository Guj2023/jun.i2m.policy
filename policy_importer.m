function [obsInfo, actInfo, policy, actor] = policy_importer(model_dir)
    [obsInfo, actInfo, actor, net, layout] = create_actor(model_dir + "/normalizer.json", model_dir + "/policy.onnx");
    policy = rlDeterministicActorPolicy(actor);
    generatePolicyBlock(policy, MATFileName= model_dir + "/policy.mat");
end


function [obsInfo, actInfo, actor, net, layout] = create_actor(jsonPath, onnxPath, actLow, actHigh)
% Build RL actor from ONNX by scraping actor_* params from Layers(2,1).
% - JSON normalizer: keys "_mean", "_std" or "_var", nested [[...]] supported.
% - ONNX: parameters live under isaac_sim_agent.Layers(2,1).actor_<idx>_(weight|bias)

    if nargin < 3 || isempty(actLow),  actLow  = -130; end
    if nargin < 4 || isempty(actHigh), actHigh =  130; end

    % ---- 1) Normalizer ----
    [mu, sigma] = loadNormalizer(jsonPath);  % 1 x D
    obsDim = numel(mu);
    sigmaSafe = sigma; 
    sigmaSafe(sigmaSafe < 1e-6) = 0.01;  % pass-through scale

    % ---- 2) Import ONNX and scrape Layers(2,1) ----
    isaac_sim_agent = importNetworkFromONNX(onnxPath);  % DAGNetwork
    L2 = isaac_sim_agent.Layers(2,1);                   % your known container

    FC = parseActorParamsFromLayer2(L2);                % struct array: idx, W, b
    if isempty(FC)
        error("No fields 'actor_<idx>_(weight|bias)' found in Layers(2,1).");
    end

    % Sort by idx (0,2,4,6,... typical)
    [~, order] = sort([FC.idx]);
    FC = FC(order);

    % ---- 3) Coerce orientations and infer sizes ----
    inSizes  = zeros(numel(FC),1);
    outSizes = zeros(numel(FC),1);
    for i = 1:numel(FC)
        [FC(i).W, FC(i).b, outSizes(i), inSizes(i)] = coerceFC(FC(i).W, FC(i).b);
    end

    % Sanity: first FC input size must match obsDim
    if inSizes(1) ~= obsDim
        error("First FC input size (%d) != JSON obsDim (%d).", inSizes(1), obsDim);
    end

    actDim = outSizes(end);
    [actLowV, actHighV] = expandBounds(actLow, actHigh, actDim);

    % ---- 4) Specs ----
    obsInfo = rlNumericSpec([obsDim 1], ...
        'LowerLimit', -inf(obsDim,1), ...
        'UpperLimit',  inf(obsDim,1));
    obsInfo.Name = "observations";

    actInfo = rlNumericSpec([actDim 1], ...
        'LowerLimit', actLowV, ...
        'UpperLimit', actHighV);
    actInfo.Name = "action";

    % ---- 5) Build network: featureInput(zscore) + FC[*] (+ ReLU except last) ----
    layers = [
        featureInputLayer(obsDim, 'Normalization','zscore', ...
            'Mean', double(mu), 'StandardDeviation', double(sigmaSafe), ...
            'Name','state')
    ];

    % Debug: pass zeros through featureInputLayer normalization
    testInput = zeros(obsDim,1);
    normInput = (testInput(:) - double(mu(:))) ./ double(sigmaSafe(:));
    disp('featureInputLayer zeros input normalized:');
    disp(normInput);
    
    for i = 1:numel(FC)
        layers = [layers
            fullyConnectedLayer(outSizes(i), 'Name', sprintf('fc%d',i))]; %#ok<AGROW>
        if i < numel(FC)
            layers = [layers
                reluLayer('Name', sprintf('relu%d',i))]; %#ok<AGROW>
        end
    end

    net = dlnetwork(layers);

    % ---- 6) Load weights/biases into net ----
    NL = net.Learnables;
    for i = 1:numel(FC)
        li = sprintf('fc%d',i);
        NL.Value(strcmp(NL.Layer, li) & strcmpi(NL.Parameter,'Weights')) = {dlarray(single(FC(i).W))};
        NL.Value(strcmp(NL.Layer, li) & strcmpi(NL.Parameter,'Bias'))    = {dlarray(single(FC(i).b))};
    end
    net.Learnables = NL;

    % ---- 7) Wrap actor ----
    actor = rlContinuousDeterministicActor(net, obsInfo, actInfo);

    % ---- 8) Debug layout ----
    layout = struct('indices',[FC.idx], 'inSizes',inSizes, 'outSizes',outSizes, ...
                    'obsDim',obsDim, 'actDim',actDim);
end


%% ================= Helpers =================

function FC = parseActorParamsFromLayer2(L2)
% Parse fields in Layers(2,1) of the imported DAGNetwork that match:
%   actor_<idx>_weight, actor_<idx>_bias
% Return struct array with fields: idx, W, b

    fns = string(fieldnames(L2));
    tmp = struct('idx',{},'W',{},'b',{});

    for k = 1:numel(fns)
        fn = fns(k);
        m = regexp(fn, '^actor_(\d+)_(weight|bias)$', 'tokens', 'once');
        if isempty(m), continue; end
        idx = str2double(m{1});
        kind = m{2};  % 'weight' or 'bias'
        val = L2.(fn);

        % find or create entry
        pos = find([tmp.idx]==idx, 1);
        if isempty(pos)
            tmp(end+1).idx = idx; tmp(end).W = []; tmp(end).b = []; %#ok<AGROW>
            pos = numel(tmp);
        end
        if strcmp(kind,'weight')
            tmp(pos).W = val;
        else
            tmp(pos).b = val;
        end
    end

    % keep only complete FCs
    FC = tmp(arrayfun(@(e) ~isempty(e.W) && ~isempty(e.b), tmp));
end


function [Wout, bout, outSize, inSize] = coerceFC(W, b)
% Ensure weights are [out x in] and bias is [out x 1]
    if isa(W,'dlarray'), W = extractdata(W); end
    if isa(b,'dlarray'), b = extractdata(b); end
    b = b(:);
    outLen = numel(b);

    % ONNX via your container usually stores W as [in x out]; transpose.
    W = reshape(W, size(W,1), []);   % squash trailing dims if any
    s = size(W);
    if s(2) == outLen
        Wout = W.'; bout = b; outSize = s(2); inSize = s(1);    % typical case
    elseif s(1) == outLen
        Wout = W;  bout = b; outSize = s(1); inSize = s(2);
    else
        % choose orientation closest to bias length
        if abs(s(1)-outLen) <= abs(s(2)-outLen)
            Wout = W;  bout = b;  outSize = s(1); inSize = s(2);
        else
            Wout = W.'; bout = b; outSize = s(2); inSize = s(1);
        end
    end
end


function [lo, hi] = expandBounds(loIn, hiIn, d)
% Expand scalar to vector length d, or validate provided vectors.
    if isscalar(loIn), lo = repmat(loIn, d, 1); else, lo = loIn(:); end
    if isscalar(hiIn), hi = repmat(hiIn, d, 1); else, hi = hiIn(:); end
    if numel(lo) ~= d || numel(hi) ~= d
        error("actLow/actHigh must be scalar or length == %d.", d);
    end
end


function [mu, sigma] = loadNormalizer(jsonPath)
% Read JSON normalizer: supports _mean + (_std or _var). Handles [[...]].
    jn = jsondecode(fileread(jsonPath));
    mu = pickFirstRow(jn,"x_mean");
    if isfield(jn,"x_std")
        sigma = pickFirstRow(jn,"x_std");
    elseif isfield(jn,"x_var")
        v = pickFirstRow(jn,"x_var"); v(v<0)=0; sigma = sqrt(v);
    else
        error("JSON missing '_std' or '_var'.");
    end
    mu    = double(mu(:)).';
    sigma = double(sigma(:)).';
end


function row = pickFirstRow(jn, key)
% Handles [[...]] and [...] numeric payloads.
    if ~isfield(jn,key), error("Field '%s' not found in JSON.", key); end
    val = jn.(key);
    if iscell(val)
        v = val{1}; if iscell(v), v = v{1}; end
        if ~isnumeric(v), error("JSON '%s' is not numeric.", key); end
        row = reshape(v,1,[]);
    elseif isnumeric(val)
        if ismatrix(val) && size(val,1) > 1
            row = val(1,:);
        else
            row = reshape(val,1,[]);
        end
    else
        error("Unsupported JSON field '%s' format.", key);
    end
end