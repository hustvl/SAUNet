clear;clc;
res_path = 'results/Test_result.mat';
load(res_path);


niqe_pred = zeros(5,1);
for img = 1 : 5
    pred_ = squeeze(pred(img,:,:,:));
    
    for ch1 = 1:28
        niqe_ = niqe(pred_(:,:,ch1));
        niqe_pred(img) = niqe_pred(img) + niqe_;
    end 
    niqe_pred(img) = niqe_pred(img) / 28;
end

niqe_pred
fprintf('The NIQE=%f\n',mean(niqe_pred));