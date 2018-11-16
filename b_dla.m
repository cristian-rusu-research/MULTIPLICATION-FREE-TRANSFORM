function [U, X, positions, values, tus, err] = b_dla(Data, k0, m, NN)

tic;

NN = min(15, NN);

Gs = zeros(15,2,2);
Gs(1,:,:) = 1/sqrt(2)*[-1 1; 1 1]; Gs(2,:,:) = 1/sqrt(2)*[1 1; -1 1];
Gs(3,:,:) = 1/sqrt(2)*[1 -1; 1 1]; Gs(4,:,:) = 1/sqrt(2)*[1 1; 1 -1];
Gs(5,:,:) = -1/sqrt(2)*[-1 1; 1 1]; Gs(6,:,:) = -1/sqrt(2)*[1 1; -1 1];
Gs(7,:,:) = -1/sqrt(2)*[1 -1; 1 1]; Gs(8,:,:) = -1/sqrt(2)*[1 1; 1 -1];
Gs(9,:,:) = [0 1; -1 0]; Gs(10,:,:) = [0 -1; 1 0];
Gs(11,:,:) = [1 0; 0 -1]; Gs(12,:,:) = [-1 0; 0 1];
Gs(13,:,:) = [0 -1; -1 0]; Gs(14,:,:) = [0 1; 1 0];
Gs(15,:,:) = [-1 0; 0 -1];

[n, ~] = size(Data);
[U, ~, ~] = svd(Data, 'econ');
X = omp_forortho(U'*Data, k0);

positions = zeros(2, m);
values = zeros(4, m);

% number of iterations
K = 10;

% the initialization
Z = Data*X'; trZ = trace(Z);
scores_nuclear = zeros(15,n,n);
err = [];
for i = 1:n
    for j = i+1:n
        scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
        scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
        scores_nuclear(3, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
        scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
        scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
        scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
        scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
        scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
        scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
        scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
        scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
        scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
        scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%         scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
        scores_nuclear(14, i, j) = inf;
        scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
    end
end

for kk = 1:m
    vals = zeros(NN, 1);
    inds = zeros(NN,1);
    for kkk = 1:NN
        [val, index_nuc] = min(vec(squeeze(scores_nuclear(kkk,:,:))));
        vals(kkk) = val;
        inds(kkk) = index_nuc;
    end
    [min_val, min_ind] = min(vals);
    
    index_nuc = inds(min_ind);
    GG = squeeze(Gs(min_ind, :, :));

    [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    values(:, kk) = vec(GG);
    
    Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, values(:, kk)); trZ = trace(Z);

    for i = [i_nuc j_nuc]
        for j = i+1:n
            scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
            scores_nuclear(3, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
            scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
            scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
            scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
            scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
            scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
            scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
            scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%             scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
            scores_nuclear(14, i, j) = inf;
            scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
        end
    end

    for j = [i_nuc j_nuc]
        for i = 1:j-1
            scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
            scores_nuclear(3, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
            scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
            scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
            scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
            scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
            scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
            scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
            scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
            scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%             scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
            scores_nuclear(14, i, j) = inf;
            scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
        end
    end
    
%     err = [err norm(Data - workingX, 'fro')^2/norm(Data, 'fro')^2*100];
end

P = Data;
for h = m:-1:1
    P = applyGTransformOnLeftTransp(P, positions(1, h), positions(2, h), values(:, h));
end
X = omp_forortho(P, k0);

err = zeros(K, 1);
for k = 1:K
    the_Data = Data;
    for h = m:-1:1
        the_Data = applyGTransformOnLeftTransp(the_Data, positions(1, h), positions(2, h), values(:, h));
    end
    
    Z = the_Data*X';
    
    for kk = 1:m

        Z = applyGTransformOnLeft(Z, positions(1, kk), positions(2, kk), values(:, kk)); trZ = trace(Z);
        if (kk == 1)
            scores_nuclear = zeros(15,n,n);
            for i = 1:n
                for j = i+1:n
                    scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
                    scores_nuclear(3, i, j) = -2*trZ +  (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
                    scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
                    scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
                    scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
                    scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
                    scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%                     scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
                    scores_nuclear(14, i, j) = inf;
                    scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
                end
            end
        else
            for i = [i_nuc j_nuc positions(1, kk) positions(2, kk)]
                for j = i+1:n
                    scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
                    scores_nuclear(3, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
                    scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
                    scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
                    scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
                    scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
                    scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%                     scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
                    scores_nuclear(14, i, j) = inf;
                    scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
                end
            end
            
            for j = [i_nuc j_nuc positions(1, kk) positions(2, kk)]
                for i = 1:j-1
                    scores_nuclear(1, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(2, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) - sqrt(2)*Z(i,j) + sqrt(2)*Z(j,i);
                    scores_nuclear(3, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2-sqrt(2)) + sqrt(2)*Z(i,j) - sqrt(2)*Z(j,i);
                    scores_nuclear(4, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) - sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(5, i, j) = -2*trZ + Z(i,i)*(2-sqrt(2)) + Z(j,j)*(2+sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(6, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) + sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(7, i, j) = -2*trZ + (Z(i,i) + Z(j,j))*(2+sqrt(2)) - sqrt(2)*(Z(i,j) - Z(j,i));
                    scores_nuclear(8, i, j) = -2*trZ + Z(i,i)*(2+sqrt(2)) + Z(j,j)*(2-sqrt(2)) + sqrt(2)*(Z(i,j) + Z(j,i));
                    scores_nuclear(9, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) + Z(j,i));
                    scores_nuclear(10, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) - Z(j,i));
                    scores_nuclear(11, i, j) = -2*trZ + 4*Z(j,j);
                    scores_nuclear(12, i, j) = -2*trZ + 4*Z(i,i);
                    scores_nuclear(13, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) + Z(i,j) + Z(j,i));
%                     scores_nuclear(14, i, j) = -2*trZ + 2*(Z(i,i) + Z(j,j) - Z(i,j) - Z(j,i));
                    scores_nuclear(14, i, j) = inf;
                    scores_nuclear(15, i, j) = -2*trZ + 4*(Z(i,i) + Z(j,j));
                end
            end
        end
        
        vals = zeros(NN, 1);
        inds = zeros(NN,1);
        for kkk = 1:NN
            [val, index_nuc] = min(vec(squeeze(scores_nuclear(kkk,:,:))));
            vals(kkk) = val;
            inds(kkk) = index_nuc;
        end
        [min_val, min_ind] = min(vals);

        index_nuc = inds(min_ind);
        GG = squeeze(Gs(min_ind, :, :));

        [i_nuc, j_nuc] = ind2sub([n n], index_nuc);
        
        positions(1, kk) = i_nuc;
        positions(2, kk) = j_nuc;
        values(:, kk) = vec(GG);
        
        Z = applyGTransformOnRightTransp(Z, positions(1, kk), positions(2, kk), values(:, kk));
    end
    
    P = Data;
    for h = m:-1:1
        P = applyGTransformOnLeftTransp(P, positions(1, h), positions(2, h), values(:, h));
    end
    X = omp_forortho(P, k0);

    UX = X;
    for h = 1:m
        UX = applyGTransformOnLeft(UX, positions(1, h), positions(2, h), values(:, h));
    end
    err(k) = norm(Data-UX, 'fro')^2/norm(Data, 'fro')^2*100;
    
    if (k >= 3)
        if (abs( err(k) - err(k-1) ) <= 10e-6)
            if (NN >= 15)
                break;
            else
                NN = 15;
                err(k) = err(1);
            end
        end
    end
end

% explicit dictionary
U = eye(n);
for h = 1:m
    U = applyGTransformOnLeft(U, positions(1, h), positions(2, h), values(:, h));
end
tus = toc;
