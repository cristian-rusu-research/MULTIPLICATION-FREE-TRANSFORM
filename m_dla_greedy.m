function [U, X, S, positions, values, tus, err] = m_dla_greedy(Data, k0, stages)
tic;

Gs = zeros(8,2,2);
Gs(1,:,:) = 1/sqrt(2)*[-1 1; 1 1]; Gs(2,:,:) = 1/sqrt(2)*[1 1; -1 1];
Gs(3,:,:) = 1/sqrt(2)*[1 -1; 1 1]; Gs(4,:,:) = 1/sqrt(2)*[1 1; 1 -1];
Gs(5,:,:) = -1/sqrt(2)*[-1 1; 1 1]; Gs(6,:,:) = -1/sqrt(2)*[1 1; -1 1];
Gs(7,:,:) = -1/sqrt(2)*[1 -1; 1 1]; Gs(8,:,:) = -1/sqrt(2)*[1 1; 1 -1];

[n, ~] = size(Data);
[U, ~, ~] = svd(Data, 'econ');
workingX = omp_forortho(U'*Data, k0);

m = n/2;

positions = zeros(2, stages*m);
values = zeros(4, stages*m);

S = cell(stages, 1);

Z = Data*workingX';
for kkk = 1:stages
    trZ = trace(Z);
    scores_nuclear = inf(8,n,n);
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
        end
    end

    Gtotal = eye(n);
    for kk = 1:m
        vals = zeros(8, 1);
        inds = zeros(8,1);
        for kkkk = 1:8
            [val, index_nuc] = min(vec(squeeze(scores_nuclear(kkkk,:,:))));
            vals(kkkk) = val;
            inds(kkkk) = index_nuc;
        end
        [min_val, min_ind] = min(vals);

        index_nuc = inds(min_ind);
        GG = squeeze(Gs(min_ind, :, :));
        
        [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

        positions(1, (kkk-1)*m+kk) = i_nuc;
        positions(2, (kkk-1)*m+kk) = j_nuc;
        values(:, (kkk-1)*m+kk) = vec(GG);

        Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, vec(GG));
        
        Gtotal([i_nuc j_nuc], [i_nuc j_nuc]) = GG;
        
        for i = [i_nuc j_nuc]
            for j = i+1:n
                scores_nuclear(:, i, j) = inf;
            end
        end

        for j = [i_nuc j_nuc]
            for i = 1:j-1
                scores_nuclear(:, i, j) = inf;
            end
        end
    end
    S{kkk} = sparse(Gtotal);
end

P = Data;
for h = stages*m:-1:1
    P = applyGTransformOnLeftTransp(P, positions(1, h), positions(2, h), values(:, h));
end
X = omp_forortho(P, k0);

UX = X;
for h = 1:stages*m
    UX = applyGTransformOnLeft(UX, positions(1, h), positions(2, h), values(:, h));
end
err = norm(Data-UX, 'fro')^2/norm(Data, 'fro')^2*100;

% explicit dictionary
U = eye(n);
for h = 1:stages*m
    U = applyGTransformOnLeft(U, positions(1, h), positions(2, h), values(:, h));
end

tus = toc;
