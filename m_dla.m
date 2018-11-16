function [U, X, S, positions, values, tus, err] = m_dla(Data, k0, stages)

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

err = [];
% number of iterations
for iii = 1:10
    for kkk = 1:stages
        Z = Data*workingX'; trZ = trace(Z);
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

        score_total = inf(n,n);
        score_index = zeros(n,n);
        edgeData = zeros((n^2-n)/2, 3); index = 0;
        for i = 1:n
            for j = i+1:n
                [score_total(i,j), score_index(i,j)] = min(vec(squeeze(scores_nuclear(:,i,j))));
                index = index + 1;
                edgeData(index, :) = [i j -score_total(i,j)];
            end
        end

        result = maxWeightMatching(edgeData, true);

        Gtotal = eye(n);
        old_kk = 0;
        for kk = 1:n
            if isinf(result(kk))
                continue;
            end

            old_kk = old_kk + 1;

            i_nuc = kk;
            j_nuc = result(i_nuc);

            result(kk) = inf;
            result(j_nuc) = inf;

            GG = squeeze(Gs(score_index(i_nuc, j_nuc), :, :));

            positions(1, (kkk-1)*m+old_kk) = i_nuc;
            positions(2, (kkk-1)*m+old_kk) = j_nuc;
            values(:, (kkk-1)*m+old_kk) = vec(GG);

            workingX = applyGTransformOnLeft(workingX, i_nuc, j_nuc, vec(GG));

            Gtotal([i_nuc j_nuc], [i_nuc j_nuc]) = GG;
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
    err = [err norm(Data-UX, 'fro')^2/norm(Data, 'fro')^2*100];
    workingX = X;
end

% explicit dictionary
U = eye(n);
for h = 1:stages*m
    U = applyGTransformOnLeft(U, positions(1, h), positions(2, h), values(:, h));
end

tus = toc;
