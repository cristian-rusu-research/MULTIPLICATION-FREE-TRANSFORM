function [U, D, X, positions, values, tus, err] = s_dla(Data, k0, m, bits, positions, values)

tic;
[n, ~] = size(Data);
[U, ~, ~] = svd(Data, 'econ');
% [U, ~] = qr(randn(n));
X = omp_forortho(U'*Data, k0);

% number of iterations
K = 15;
D = ones(n, 1);

if (~exist('positions', 'var') && ~exist('values', 'var'))
    positions = zeros(2, m);
    values = zeros(4, m);

    Z = Data*X';
    W = X*X';
    scores = inf(n);
    for i = 1:n
        for j = i+1:n
            scores(i, j) = - (Z(j,i) - W(i,j))^2/W(i,i);
            scores(j, i) = - (Z(i,j) - W(j,i))^2/W(j,j);
        end
    end
    scores_scale = inf(n,1);
    for i = 1:n
        gamma = Z(i,i)/W(i,i);
        if (bits == inf)
            alpha = 1;
        else
            alpha = 1/abs(gamma)*2^round(log2(abs(gamma)));
        end

        scores_scale(i) = - 2*Z(i,i)*(alpha*gamma-1) + W(i,i)*( (alpha*gamma)^2 - 1);
    end

    for kk = 1:m
        [val1, index_nuc] = min(scores(:));
        [val2, index_scale] = min(scores_scale);

        if (val1 < val2)
            [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

            if (i_nuc < j_nuc)
                positions(1, kk) = i_nuc;
                positions(2, kk) = j_nuc;
                GG = [1 0; nearestpow2sum((Z(j_nuc,i_nuc)-W(i_nuc,j_nuc))/W(i_nuc,i_nuc), bits) 1];
            else
                positions(1, kk) = j_nuc;
                positions(2, kk) = i_nuc;
                i_nuc = positions(1, kk); j_nuc = positions(2, kk);
                GG = [1 nearestpow2sum((Z(i_nuc,j_nuc)-W(j_nuc,i_nuc))/W(j_nuc,j_nuc), bits); 0 1];
            end

            values(:, kk) = vec(GG);

            Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, values(:, kk));
            W = applyGTransformOnLeft(W, i_nuc, j_nuc, values(:, kk));
            W = applyGTransformOnRightTransp(W, i_nuc, j_nuc, values(:, kk));

            for i = [i_nuc j_nuc]
                for j = i+1:n
                    scores(i, j) = -(Z(j,i) - W(i,j))^2/W(i,i);
                    scores(j, i) = -(Z(i,j) - W(j,i))^2/W(j,j);
                end
            end

            for j = [i_nuc j_nuc]
                for i = 1:j-1
                    scores(i, j) = -(Z(j,i) - W(i,j))^2/W(i,i);
                    scores(j, i) = -(Z(i,j) - W(j,i))^2/W(j,j);
                end
            end

            for i = [i_nuc j_nuc]
                gamma = Z(i,i)/W(i,i);
                if (bits == inf)
                    alpha = 1;
                else
                    alpha = 1/abs(gamma)*2^round(log2(abs(gamma)));
                end

                scores_scale(i) = -2*Z(i,i)*(alpha*gamma-1) + W(i,i)*( (alpha*gamma)^2 - 1);
            end
        else
            positions(1, kk) = index_scale;
            positions(2, kk) = index_scale;

            gamma = Z(index_scale,index_scale)/W(index_scale,index_scale);
            if (bits == inf)
                alpha = 1;
            else
                alpha = 1/abs(gamma)*2^round(log2(abs(gamma)));
            end

            values(:, kk) = vec([alpha*gamma 0 0 0]);

            Z(:, index_scale) = Z(:, index_scale)*values(1, kk);
            W(:, index_scale) = W(:, index_scale)*values(1, kk);
            W(index_scale, :) = values(1, kk)*W(index_scale, :);

            for i = [index_scale]
                for j = i+1:n
                    scores(i, j) = -(Z(j,i) - W(i,j))^2/W(i,i);
                    scores(j, i) = -(Z(i,j) - W(j,i))^2/W(j,j);
                end
            end

            for j = [index_scale]
                for i = 1:j-1
                    scores(i, j) = -(Z(j,i) - W(i,j))^2/W(i,i);
                    scores(j, i) = -(Z(i,j) - W(j,i))^2/W(j,j);
                end
            end

            for i = [index_scale]
                gamma = Z(i,i)/W(i,i);
                if (bits == inf)
                    alpha = 1;
                else
                    alpha = 1/abs(gamma)*2^round(log2(abs(gamma)));
                end

                scores_scale(i) = -2*Z(i,i)*(alpha*gamma-1) + W(i,i)*( (alpha*gamma)^2 - 1);
            end
        end

    end
end

S = eye(n);
for h = m:-1:1
    if (positions(1, h) == positions(2, h))
        S(:, positions(1, h)) = S(:, positions(1, h))*values(1,h);
    else
        S = applyGTransformOnRight(S, positions(1, h), positions(2, h), values(:, h));
    end
end

for jj = 1:n
    D(jj) = 1/norm(S(:,jj));
end
X = omp((S*diag(D))'*Data, (S*diag(D))'*(S*diag(D)), k0);

err = zeros(K, 1);
y = vec(Data);
for k = 1:K
    Xk = diag(D)*X;

    Ak = eye(n);
    for h = m:-1:1
        if (positions(1, h) == positions(2, h))
            Ak(:, positions(1, h)) = Ak(:, positions(1, h))*values(1,h);
        else
            Ak = applyGTransformOnRight(Ak, positions(1, h), positions(2, h), values(:, h));
        end
    end
    
    for kk = 1:m
        
        if (kk > 1)
            if (positions(1, kk-1) == positions(2, kk-1))
                Xk(positions(1, kk-1), :) = values(1, kk-1)*Xk(positions(1, kk-1), :);
            else
                Xk = applyGTransformOnLeft(Xk, positions(1, kk-1), positions(2, kk-1), values(:, kk-1));
            end
        end
        
        if (positions(1, kk) == positions(2, kk))
            Ak(:, positions(1, kk)) = Ak(:, positions(1, kk))/values(1,kk);
        else
            vals = inv(reshape(values(:, kk), 2, 2));
            Ak = applyGTransformOnRight(Ak, positions(1, kk), positions(2, kk), vec(vals));
        end
        
        f = y;
        for i = 1:n
            f = f - kron(Xk(i, :)', Ak(:, i));
        end
        
        norms2_Xk = norms(Xk, 2, 2).^2;
        norms2_Ak = norms(Ak, 2, 1).^2;
        
        for i = 1:n
            scores_scale(i) = -(f'*kron(Xk(i, :)', Ak(:, i)))^2/norms2_Xk(i)/norms2_Ak(i);
        end
        
        for i = 1:n
            for j = i+1:n
                scores(i, j) = -(f'*kron(Xk(i, :)', Ak(:, j)))^2/norms2_Xk(i)/norms2_Ak(j);
                scores(j, i) = -(f'*kron(Xk(j, :)', Ak(:, i)))^2/norms2_Xk(j)/norms2_Ak(i);
            end
        end
        
        [val1, index_nuc] = min(scores(:));
        [val2, index_scale] = min(scores_scale);

        if (val1 < val2)
            [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

            if (i_nuc < j_nuc)
                positions(1, kk) = i_nuc;
                positions(2, kk) = j_nuc;
                GG = [1 0; nearestpow2sum(f'*kron(Xk(i_nuc, :)', Ak(:, j_nuc))/norms2_Xk(i_nuc)/norms2_Ak(j_nuc), bits) 1];
            else
                positions(1, kk) = j_nuc;
                positions(2, kk) = i_nuc;
                i_nuc = positions(1, kk); j_nuc = positions(2, kk);
                GG = [1 nearestpow2sum(f'*kron(Xk(j_nuc, :)', Ak(:, i_nuc))/norms2_Xk(j_nuc)/norms2_Ak(i_nuc), bits); 0 1];
            end

            values(:, kk) = vec(GG);
        else
            positions(1, kk) = index_scale;
            positions(2, kk) = index_scale;

            gamma = f'*kron(Xk(index_scale, :)', Ak(:, index_scale))/norms2_Xk(index_scale)/norms2_Ak(index_scale)+1;
            if (bits == inf)
                alpha = 1;
            else
                alpha = 1/abs(gamma)*2^round(log2(abs(gamma)));
            end

            values(:, kk) = vec([alpha*gamma 0 0 0]);
        end

    end
    
    S = eye(n);
    for h = m:-1:1
        if (positions(1, h) == positions(2, h))
            S(:, positions(1, h)) = S(:, positions(1, h))*values(1,h);
        else
            S = applyGTransformOnRight(S, positions(1, h), positions(2, h), values(:, h));
        end
    end

    for jj = 1:n
        D(jj) = 1/norm(S(:,jj));
    end
    X = omp((S*diag(D))'*Data, (S*diag(D))'*(S*diag(D)), k0);

    err(k) = norm(Data-S*diag(D)*X, 'fro')^2/norm(Data, 'fro')^2*100;
end

tus = toc;
