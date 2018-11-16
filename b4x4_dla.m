function [U, X, positions, values, tus, err] = b4x4_dla(Data, k0, m)
tic;

[n, ~] = size(Data);

if (n > 100)
    error('Don''t be brave ... do not run this with n > 100');
end

if (m > 150)
    error('Don''t be brave ... do not run this with m > 150');
end

% initialize X with Q-DLA
[U, ~, ~] = svd(Data, 'econ');
X = omp_forortho(U'*Data, k0);
err = norm(Data - U*X, 'fro')^2;
k = 0;
while(1)
    k = k + 1;
    
    [Usvd, ~, Vsvd] = svd(Data*X');
    U = Usvd*Vsvd';
    X = omp_forortho(U'*Data, k0);
    
    err = [err norm(Data - U*X, 'fro')^2];
    
    if (k > 1)
        if (abs(err(k) - err(k-1)) < 10e-2)
            break;
        end
    end
end

positions = zeros(4, m);
values = zeros(16, m);

% load all 4x4 orthonormal binary transforms
solutions = [];
load('solutions_4x4.mat');

how_many = 768;

% the initialization of the binary 4x4 algoritm
Z = Data*X'; trZ = trace(Z);
W = X*X';
encoding = zeros(1, n*(n-1)*(n-2)*(n-3)/24);
scores_nuclear = zeros(how_many, n*(n-1)*(n-2)*(n-3)/24);
err = norm(Data - X, 'fro')^2/norm(Data, 'fro')^2*100;

for t = 1:how_many
    index = 0;
    for i = 1:n
        for j = i+1:n
            for r = j+1:n
                for p = r+1:n
                    index = index + 1;
                    if (t == 1)
                        encoding(index) = p+100*r+10000*j+1000000*i;
                    end
                    
                    % for us, this line always evaluates to zero, therefore
                    % we remove it from the calculations
                    % + (solutions(t, 1)^2 + solutions(t, 5)^2 + solutions(t, 9)^2 + solutions(t, 13)^2 - 1)*W(i,i) + (solutions(t, 2)^2 + solutions(t, 6)^2 + solutions(t, 10)^2 + solutions(t, 14)^2 - 1)*W(j,j) + (solutions(t, 3)^2 + solutions(t, 7)^2 + solutions(t, 11)^2 + solutions(t, 15)^2 - 1)*W(r,r) + (solutions(t, 4)^2 + solutions(t, 8)^2 + solutions(t, 12)^2 + solutions(t, 16)^2 - 1)*W(p,p) ...
                    scores_nuclear(t, index) = 0 ...
                            + 2*solutions(t, 1)*solutions(t, 2)*W(i,j) + 2*solutions(t, 1)*solutions(t, 3)*W(i,r) + 2*solutions(t, 1)*solutions(t, 4)*W(i,p) + 2*solutions(t, 2)*solutions(t, 3)*W(j,r) + 2*solutions(t, 2)*solutions(t, 4)*W(j,p) + 2*solutions(t, 3)*solutions(t, 4)*W(r,p) ...
                            - 2*(solutions(t, 1) - 1)*Z(i,i) - 2*solutions(t, 2)*Z(i,j) - 2*solutions(t, 3)*Z(i,r) - 2*solutions(t, 4)*Z(i,p) ...
                            + 2*solutions(t, 5)*solutions(t, 6)*W(i,j) + 2*solutions(t, 5)*solutions(t, 7)*W(i,r) + 2*solutions(t, 5)*solutions(t, 8)*W(i,p) + 2*solutions(t, 6)*solutions(t, 7)*W(j,r) + 2*solutions(t, 6)*solutions(t, 8)*W(j,p) + 2*solutions(t, 7)*solutions(t, 8)*W(r,p) ...
                            -2*solutions(t, 5)*Z(j,i) - 2*(solutions(t, 6) - 1)*Z(j,j) - 2*solutions(t, 7)*Z(j,r) - 2*solutions(t, 8)*Z(j,p) ...
                            + 2*solutions(t, 9)*solutions(t, 10)*W(i,j) + 2*solutions(t, 9)*solutions(t, 11)*W(i,r) + 2*solutions(t, 9)*solutions(t, 12)*W(i,p) + 2*solutions(t, 10)*solutions(t, 11)*W(j,r) + 2*solutions(t, 10)*solutions(t, 12)*W(j,p) + 2*solutions(t, 11)*solutions(t, 12)*W(r,p) ...
                            -2*solutions(t, 9)*Z(r,i) -2*solutions(t, 10)*Z(r,j) - 2*(solutions(t, 11) - 1)*Z(r,r) - 2*solutions(t, 12)*Z(r,p) ...
                            + 2*solutions(t, 13)*solutions(t, 14)*W(i,j) + 2*solutions(t, 13)*solutions(t, 15)*W(i,r) + 2*solutions(t, 13)*solutions(t, 16)*W(i,p) + 2*solutions(t, 14)*solutions(t, 15)*W(j,r) + 2*solutions(t, 14)*solutions(t, 16)*W(j,p) + 2*solutions(t, 15)*solutions(t, 16)*W(r,p) ...
                            -2*solutions(t, 13)*Z(p,i) -2*solutions(t, 14)*Z(p,j) -2*solutions(t, 15)*Z(p,r) - 2*(solutions(t, 16) - 1)*Z(p,p);
                end
            end
        end
    end
end

scores_nuclear = -2*trZ+scores_nuclear;

% initialize all components
workingX = X;
for kk = 1:m
    [vals, inds] = min(scores_nuclear, [], 2);
    [min_val, min_ind] = min(vals);
    
    index_nuc = encoding(inds(min_ind));
    GG = reshape(solutions(min_ind, :), 4, 4)';

    p_nuc = mod(index_nuc, 100);
    index_nuc = index_nuc - p_nuc;
    index_nuc = index_nuc/100;
     
    r_nuc = mod(index_nuc, 100);
    index_nuc = index_nuc - r_nuc;
    index_nuc = index_nuc/100;
     
    j_nuc = mod(index_nuc, 100);
    index_nuc = index_nuc - j_nuc;
    index_nuc = index_nuc/100;
     
    i_nuc = mod(index_nuc, 100);

    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    positions(3, kk) = r_nuc;
    positions(4, kk) = p_nuc;
    values(:, kk) = vec(GG);
    
    nucs = [i_nuc j_nuc r_nuc p_nuc];
    G = eye(n);
    G(nucs, nucs) = GG;
    workingX = G*workingX;
    
    old_trZ = trZ;
    Z = Z*G'; trZ = trace(Z);
    W = G*W*G';
    
    err = [err norm(Data - workingX, 'fro')^2/norm(Data, 'fro')^2*100];
    % display some progress
    kk
    err(end)
    
    save(['last error s = ' num2str(k0) ' init q-dla.mat'], 'err', 'positions', 'values');
    
    scores_nuclear = scores_nuclear+2*old_trZ;
    % update i
    for t = 1:how_many
        for i = nucs
            index = 0;
            for ii = 0:i-2
                index = index + max((n-1-ii)*(n-2-ii)*(n-3-ii)/6, 0);
            end

            for j = i+1:n
                for r = j+1:n
                    for p = r+1:n
                        index = index + 1;

                            scores_nuclear(t, index) = 0 ...
                                    + 2*solutions(t, 1)*solutions(t, 2)*W(i,j) + 2*solutions(t, 1)*solutions(t, 3)*W(i,r) + 2*solutions(t, 1)*solutions(t, 4)*W(i,p) + 2*solutions(t, 2)*solutions(t, 3)*W(j,r) + 2*solutions(t, 2)*solutions(t, 4)*W(j,p) + 2*solutions(t, 3)*solutions(t, 4)*W(r,p) ...
                                    - 2*(solutions(t, 1) - 1)*Z(i,i) - 2*solutions(t, 2)*Z(i,j) - 2*solutions(t, 3)*Z(i,r) - 2*solutions(t, 4)*Z(i,p) ...
                                    + 2*solutions(t, 5)*solutions(t, 6)*W(i,j) + 2*solutions(t, 5)*solutions(t, 7)*W(i,r) + 2*solutions(t, 5)*solutions(t, 8)*W(i,p) + 2*solutions(t, 6)*solutions(t, 7)*W(j,r) + 2*solutions(t, 6)*solutions(t, 8)*W(j,p) + 2*solutions(t, 7)*solutions(t, 8)*W(r,p) ...
                                    -2*solutions(t, 5)*Z(j,i) - 2*(solutions(t, 6) - 1)*Z(j,j) - 2*solutions(t, 7)*Z(j,r) - 2*solutions(t, 8)*Z(j,p) ...
                                    + 2*solutions(t, 9)*solutions(t, 10)*W(i,j) + 2*solutions(t, 9)*solutions(t, 11)*W(i,r) + 2*solutions(t, 9)*solutions(t, 12)*W(i,p) + 2*solutions(t, 10)*solutions(t, 11)*W(j,r) + 2*solutions(t, 10)*solutions(t, 12)*W(j,p) + 2*solutions(t, 11)*solutions(t, 12)*W(r,p) ...
                                    -2*solutions(t, 9)*Z(r,i) -2*solutions(t, 10)*Z(r,j) - 2*(solutions(t, 11) - 1)*Z(r,r) - 2*solutions(t, 12)*Z(r,p) ...
                                    + 2*solutions(t, 13)*solutions(t, 14)*W(i,j) + 2*solutions(t, 13)*solutions(t, 15)*W(i,r) + 2*solutions(t, 13)*solutions(t, 16)*W(i,p) + 2*solutions(t, 14)*solutions(t, 15)*W(j,r) + 2*solutions(t, 14)*solutions(t, 16)*W(j,p) + 2*solutions(t, 15)*solutions(t, 16)*W(r,p) ...
                                    -2*solutions(t, 13)*Z(p,i) -2*solutions(t, 14)*Z(p,j) -2*solutions(t, 15)*Z(p,r) - 2*(solutions(t, 16) - 1)*Z(p,p);

                    end
                end
            end
        end
    end
    % update j
    for t = 1:how_many
        for i = setdiff(1:n, nucs)
            what = find(nucs > i);
            for j = nucs(what)
                index = 0;
                for ii = 0:i-2
                    index = index + max((n-1-ii)*(n-2-ii)*(n-3-ii)/6, 0);
                end
                for jj = i:j-2
                    index = index + max((n-1-jj)*(n-2-jj)/2, 0);
                end

                for r = j+1:n
                    for p = r+1:n
                        index = index + 1;

                            scores_nuclear(t, index) = 0 ...
                                    + 2*solutions(t, 1)*solutions(t, 2)*W(i,j) + 2*solutions(t, 1)*solutions(t, 3)*W(i,r) + 2*solutions(t, 1)*solutions(t, 4)*W(i,p) + 2*solutions(t, 2)*solutions(t, 3)*W(j,r) + 2*solutions(t, 2)*solutions(t, 4)*W(j,p) + 2*solutions(t, 3)*solutions(t, 4)*W(r,p) ...
                                    - 2*(solutions(t, 1) - 1)*Z(i,i) - 2*solutions(t, 2)*Z(i,j) - 2*solutions(t, 3)*Z(i,r) - 2*solutions(t, 4)*Z(i,p) ...
                                    + 2*solutions(t, 5)*solutions(t, 6)*W(i,j) + 2*solutions(t, 5)*solutions(t, 7)*W(i,r) + 2*solutions(t, 5)*solutions(t, 8)*W(i,p) + 2*solutions(t, 6)*solutions(t, 7)*W(j,r) + 2*solutions(t, 6)*solutions(t, 8)*W(j,p) + 2*solutions(t, 7)*solutions(t, 8)*W(r,p) ...
                                    -2*solutions(t, 5)*Z(j,i) - 2*(solutions(t, 6) - 1)*Z(j,j) - 2*solutions(t, 7)*Z(j,r) - 2*solutions(t, 8)*Z(j,p) ...
                                    + 2*solutions(t, 9)*solutions(t, 10)*W(i,j) + 2*solutions(t, 9)*solutions(t, 11)*W(i,r) + 2*solutions(t, 9)*solutions(t, 12)*W(i,p) + 2*solutions(t, 10)*solutions(t, 11)*W(j,r) + 2*solutions(t, 10)*solutions(t, 12)*W(j,p) + 2*solutions(t, 11)*solutions(t, 12)*W(r,p) ...
                                    -2*solutions(t, 9)*Z(r,i) -2*solutions(t, 10)*Z(r,j) - 2*(solutions(t, 11) - 1)*Z(r,r) - 2*solutions(t, 12)*Z(r,p) ...
                                    + 2*solutions(t, 13)*solutions(t, 14)*W(i,j) + 2*solutions(t, 13)*solutions(t, 15)*W(i,r) + 2*solutions(t, 13)*solutions(t, 16)*W(i,p) + 2*solutions(t, 14)*solutions(t, 15)*W(j,r) + 2*solutions(t, 14)*solutions(t, 16)*W(j,p) + 2*solutions(t, 15)*solutions(t, 16)*W(r,p) ...
                                    -2*solutions(t, 13)*Z(p,i) -2*solutions(t, 14)*Z(p,j) -2*solutions(t, 15)*Z(p,r) - 2*(solutions(t, 16) - 1)*Z(p,p);

                    end
                end
            end
        end
    end
    % update r
    for t = 1:how_many
        for i = setdiff(1:n, nucs)
            for j = i+1:n
                what = find(nucs > j);
                for r = nucs(what)
                    index = 0;
                    for ii = 0:i-2
                        index = index + max((n-1-ii)*(n-2-ii)*(n-3-ii)/6, 0);
                    end
                    for jj = i:j-2
                        index = index + max((n-1-jj)*(n-2-jj)/2, 0);
                    end
                    for rr = j:r-2
                        index = index + max((n-1-rr), 0);
                    end
                    
                    for p = r+1:n
                        index = index + 1;

                            scores_nuclear(t, index) = 0 ...
                                    + 2*solutions(t, 1)*solutions(t, 2)*W(i,j) + 2*solutions(t, 1)*solutions(t, 3)*W(i,r) + 2*solutions(t, 1)*solutions(t, 4)*W(i,p) + 2*solutions(t, 2)*solutions(t, 3)*W(j,r) + 2*solutions(t, 2)*solutions(t, 4)*W(j,p) + 2*solutions(t, 3)*solutions(t, 4)*W(r,p) ...
                                    - 2*(solutions(t, 1) - 1)*Z(i,i) - 2*solutions(t, 2)*Z(i,j) - 2*solutions(t, 3)*Z(i,r) - 2*solutions(t, 4)*Z(i,p) ...
                                    + 2*solutions(t, 5)*solutions(t, 6)*W(i,j) + 2*solutions(t, 5)*solutions(t, 7)*W(i,r) + 2*solutions(t, 5)*solutions(t, 8)*W(i,p) + 2*solutions(t, 6)*solutions(t, 7)*W(j,r) + 2*solutions(t, 6)*solutions(t, 8)*W(j,p) + 2*solutions(t, 7)*solutions(t, 8)*W(r,p) ...
                                    -2*solutions(t, 5)*Z(j,i) - 2*(solutions(t, 6) - 1)*Z(j,j) - 2*solutions(t, 7)*Z(j,r) - 2*solutions(t, 8)*Z(j,p) ...
                                    + 2*solutions(t, 9)*solutions(t, 10)*W(i,j) + 2*solutions(t, 9)*solutions(t, 11)*W(i,r) + 2*solutions(t, 9)*solutions(t, 12)*W(i,p) + 2*solutions(t, 10)*solutions(t, 11)*W(j,r) + 2*solutions(t, 10)*solutions(t, 12)*W(j,p) + 2*solutions(t, 11)*solutions(t, 12)*W(r,p) ...
                                    -2*solutions(t, 9)*Z(r,i) -2*solutions(t, 10)*Z(r,j) - 2*(solutions(t, 11) - 1)*Z(r,r) - 2*solutions(t, 12)*Z(r,p) ...
                                    + 2*solutions(t, 13)*solutions(t, 14)*W(i,j) + 2*solutions(t, 13)*solutions(t, 15)*W(i,r) + 2*solutions(t, 13)*solutions(t, 16)*W(i,p) + 2*solutions(t, 14)*solutions(t, 15)*W(j,r) + 2*solutions(t, 14)*solutions(t, 16)*W(j,p) + 2*solutions(t, 15)*solutions(t, 16)*W(r,p) ...
                                    -2*solutions(t, 13)*Z(p,i) -2*solutions(t, 14)*Z(p,j) -2*solutions(t, 15)*Z(p,r) - 2*(solutions(t, 16) - 1)*Z(p,p);

                    end
                end
            end
        end
    end
    % update the p
    for t = 1:how_many
        for i = setdiff(1:n, nucs)
            for j = i+1:n
                for r = j+1:n
                    what = find(nucs > r);
                    for p = nucs(what)
                        index = 0;
                        for ii = 0:i-2
                            index = index + max((n-1-ii)*(n-2-ii)*(n-3-ii)/6, 0);
                        end
                        for jj = i:j-2
                            index = index + max((n-1-jj)*(n-2-jj)/2, 0);
                        end
                        for rr = j:r-2
                            index = index + max((n-1-rr), 0);
                        end
                        index = index + (p-r);

                            scores_nuclear(t, index) = 0 ...
                                    + 2*solutions(t, 1)*solutions(t, 2)*W(i,j) + 2*solutions(t, 1)*solutions(t, 3)*W(i,r) + 2*solutions(t, 1)*solutions(t, 4)*W(i,p) + 2*solutions(t, 2)*solutions(t, 3)*W(j,r) + 2*solutions(t, 2)*solutions(t, 4)*W(j,p) + 2*solutions(t, 3)*solutions(t, 4)*W(r,p) ...
                                    - 2*(solutions(t, 1) - 1)*Z(i,i) - 2*solutions(t, 2)*Z(i,j) - 2*solutions(t, 3)*Z(i,r) - 2*solutions(t, 4)*Z(i,p) ...
                                    + 2*solutions(t, 5)*solutions(t, 6)*W(i,j) + 2*solutions(t, 5)*solutions(t, 7)*W(i,r) + 2*solutions(t, 5)*solutions(t, 8)*W(i,p) + 2*solutions(t, 6)*solutions(t, 7)*W(j,r) + 2*solutions(t, 6)*solutions(t, 8)*W(j,p) + 2*solutions(t, 7)*solutions(t, 8)*W(r,p) ...
                                    -2*solutions(t, 5)*Z(j,i) - 2*(solutions(t, 6) - 1)*Z(j,j) - 2*solutions(t, 7)*Z(j,r) - 2*solutions(t, 8)*Z(j,p) ...
                                    + 2*solutions(t, 9)*solutions(t, 10)*W(i,j) + 2*solutions(t, 9)*solutions(t, 11)*W(i,r) + 2*solutions(t, 9)*solutions(t, 12)*W(i,p) + 2*solutions(t, 10)*solutions(t, 11)*W(j,r) + 2*solutions(t, 10)*solutions(t, 12)*W(j,p) + 2*solutions(t, 11)*solutions(t, 12)*W(r,p) ...
                                    -2*solutions(t, 9)*Z(r,i) -2*solutions(t, 10)*Z(r,j) - 2*(solutions(t, 11) - 1)*Z(r,r) - 2*solutions(t, 12)*Z(r,p) ...
                                    + 2*solutions(t, 13)*solutions(t, 14)*W(i,j) + 2*solutions(t, 13)*solutions(t, 15)*W(i,r) + 2*solutions(t, 13)*solutions(t, 16)*W(i,p) + 2*solutions(t, 14)*solutions(t, 15)*W(j,r) + 2*solutions(t, 14)*solutions(t, 16)*W(j,p) + 2*solutions(t, 15)*solutions(t, 16)*W(r,p) ...
                                    -2*solutions(t, 13)*Z(p,i) -2*solutions(t, 14)*Z(p,j) -2*solutions(t, 15)*Z(p,r) - 2*(solutions(t, 16) - 1)*Z(p,p);

                    end
                end
            end
        end
    end
    
    scores_nuclear = -2*trZ+scores_nuclear;
end

P = Data;
for h = m:-1:1
	nucs = [positions(1, h) positions(2, h) positions(3, h) positions(4, h)];
	G = eye(n);
	G(nucs, nucs) = reshape(values(:,h), 4, 4)';
	P = G*P;
end
X = omp_forortho(P, k0);

workingX = X;
for h = 1:m
	nucs = [positions(1, h) positions(2, h) positions(3, h) positions(4, h)];
	G = eye(n);
	G(nucs, nucs) = reshape(values(:,h), 4, 4);
	workingX = G*workingX;
end

err = [err norm(Data - workingX, 'fro')^2/norm(Data, 'fro')^2*100];

tus = toc;
