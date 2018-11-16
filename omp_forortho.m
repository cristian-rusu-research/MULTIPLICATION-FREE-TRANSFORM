function X = omp_forortho(Data, k)
[n, N] = size(Data);
[~, ind] = sort(abs(Data), 'descend');

X = zeros(n, N);
for i = 1:N
    X(ind(1:k, i), i) = Data(ind(1:k, i), i);
end