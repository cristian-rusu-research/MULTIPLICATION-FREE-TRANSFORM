function [estimate, signs, powers] = nearestpow2sum(val, k)

if (k == inf)
    estimate = val; signs = []; powers = [];
    return;
end

signs = zeros(k, 1);
powers = -inf(k,1);

estimate = 0;
val_org = val;
res = val;
for i = 1:k
    signs(i) = sign(res);
    
    powers(i) = nearestpow2(abs(res));
    estimate = estimate + signs(i)*2^(powers(i));
    res = val_org - estimate;
    
    if (abs(res) <= 10e-7)
        break;
    end
end