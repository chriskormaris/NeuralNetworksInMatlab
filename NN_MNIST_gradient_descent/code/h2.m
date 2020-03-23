function [ h, grad_h ] = h2( a )
% h2: tanh function
%ACTIVATION_FUNCTION Summary of this function goes here
% log(1 + exp(a)) = log(exp(0) + exp(a)) = 0 + log(exp(-0) + exp(a-0))

% h
h = (exp(a) + exp(-a)) ./ (exp(a) - exp(-a));

% compute the gradient of: h(a) = (exp(a) - exp(-a)) / (exp(-a) - exp(a))
% h'(a) = - ( (1/(exp(a)) + exp(a))^2 / (-1/(exp(a)) + exp(a))^2 ) + 1
% h'(a) = - 4 * exp(2*a) / (exp(2*a) - 1)^2
% ALTERNATIVE: h'(a) = 1 - h(a)^2
if nargout > 1
    grad_h = 1 - h2(a) .^ 2;
end

end

